#!/usr/bin/env python3
"""Score frozen FMs against the public STW rigid polar, without CFD/training.

This diagnostic is not a matched-physics benchmark or a completed MDO result.
"""
import argparse
import csv
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from mdo_demo.io import read_json, write_json, sha256_file
from mdo_demo.stw_benchmark import make_stw_surface, polar_scores, UPSTREAM_COMMIT
from mdo_demo.input_contract import encode_condition


def prepare(assets, output):
    t0 = time.perf_counter()
    manifest = read_json(assets / 'source_manifest.json')
    if manifest['commit'] != UPSTREAM_COMMIT:
        raise ValueError('Wrong upstream version')
    for f in manifest['files']:
        if sha256_file(assets/f['path']) != f['sha256']:
            raise ValueError('Upstream asset hash mismatch')
    geometry_path = assets/'STW-Files/geometry/wing.dat'
    reference_path = assets/'STW-Files/ReferenceResults/UM/BenchmarkAnalyses/Aero/RigidPolarResults.csv'
    reference = np.genfromtxt(reference_path, delimiter=',', names=True)
    if len(reference) != 21 or not np.allclose(reference['aoa'], np.arange(21)*.25):
        raise ValueError('Expected all 21 public STW conditions')
    geometry, audit = make_stw_surface(geometry_path)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'input.npz', **geometry,
                        alpha=reference['aoa'], reference_CL=reference['cl'], reference_CD=reference['cd'])
    audit.update(upstream_commit=UPSTREAM_COMMIT, preparation_seconds=time.perf_counter()-t0,
                 geometry_sha256=sha256_file(geometry_path), reference_sha256=sha256_file(reference_path),
                 input_sha256=sha256_file(output/'input.npz'),
                 reference_url=f'https://github.com/MDOBenchmarks/MDOAeroelasticBenchmark/blob/{UPSTREAM_COMMIT}/STW-Files/ReferenceResults/UM/BenchmarkAnalyses/Aero/RigidPolarResults.csv',
                 n_conditions=21, selected_all_rows=True, training_on_STW=False,
                 error_bars='No statistical accuracy CI from one geometry; timing technical repetitions only')
    write_json(output/'input_audit.json', audit)
    print('Prepared all 21 public cases; exact_physics_match=False, mdo_score_eligible=False')


def evaluate(project, input_dir, output):
    import fcntl
    import torch
    from mdo_demo.aerotransformer import AeroTransformerPredictor
    audit = read_json(input_dir/'input_audit.json')
    if sha256_file(input_dir/'input.npz') != audit['input_sha256']:
        raise ValueError('Input data hash mismatch')
    with np.load(input_dir/'input.npz', allow_pickle=False) as f:
        arrays = {k: np.array(f[k], copy=True) for k in f.files}
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    checkpoints = {'pretrained': project/'models/AeroTransformer/ATsurf_L',
                   'adapted': Path((project/'manifests/model_pilot_run.path').read_text().strip())/'adapted_checkpoint'}
    report = {'schema': 'stw_public_polar_transfer_v1', 'status': 'running', 'input_audit': audit,
              'cfd_executed': False, 'new_training': False, 'mdo_score_eligible': False,
              'models': {}, 'code_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                   [Path(__file__), ROOT/'src/mdo_demo/stw_benchmark.py', ROOT/'src/mdo_demo/aerotransformer.py']}}
    with (project/'manifests/compute.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        load = os.getloadavg()
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        ram = int(mem['MemAvailable'].split()[0])/1024**2
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
        space = os.statvfs(project).f_bavail*os.statvfs(project).f_frsize/1024**3
        if load[0]>24 or ram<8 or space<3 or int(gpu.split(',')[-1].split()[0])>30:
            raise RuntimeError('Shared resources busy; no evaluation started')
        output.mkdir(parents=True, exist_ok=False)
        report['hardware'] = {'cpu_load_start': load, 'available_ram_GiB': ram, 'free_disk_GiB': space,
                               'gpu': gpu, 'torch_threads': 1, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                               'shared_resources': True}
        run_t0 = time.perf_counter()
        for name, checkpoint in checkpoints.items():
            model = AeroTransformerPredictor(checkpoint, 'cuda')
            base = {k: arrays[k] for k in ['original_geometry', 'geometry', 'ref_area']}
            base['condition'] = encode_condition(alpha_deg=arrays['alpha'][0], mach=.77)
            tw = time.perf_counter()
            for _ in range(10): model.predict(base)
            warmup_seconds = time.perf_counter()-tw
            rows, fields = [], []
            stage_t0 = time.perf_counter()
            for i, alpha in enumerate(arrays['alpha']):
                sample = dict(base, condition=encode_condition(alpha_deg=alpha, mach=.77))
                reps = []
                values = []
                for rep in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    prediction = model.predict(sample)
                    torch.cuda.synchronize()
                    reps.append({'wall_seconds': time.perf_counter()-started, **prediction['timings']})
                    values.append([prediction['coefficients'][k] for k in ['CL', 'CD']])
                if not np.allclose(values, values[0], rtol=1e-6, atol=1e-8):
                    raise ValueError('Unexpected prediction instability')
                cl, cd = values[0]
                ref_cl, ref_cd = float(arrays['reference_CL'][i]), float(arrays['reference_CD'][i])
                row = {'model': name, 'alpha_deg': float(alpha), 'Mach': .77,
                       'CL_reference': ref_cl, 'CD_reference': ref_cd,
                       'CL_prediction': cl, 'CD_prediction': cd,
                       'CL_difference': cl-ref_cl, 'CD_difference_counts': 1e4*(cd-ref_cd),
                       'prediction_plus_integration_s': float(np.median([r['wall_seconds'] for r in reps])),
                       'timing_repetitions': reps, 'alpha_in_training_range': bool(2<=alpha<=12)}
                rows.append(row); fields.append(prediction['fields'])
            reference = np.column_stack([arrays['reference_CL'], arrays['reference_CD']])
            predicted = np.array([[r['CL_prediction'], r['CD_prediction']] for r in rows])
            in_range = arrays['alpha']>=2
            times = np.array([r['wall_seconds'] for row in rows for r in row['timing_repetitions']])
            report['models'][name] = {'all_21': polar_scores(predicted, reference),
                 'alpha_2_to_5_only': polar_scores(predicted[in_range], reference[in_range]),
                 'load_seconds': model.load_seconds, 'warmup_seconds_10_calls': warmup_seconds,
                 'evaluation_wall_seconds_63_calls': time.perf_counter()-stage_t0,
                 'prediction_plus_integration_seconds_median': float(np.median(times)),
                 'prediction_plus_integration_seconds_IQR': np.percentile(times, [25, 75]).tolist(),
                 'timing_scope': 'Batch one; device synchronized; prepared surface to fields and CL/CD. Excludes preparation/loading/warmup/disk writes/structure/optimization',
                 'model_provenance': model.provenance, 'rows': rows}
            np.savez_compressed(output/f'{name}_fields.npz', fields=np.asarray(fields),
                                alpha=arrays['alpha'], predicted_CL_CD=predicted)
            with (output/f'{name}_scores.csv').open('w', newline='') as f:
                keys = [k for k in rows[0] if k != 'timing_repetitions']
                writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
                writer.writerows([{k: row[k] for k in keys} for row in rows])
            write_json(output/'summary.json', report)
            print(name, report['models'][name]['all_21'], flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
        report.update(status='completed', complete_two_model_wall_seconds=time.perf_counter()-run_t0)
        report['hardware']['cpu_load_end'] = os.getloadavg()
        write_json(output/'summary.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--assets', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('evaluate'); a.add_argument('--project', type=Path, required=True); a.add_argument('--input', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'prepare': prepare(args.assets.resolve(), args.output.resolve())
    else: evaluate(args.project.resolve(), args.input.resolve(), args.output.resolve())


if __name__ == '__main__': main()
