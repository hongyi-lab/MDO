#!/usr/bin/env python3
"""FM-only shape optimization with actual TACS; never calls a CFD solver."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','request','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/fm_shape_design_v1.json')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import fcntl
    import numpy as np
    import torch
    from scipy.optimize import minimize
    from mdo_demo.shape_design import (VARIABLES, default_design,validate_config,make_shape_bundle,shape_margins)
    from mdo_demo.aerostructural import validate_protocol
    from mdo_demo.aerostructural_runtime import UnifiedRuntime
    from mdo_demo.aerotransformer import AeroTransformerPredictor
    from mdo_demo.aero_contract import from_fm_surface,save_loads
    from mdo_demo.matched_cfd import prediction_sample
    from mdo_demo.io import read_json,write_json,sha256_file
    from mdo_demo.shared_resources import wait_for_resources

    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=validate_config(read_json(args.protocol));write_json(output/'protocol.json',cfg)
    history=[];cache={}; torch.set_num_threads(1)
    process_start=time.perf_counter()
    status={'schema':'fm_shape_result_v1','status':'waiting_for_compute_lock','new_training':False,
            'cfd_executed':False,'independent_cfd_verification':False,'mdo_speedup_verified':False,
            'scope':cfg['scope'],'protocol_sha256':cfg['protocol_sha256'],'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'result.json',status)
    with (args.project/'manifests/compute.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        lock_wait=time.perf_counter()-process_start
        resource_wait=wait_for_resources(args.project,output/'resources.jsonl',need_gpu=True,ranks=1)
        try:
            # Reuse only the audited gas state and genuine TACS subprocess entry.
            # No fixed-seed protocol or CFD provider is constructed in this run.
            structural_cfg=validate_protocol(read_json(ROOT/'configs/aerostructural_pilot_v1.json'))
            runtime=UnifiedRuntime(args.project,args.request,structural_cfg,output/'runtime',backend='fm',checkpoint=args.checkpoint)
            # All scripts live in the isolated, frozen deployment for this run.
            runtime.code=ROOT
            model=AeroTransformerPredictor(args.checkpoint,'cuda')
            write_json(output/'model_provenance.json',model.provenance)
            code_files={str(p.relative_to(ROOT)):sha256_file(p) for folder in ('src/mdo_demo','scripts')
                        for p in (ROOT/folder).glob('*.py')}
            write_json(output/'code_receipt.json',code_files)
            initial=default_design(cfg)
            scales=np.array([cfg['design'][v]['scale'] for v in VARIABLES])
            initials=np.array([initial[v] for v in VARIABLES])
            bounds=[((cfg['design'][v]['lower']-initial[v])/scales[i],
                     (cfg['design'][v]['upper']-initial[v])/scales[i]) for i,v in enumerate(VARIABLES)]
            optimization_start=time.perf_counter()
            status['status']='running';write_json(output/'result.json',status)

            def evaluate(x,force=False):
                x=np.asarray(x,dtype=float);key=tuple(x.tolist())
                if not force and key in cache:return cache[key]
                design={v:float(initials[i]+scales[i]*x[i]) for i,v in enumerate(VARIABLES)}
                design['alpha_deg']=float(np.float32(design['alpha_deg']))
                index=len(history);folder=output/'evaluations'/f'eval_{index:04d}';folder.mkdir(parents=True)
                began=time.perf_counter()
                record={'index':index,'design':design,'path':str(folder.relative_to(output)),
                        'independent_cfd_verification':False,'feasible_under_fm':False}
                bound_margin=min(min(x[i]-lo,hi-x[i]) for i,(lo,hi) in enumerate(bounds))
                if bound_margin < -1e-8:
                    record.update(status='domain_rejected',reason='outside frozen variable bounds',
                                  optimizer_objective=10+abs(bound_margin),optimizer_constraints=[-1-abs(bound_margin)]*6)
                else:
                    try:
                        bundle,native,manifest=make_shape_bundle(args.request,design,folder/'aero/bundle',mesh_options=structural_cfg['mesh'])
                    except ValueError as exc:
                        record.update(status='geometry_rejected',reason=str(exc),optimizer_objective=10.,optimizer_constraints=[-1.]*6)
                    else:
                        runtime.native_vertices=native
                        runtime.base_identity=manifest['identity']
                        request=runtime.physical_request(manifest)
                        write_json(folder/'aero/physical_request.json',request)
                        sample,_=prediction_sample(bundle)
                        prediction=model.predict(sample)
                        loads=from_fm_surface(request,sample,prediction['fields'],timing={**prediction['timings'],
                               'scope':'FM input preparation, prediction and coefficient integral; geometry, transfer and TACS timed in complete evaluation'},provenance=model.provenance)
                        if loads['status']!='ok' or not loads['interface_compatible']:
                            raise RuntimeError('FM load contract failed')
                        save_loads(loads,folder/'aero/loads.json')
                        np.savez_compressed(folder/'aero/prediction.npz',fields=prediction['fields'],
                                            original_geometry=sample['original_geometry'])
                        structural=runtime.structure(loads,design,folder/'structure')
                        if structural['status']!='ok' or not structural['convergence']['converged'] or not structural['transfer_audit']['passed']:
                            raise RuntimeError('TACS or conservative transfer failed')
                        margins=shape_margins(loads['coefficients'],structural,manifest['geometry_measures'],manifest['baseline_geometry_measures'],cfg)
                        record.update(status='ok',mass_kg=structural['mass_kg'],tip_displacement_m=structural['tip_displacement_m'],
                                      ks_failure=structural['ks_failure'],coefficients=loads['coefficients'],
                                      constraint_margins=margins,minimum_margin=min(margins.values()),
                                      feasible_under_fm=min(margins.values())>=-cfg['optimizer']['catol'],
                                      geometry_measures=manifest['geometry_measures'],case_sha256=manifest['case_sha256'],
                                      mesh_sha256=structural['mesh_sha256'],geometry_sha256=manifest['identity']['surface_sha256'],
                                      frame_contract=loads['frame_contract'],sampling_contract=loads['sampling_contract'],
                                      prediction_timing=prediction['timings'],tacs_solve_seconds=structural['solve_seconds'],
                                      optimizer_objective=structural['mass_kg']/cfg['limits']['objective_scale_kg'],
                                      optimizer_constraints=list(margins.values()))
                record['wall_seconds']=time.perf_counter()-began
                record['cumulative_optimization_seconds']=time.perf_counter()-optimization_start
                write_json(folder/'evaluation.json',record)
                history.append(record);cache[key]=record
                with (output/'history.jsonl').open('a') as f:
                    import json
                    f.write(json.dumps(record,allow_nan=False)+'\n')
                status.update(evaluation_count=len(history),last_evaluation=record['index'])
                write_json(output/'result.json',status)
                print(index,record['status'],'mass',record.get('mass_kg'),'margin',record.get('minimum_margin'),flush=True)
                return record

            first=evaluate(np.zeros(len(VARIABLES)))
            if first['status']!='ok':raise RuntimeError('Initial physical evaluation failed')
            if args.preflight:
                xp=np.zeros(len(VARIABLES));xp[3]=.2;xp[4]=-.2
                twisted=evaluate(xp)
                xp=np.zeros(len(VARIABLES));xp[5]=.2;xp[6]=-.2;xp[7]=.2;xp[8]=.2;xp[9]=.2
                shaped=evaluate(xp)
                if any(r['status']!='ok' for r in (twisted,shaped)):raise RuntimeError('Shape preflight rejected physical analysis')
                if len({r['mesh_sha256'] for r in history})!=3:raise RuntimeError('Structural mesh did not follow each new geometry')
                status.update(status='preflight_passed',optimizer_converged=False)
            else:
                optimized=minimize(lambda x:evaluate(x)['optimizer_objective'],np.zeros(len(VARIABLES)),method='COBYLA',
                     bounds=bounds,constraints=[{'type':'ineq','fun':lambda x:np.array(evaluate(x)['optimizer_constraints'])}],
                     options={'maxiter':cfg['optimizer']['max_evaluations'],'rhobeg':cfg['optimizer']['rhobeg'],
                              'tol':cfg['optimizer']['tol'],'catol':cfg['optimizer']['catol']})
                status.update(status='completed' if optimized.success else 'budget_limited',
                              optimizer_converged=bool(optimized.success),optimizer_message=str(optimized.message))
            valid=[r for r in history if r['status']=='ok'];feasible=[r for r in valid if r['feasible_under_fm']]
            best=min(feasible,key=lambda r:r['mass_kg']) if feasible else max(valid,key=lambda r:(r['minimum_margin'],-r['mass_kg']))
            status.update(initial=first,best_candidate=best,feasible_candidate_found=bool(feasible),
                          evaluation_count=len(history),physical_evaluation_count=len(valid),
                          feasible_evaluation_count=len(feasible),optimization_wall_seconds=time.perf_counter()-optimization_start,
                          lock_wait_seconds=lock_wait,resource_wait_seconds=resource_wait,model_loading_seconds=model.load_seconds,
                          total_wall_seconds=time.perf_counter()-process_start,
                          hardware={'gpu':torch.cuda.get_device_name(0),'torch_threads':torch.get_num_threads(),'tacs_ranks':1},
                          finished_utc=datetime.now(timezone.utc).isoformat())
            for label,record in [('initial',first),('candidate',best)]:
                source=output/record['path'];target=output/label
                target.mkdir()
                for name in ('evaluation.json','aero/bundle/wing.xyz','aero/bundle/request.json','aero/bundle/native_input.json',
                             'aero/bundle/fm_input.npz','aero/prediction.npz','structure/structure_arrays.npz',
                             'structure/structure_result.json','structure/input.npz','structure/wingbox.bdf'):
                    f=target/name;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,f)
            fields=['index','status','mass_kg','tip_displacement_m','ks_failure','minimum_margin','feasible_under_fm','wall_seconds','cumulative_optimization_seconds',*VARIABLES]
            with (output/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
                for r in history:writer.writerow({**r,**r['design']})
            write_json(output/'result.json',status)
            return 0
        except Exception as exc:
            status.update(status='failed',error_type=type(exc).__name__,error=str(exc),total_wall_seconds=time.perf_counter()-process_start)
            write_json(output/'result.json',status)
            raise


if __name__=='__main__':
    raise SystemExit(main())
