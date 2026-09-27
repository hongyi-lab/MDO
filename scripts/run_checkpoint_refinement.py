#!/usr/bin/env python3
"""One bounded ADflow checkpoint diagnostic, never automatic MDO continuation.

The launcher must hold the project compute.lock. Run inside the source result's
original path namespace with eight MPI ranks. Old outputs remain immutable.
"""
import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdo_demo.cfd import run_adflow, sha256_file
from mdo_demo.cfd_restart import validate_restart_source, audit_restart
from mdo_demo.io import read_json, write_json
from mdo_demo.matched_cfd import pressure_convention_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-result', 'request', 'checkpoint', 'grid-audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    # Hashes and case validation are intentionally before importing ADflow.
    normalized, provenance = validate_restart_source(args.source_result, args.request.parent,
                                                     args.checkpoint, args.checkpoint_sha256)
    grid = read_json(args.grid_audit)
    if not grid.get('passed'):
        raise ValueError('Checkpoint grid-coordinate audit did not pass')
    if grid.get('mesh_sha256') != normalized['identity']['volume_mesh_sha256']:
        raise ValueError('Grid audit belongs to a different mesh')
    if grid.get('checkpoint_sha256') != args.checkpoint_sha256:
        raise ValueError('Grid audit belongs to a different checkpoint')
    provenance['grid_audit_sha256'] = sha256_file(args.grid_audit)
    normalized['provenance']['restart'] = provenance
    normalized['numerics'].update(preset='robust_rans_ank_polish', max_cycles=1200, l2_convergence=1e-10)
    if args.validate_only:
        print('Checkpoint, mesh, physical identity and recorded provenance verified; no solve performed.')
        return 0
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.size != 8:
        raise ValueError('Diagnostic allocation is frozen at eight MPI ranks')
    output = args.output.resolve()
    error = None
    if comm.rank == 0:
        if output.exists():
            error = 'Choose a fresh refinement output directory'
        else:
            output.mkdir(parents=True)
            write_json(output/'restart_request.json', normalized)
            write_json(output/'result.json', {'status':'running', 'convergence':{'converged':False},
                                             'provenance':normalized['provenance']})
    error = comm.bcast(error, root=0)
    if error:
        raise ValueError(error)
    comm.Barrier()
    source = read_json(args.source_result)
    result = run_adflow(normalized, output/'solver_surface', comm=comm,
                        function_groups={'mainwing':'mainwing','trailingedge':'trailingedge','tip':'tip'},
                        restart_file=args.checkpoint)
    result['volume_case_sha256'] = result['case_sha256']
    result['case_sha256'] = source['case_sha256']
    result['bundle_identity'] = source['bundle_identity']
    result['timing']['refinement_stage_seconds'] = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)
    result['timing']['stage_scope'] = 'Validation, checkpoint/grid hashing, MPI synchronization, solver setup, restart solve and solver outputs; excludes process launch and final JSON serialization'
    result['mesh_quality'] = source['mesh_quality']
    result['mesh_options'] = source['mesh_options']
    result['pressure_convention_audit'] = pressure_convention_audit()
    result['matched_speedup_eligible'] = False
    result['comparison_note'] = 'Same-case checkpoint refinement diagnostic. Not a completed MDO run or fair speedup measurement. Source computation and this stage must both be counted.'
    try:
        audit = audit_restart(source, result)
        result['restart_audit'] = audit
        if not audit.get('accepted'):
            result['status'] = 'restart_unqualified'
            result['convergence']['converged'] = False
    except Exception as exc:
        result['status'] = 'restart_audit_failed'
        result['restart_audit'] = {'passed':False, 'accepted':False, 'error':str(exc)}
        result['convergence']['converged'] = False
    if comm.rank == 0:
        write_json(output/'result.json', result)
        print('Checkpoint refinement status='+result['status'], flush=True)
    return 0 if result['status'] == 'ok' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        # Avoid a hung MPI rank when validation or IO fails on only one rank.
        if 'mpi4py.MPI' in sys.modules:
            comm = sys.modules['mpi4py.MPI'].COMM_WORLD
            if comm.size > 1:
                comm.Abort(3)
        raise
