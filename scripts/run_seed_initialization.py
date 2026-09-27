#!/usr/bin/env python3
"""One fixed-seed, new-angle CFD diagnostic with an independent target residual.

The host launcher holds compute.lock. This cannot silently resume the old MDO
queue; any subsequent common numerical policy must be frozen and tested first.
"""
import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from mdo_demo.cfd import run_adflow
from mdo_demo.cfd_seed import validate_seed_source,audit_seed
from mdo_demo.io import read_json,write_json
from mdo_demo.matched_cfd import load_bundle,pressure_convention_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-result','source-request','request','checkpoint','grid-audit','seed-cost-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    started=time.perf_counter()
    normalized,prov=validate_seed_source(args.source_result,args.source_request,args.request,args.checkpoint,
                                         args.checkpoint_sha256,args.grid_audit,
                                         seed_cost_manifest_path=args.seed_cost_manifest)
    target=load_bundle(args.request)
    # The existing bundle stores its native condition through a float32 array.
    if abs(target['identity']['condition']['alpha_deg'] - .2) > 1e-8:
        raise ValueError('This first fixed-seed diagnostic is frozen at native alpha +0.2 degrees')
    if args.validate_only:
        print('Fixed-seed source, new target, grid/fields and separate preparation ledger verified; no CFD run.')
        return 0
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    if comm.size != 8:
        raise ValueError('Diagnostic allocation is frozen at eight MPI ranks')
    output=args.output.resolve();error=None
    if comm.rank==0:
        if output.exists(): error='Choose a fresh new-angle output directory'
        else:
            output.mkdir(parents=True)
            write_json(output/'seed_request.json',normalized)
            write_json(output/'result.json',{'status':'running','convergence':{'converged':False},
                                             'provenance':normalized['provenance']})
    error=comm.bcast(error,root=0)
    if error: raise ValueError(error)
    comm.Barrier()
    source=read_json(args.source_result)
    result=run_adflow(normalized,output/'solver_surface',comm=comm,
                      function_groups={'mainwing':'mainwing','trailingedge':'trailingedge','tip':'tip'},
                      restart_file=args.checkpoint,restart_mode='fixed_seed')
    result['volume_case_sha256']=result['case_sha256']
    result['case_sha256']=target['case_sha256']
    result['bundle_identity']=target['identity']
    result['timing']['seed_initialization_stage_seconds']=comm.allreduce(time.perf_counter()-started,op=MPI.MAX)
    result['timing']['stage_scope']='Target/seed validation, hash checks, MPI synchronization, solver setup, initialized target solve and output; excludes launcher and final field audit'
    result['mesh_options']=source['mesh_options']
    result['mesh_quality']=source['mesh_quality']
    result['pressure_convention_audit']=pressure_convention_audit()
    result['matched_speedup_eligible']=False
    result['comparison_note']='Fixed 0-degree initial state used for a new +0.2-degree physical case. Shared seed preparation is separate. Not a completed MDO comparison.'
    try:
        result['seed_audit']=audit_seed(source,result)
        if not result['seed_audit']['accepted']:
            result['status']='seed_initialization_unqualified'
            result['convergence']['converged']=False
    except Exception as exc:
        result['status']='seed_audit_failed'
        result['seed_audit']={'passed':False,'accepted':False,'error':str(exc)}
        result['convergence']['converged']=False
    field_audit=None
    if result['status']=='ok':
        if comm.rank==0:
            from check_restart_grid import inspect,LIBRARY
            try:
                field_audit=inspect(Path(normalized['mesh_path']),output/'solver_surface/wing_000_vol.cgns',LIBRARY)
            except Exception as exc:
                field_audit={'passed':False,'error':str(exc)}
            write_json(output/'final_volume_audit.json',field_audit)
        field_audit=comm.bcast(field_audit,root=0)
        result['final_field_audit_passed']=field_audit['passed']
        if not field_audit['passed']:
            result['status']='final_field_audit_failed'
            result['convergence']['converged']=False
    result['timing']['full_diagnostic_seconds']=comm.allreduce(time.perf_counter()-started,op=MPI.MAX)
    if comm.rank==0:
        write_json(output/'result.json',result)
        print('Fixed-seed initialization status='+result['status'],flush=True)
    return 0 if result['status']=='ok' else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        if 'mpi4py.MPI' in sys.modules:
            comm=sys.modules['mpi4py.MPI'].COMM_WORLD
            if comm.size>1: comm.Abort(3)
        raise
