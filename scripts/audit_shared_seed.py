#!/usr/bin/env python3
"""Audit a shared-seed route in the solver's original path namespace, no solve.

An eligible failed target gets an explicit continuation receipt, never a
successful CFD receipt. The host may perform exactly one bounded continuation.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from mdo_demo.io import read_json, write_json, sha256_file
from mdo_demo.matched_cfd import canonical_hash, load_bundle
from mdo_demo.aerostructural import validate_protocol
from mdo_demo.cfd_seed import validate_seed_source, audit_seed
from mdo_demo.cfd_restart import validate_restart_source, audit_restart
from mdo_demo.seed_policy import load_seed_manifest, validate_target_policy, continuation_eligible


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','protocol','initialization-manifest','request','result','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--first-result',type=Path)
    parser.add_argument('--allow-continuation',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous audit evidence')
    cfg=validate_protocol(read_json(args.protocol))
    _,files,ledger=load_seed_manifest(args.project,args.initialization_manifest,cfg)
    target=load_bundle(args.request)
    validate_target_policy(cfg,target)
    validate_seed_source(files['source_result'],files['source_request'],args.request,
                         files['checkpoint'],sha256_file(files['checkpoint']),files['grid_audit'],
                         seed_cost_manifest_path=files['seed_cost_manifest'])
    source=read_json(files['source_result'])
    result=read_json(args.result)
    if result.get('case_sha256') != target['case_sha256'] or result.get('bundle_identity') != target['identity']:
        raise ValueError('Result differs from the requested target bundle')
    is_initial=args.result.resolve()==files['source_result']
    first=None
    eligible=False
    if is_initial:
        if args.first_result is not None or target['identity']['condition']['alpha_deg'] != 0.:
            raise ValueError('Only the immutable zero-degree cache can be reused')
        accepted=True  # qualified source was revalidated above, not merely a status flag
        mode='qualified_zero_cache'
    elif args.first_result is None:
        audit=audit_seed(source,result)
        if audit != result.get('seed_audit'):
            raise ValueError('Seed-target audit is not reproducible')
        accepted=audit['accepted']
        eligible=args.allow_continuation and continuation_eligible(result)
        mode='fixed_zero_seed'
    else:
        if args.allow_continuation:
            raise ValueError('A second continuation is forbidden')
        first=read_json(args.first_result)
        if not continuation_eligible(first) or audit_seed(source,first) != first.get('seed_audit'):
            raise ValueError('First target was not a valid bounded continuation source')
        audit=audit_restart(first,result)
        if audit != result.get('restart_audit'):
            raise ValueError('Continuation audit is not reproducible')
        accepted=audit['accepted']
        mode='fixed_zero_seed_plus_one_continuation'
    if not accepted and not eligible:
        raise ValueError('Target is neither qualified nor eligible for the one bounded continuation')
    # Every route, including the cached zero, needs the actual complete volume
    # audit. Surface output or a volume grid alone cannot serve as a checkpoint.
    from check_restart_grid import inspect,LIBRARY
    checkpoint=Path(result['surface_output_directory'])/'wing_000_vol.cgns'
    if checkpoint.parent.resolve() != (args.result.parent/'solver_surface').resolve():
        raise ValueError('Result output does not belong to its recorded directory')
    field=inspect(Path(result['solver']['options']['gridFile']),checkpoint,LIBRARY)
    if not field.get('passed'):
        raise ValueError('Complete volume/mesh audit failed')
    volume_path=args.output.with_name(args.output.stem+'_volume.json')
    if volume_path.exists():
        raise FileExistsError('Preserve previous volume audit')
    write_json(volume_path,field)
    if eligible:
        # Also reconstruct provenance and the original target freestream norm
        # before the host is allowed to launch a continuation.
        validate_restart_source(args.result,args.request.parent,checkpoint,sha256_file(checkpoint))
    receipt={
        'schema':'aerostructural_fixed_seed_receipt_v3','mode':mode,
        'protocol_sha256':cfg['protocol_sha256'],'initialization_manifest_sha256':ledger['manifest_sha256'],
        'case_sha256':target['case_sha256'],'alpha_deg':target['identity']['condition']['alpha_deg'],
        'source_result_content_sha256':canonical_hash(source),
        'result_content_sha256':canonical_hash(result),'result_file_sha256':sha256_file(args.result),
        'accepted':bool(accepted),'continuation_eligible':bool(eligible),
        'field_audit_passed':True,'field_audit_sha256':sha256_file(volume_path),
        'checkpoint_sha256':field['checkpoint_sha256'],
        'shared_preparation':ledger,'shared_cost_added_to_call':False,
        'actual_solver_options':result['solver']['options'],'actual_mpi_ranks':result['timing']['mpi_ranks'],
        'relative_residual':result['convergence']['residual_relative_to_freestream'],
        'continuation_count':int(first is not None),
        'scope':'Numerical and identity qualification only; not physical ground truth or completed optimization.'}
    write_json(args.output,receipt)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
