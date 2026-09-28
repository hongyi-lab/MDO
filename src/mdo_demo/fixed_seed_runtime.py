"""Host orchestration for a single immutable seed and one bounded continuation.

Requires the caller's compute.lock. Never substitutes a previous design's state,
edits old results, retries a fatal failure, or chooses a different cold policy.
"""
from pathlib import Path
import time

from .io import read_json, write_json, sha256_file
from .matched_cfd import load_bundle
from .aerostructural import PhysicsFailure
from .seed_policy import load_seed_manifest, validate_target_policy


class FixedSeedCFD:
    def __init__(self, runtime, manifest_path):
        self.runtime = runtime
        self.path = Path(manifest_path).resolve()
        self.manifest, self.files, self.ledger = load_seed_manifest(
            runtime.project, self.path, runtime.protocol)
        self.protocol_path = runtime.output / 'fixed_seed_protocol.json'
        write_json(self.protocol_path, runtime.protocol)

    def common_args(self):
        return ['--project', self.runtime.project, '--protocol', self.protocol_path,
                '--initialization-manifest', self.path]

    def seed_args(self, request_path, output):
        f = self.files
        return ['python', self.runtime.code / 'scripts/run_seed_initialization.py',
                '--source-result', f['source_result'], '--source-request', f['source_request'],
                '--request', request_path, '--checkpoint', f['checkpoint'],
                '--checkpoint-sha256', self.manifest['files']['checkpoint']['sha256'],
                '--grid-audit', f['grid_audit'], '--seed-cost-manifest', f['seed_cost_manifest'],
                '--output', output, *self.common_args()]

    def preflight(self, request_path):
        # Image paths of historical provenance are preserved. Validation only;
        # the zero-degree seed is never recomputed here.
        self.runtime.image_call(self.seed_args(request_path, self.runtime.output / 'unused_validate_only')
                                + ['--validate-only'], self.runtime.output / 'fixed_seed_preflight.log')

    def audit(self, request_path, result_path, output, *, first_result=None, allow_continuation=False):
        args = ['python', self.runtime.code / 'scripts/audit_shared_seed.py', *self.common_args(),
                '--request', request_path, '--result', result_path, '--output', output]
        if first_result is not None:
            args += ['--first-result', first_result]
        if allow_continuation:
            args += ['--allow-continuation']
        self.runtime.image_call(args, output.with_suffix('.log'))
        return read_json(output)

    def run(self, request_path, output):
        output = Path(output)
        validate_target_policy(self.runtime.protocol, load_bundle(request_path))
        # Recheck immutable file hashes per new design, not just at construction.
        load_seed_manifest(self.runtime.project, self.path, self.runtime.protocol)
        start = time.perf_counter()
        process_calls = []
        target = load_bundle(request_path)
        alpha = target['identity']['condition']['alpha_deg']
        if alpha == 0.:
            source = read_json(self.files['source_result'])
            if (source['case_sha256'] != target['case_sha256']
                    or source['bundle_identity'] != target['identity']):
                raise PhysicsFailure('Zero cache serialization differs from the frozen request; do not relabel it')
            result_path = self.files['source_result']
            receipt = self.audit(request_path, result_path, output / 'fixed_seed_audit.json')
            export = self.files['initial_surface_export']
            export_meta = self.files['initial_export_metadata']
        else:
            mpi = ['mpirun', '--bind-to', 'none', '-np', '8']
            first_path = output / 'fixed_seed' / 'result.json'
            launch = self.runtime.image_process(mpi + self.seed_args(request_path, first_path.parent),
                            output / 'fixed_seed.log', allowed_exit_codes=(0, 2))
            process_calls.append(launch)
            receipt = self.audit(request_path, first_path, output / 'first_audit.json', allow_continuation=True)
            if (launch['exit_code'] == 0) != receipt['accepted']:
                raise PhysicsFailure('Solver exit status and qualified CFD receipt disagree')
            result_path = first_path
            if not receipt['accepted']:
                if not receipt['continuation_eligible']:
                    raise PhysicsFailure('Fixed seed failed without a qualified continuation source')
                checkpoint = first_path.parent / 'solver_surface/wing_000_vol.cgns'
                result_path = output / 'continuation' / 'result.json'
                args = mpi + ['python', self.runtime.code / 'scripts/run_checkpoint_refinement.py',
                        '--source-result', first_path, '--request', request_path, '--checkpoint', checkpoint,
                        '--checkpoint-sha256', sha256_file(checkpoint),
                        '--grid-audit', output / 'first_audit_volume.json', '--output', result_path.parent]
                # The second failure is final for this route. No loop/retry exists.
                launch = self.runtime.image_process(args, output / 'continuation.log', allowed_exit_codes=(0, 2))
                process_calls.append(launch)
                write_json(output / 'cfd_process_costs.json', process_calls)
                if launch['exit_code']:
                    raise PhysicsFailure('The sole bounded continuation failed; later queue stages must stop')
                receipt = self.audit(request_path, result_path, output / 'fixed_seed_audit.json',
                                     first_result=first_path)
            surfaces = sorted((result_path.parent / 'solver_surface').glob('*_surf.cgns'))
            if len(surfaces) != 1:
                raise PhysicsFailure('Expected exactly one qualified native surface')
            export = output / 'native_surface.npz'
            export_meta = export.with_suffix('.json')
            self.runtime.image_call(['python', self.runtime.code / 'scripts/export_native_cfd_surface.py',
                    '--request', request_path, '--result', result_path, '--surface', surfaces[0],
                    '--output', export], output / 'export.log')
        if not receipt['accepted'] or not receipt['field_audit_passed']:
            raise PhysicsFailure('Only a qualified field can enter structural analysis')
        receipt.update(host_solver_processes=process_calls,
                       host_solver_chain_seconds=sum(p['wall_seconds'] for p in process_calls),
                       host_route_and_export_seconds=time.perf_counter() - start,
                       shared_cost_added_to_call=False)
        write_json(output / 'fixed_seed_receipt.json', receipt)
        write_json(output / 'cfd_process_costs.json', process_calls)
        return result_path, export, export_meta, receipt
