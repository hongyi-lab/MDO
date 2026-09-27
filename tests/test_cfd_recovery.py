"""Recovery keeps physical identity and acceptance fixed; no CFD is mocked as truth."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mdo_demo.aerostructural import cfd_numerical_policy, validate_protocol
from mdo_demo.cfd import convergence_report, solver_options, validate_request
from mdo_demo.matched_cfd import run_bundle

ROOT = Path(__file__).resolve().parents[1]


class CFDRecoveryTests(unittest.TestCase):
    def test_numerical_recovery_keeps_case_and_acceptance_but_changes_declared_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            mesh = directory/'wing.cgns'
            mesh.write_bytes(b'identity test only, not a usable CFD mesh')
            raw = dict(schema_version=1, problem_id='test', geometry_id='wing', mesh_path=str(mesh),
                       condition=dict(mach=.8, alpha_deg=0., reynolds=20e6, reynolds_length_m=1., temperature_k=300.),
                       reference=dict(area_m2=2.7, chord_m=1., moment_center_m=[.25, 0., 0.]),
                       numerics=dict(preset='robust_rans'))
            old = validate_request(raw)
            raw['numerics'] = dict(preset='robust_rans_late_nk', max_cycles=9000)
            new = validate_request(raw)
            self.assertEqual(old['case_sha256'], new['case_sha256'])
            before, after = (solver_options(case, directory) for case in (old, new))
            self.assertEqual(before['NKSwitchTol'], 1e-5)
            self.assertEqual(before['nCycles'], 3000)
            self.assertFalse(before['writeVolumeSolution'])
            for key in ('gridFile', 'equationType', 'turbulenceModel', 'L2Convergence',
                        'L2ConvergenceRel', 'ANKSecondOrdSwitchTol', 'nSubiterTurb'):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after['L2Convergence'], 1e-10)
            self.assertEqual(after['NKSwitchTol'], 1e-7)
            self.assertEqual(after['nCycles'], 9000)
            self.assertTrue(after['writeVolumeSolution'])
            self.assertEqual(after['solutionPrecision'], 'double')
            self.assertNotIn('restartFile', after)

    def test_frozen_policy_changes_hash_without_changing_design_task(self):
        old = json.loads((ROOT/'configs/aerostructural_pilot_v1.json').read_text())
        new = json.loads((ROOT/'configs/aerostructural_pilot_late_nk_v2.json').read_text())
        for key in ('design', 'limits', 'material', 'mesh', 'optimizer', 'comparison'):
            self.assertEqual(old[key], new[key])
        self.assertNotEqual(validate_protocol(old)['protocol_sha256'], validate_protocol(new)['protocol_sha256'])
        self.assertEqual(cfd_numerical_policy(old)['max_cycles'], 3000)
        self.assertEqual(cfd_numerical_policy(new)['max_cycles'], 9000)

    def test_policy_cannot_relax_tolerance_or_expand_shared_cpu_allocation(self):
        cfg = json.loads((ROOT/'configs/aerostructural_pilot_late_nk_v2.json').read_text())
        for key, value in [('l2_convergence', 1e-6), ('mpi_ranks', 16),
                           ('max_cycles', True), ('max_cycles', 0), ('max_cycles', 1.5)]:
            changed = copy.deepcopy(cfg)
            changed['cfd_solver'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_protocol(changed)

    def test_invalid_run_budget_fails_before_mpi_or_filesystem_work(self):
        for invalid in (0, -1, True, 1.5, '9000'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'max_cycles'):
                run_bundle(Path('nonexistent-request'), Path('must-not-be-created'), max_cycles=invalid)

    def test_cli_invalid_budget_rejects_before_optional_mpi_import(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'untouched'
            for value in ('0', '-1', 'abc'):
                command = [sys.executable, str(ROOT/'scripts/run_matched_cfd.py'), 'run',
                           '--request', str(Path(temp)/'missing.json'), '--output', str(out), '--max-cycles', value]
                proc = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn('--max-cycles', proc.stderr)
                self.assertNotIn('ModuleNotFoundError', proc.stderr)
                self.assertFalse(out.exists())

    def test_observed_failed_residual_stays_rejected_after_recovery(self):
        result = convergence_report((142581965.0852572, 142581965.0852572, 9.463003476378178),
            tolerance=1e-10, solve_failed=False, fatal_failed=False, coefficients={'CL': .7112723, 'CD': .0322557})
        self.assertFalse(result['converged'])
        self.assertFalse(result['residual_pass'])


if __name__ == '__main__':
    unittest.main()
