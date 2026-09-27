"""One explicitly audited continuation of a failed new-angle CFD target.

Reuses the existing fixed-seed contract fixture, whose files are placeholders;
none of these tests purports to run CFD or certify a real checkpoint's fields.
"""
import copy
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_cfd_seed as seed_fixture
from mdo_demo.cfd import run_adflow, solver_options
from mdo_demo.cfd_restart import audit_restart, validate_restart_source
from mdo_demo.cfd_seed import audit_seed
from mdo_demo.io import read_json, sha256_file, write_json


class SeedContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = seed_fixture.FixedSeedTests('test_new_target_uses_clean_provenance_and_actual_float32_alpha')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.output = f.root/'failed_target'/'solver_surface'
        self.output.mkdir(parents=True)
        self.checkpoint = self.output/'wing_000_vol.cgns'
        self.checkpoint.write_bytes(b'contract-only failed new-alpha volume checkpoint')
        self.source = f.target_result()
        self.source['volume_case_sha256'] = self.source['case_sha256']
        self.source['case_sha256'] = read_json(f.target_request)['case_sha256']
        self.source['bundle_identity'] = read_json(f.target_request)['identity']
        self.source.update(status='seed_initialization_unqualified',
            convergence=f.convergence(2e8,3e6,15.), surface_output_directory=str(self.output),
            timing=dict(total_seconds=1438.,seed_initialization_stage_seconds=1440.,mpi_ranks=8),
            internal_iterations_last_solve=1208)
        self.source['solver']['options']['outputDirectory'] = str(self.output)
        self.source['convergence']['solve_failed'] = True
        self.source['seed_audit'] = audit_seed(f.seed,self.source)
        self.source_path = self.output.parent/'result.json'
        write_json(self.source_path,self.source)

    def validate(self):
        return validate_restart_source(self.source_path,self.fixture.target_request.parent,
                                       self.checkpoint,sha256_file(self.checkpoint))

    def continuation(self):
        request,_ = self.validate()
        result = copy.deepcopy(self.source)
        result.pop('seed_audit')
        result.update(status='ok',provenance=request['provenance'],
            convergence=self.fixture.convergence(2e8,15.,.01),
            solver={'options':solver_options(request,self.fixture.root/'continued'/'solver_surface')},
            timing=dict(total_seconds=115.,refinement_stage_seconds=120.,mpi_ranks=8),
            internal_iterations_last_solve=600)
        result['solver']['options']['restartFile'] = str(self.checkpoint.resolve())
        result['solver']['options']['infChangeCorrection'] = False
        return result

    def test_accepts_only_audited_unqualified_target_and_cleans_contract(self):
        request,prov = self.validate()
        self.assertEqual(request['identity'],self.source['identity'])
        self.assertEqual(request['numerics']['preset'],'robust_rans_ank_polish')
        self.assertNotIn('seed_initialization',request['provenance'])
        self.assertEqual(prov['prior_seed_initialization'],self.source['provenance']['seed_initialization'])
        self.assertFalse(prov['shared_seed_cost_added_to_source_stage'])

    def test_false_seed_audit_and_wrong_ancestor_are_rejected(self):
        for change in ('audit','ancestor','status','fatal'):
            modified=copy.deepcopy(self.source)
            if change=='audit': modified['seed_audit']['accepted']=True
            if change=='ancestor': modified['provenance']['seed_initialization']['source_result_file_sha256']='0'*64
            if change=='status': modified['status']='failed'
            if change=='fatal': modified['convergence']['fatal_failed']=True
            write_json(self.source_path,modified)
            with self.subTest(change=change),self.assertRaises(ValueError): self.validate()

    def test_second_same_case_refinement_is_still_rejected(self):
        self.source['provenance']['restart']={'already':'continued'}
        write_json(self.source_path,self.source)
        with self.assertRaisesRegex(ValueError,'one checkpoint refinement'): self.validate()

    def test_target_source_and_refinement_costs_exclude_shared_seed(self):
        result=self.continuation()
        report=audit_restart(self.source,result)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['continuation_chain_seconds'],1560.)
        self.assertEqual(report['source_stage_cost']['seconds'],1440.)
        self.assertEqual(report['shared_seed_preparation']['cfd_continuation_seconds'],9648.)
        self.assertFalse(report['shared_seed_cost_added_to_continuation'])
        self.assertEqual(report['target_internal_iterations_total'],1808)
        self.assertEqual(report['max_additional_internal_iterations'],1200)

    def test_denominator_keeps_new_target_and_does_not_use_zero_seed(self):
        result=self.continuation()
        report=audit_restart(self.source,result)
        self.assertEqual(report['source_freestream_residual'],2e8)
        self.assertEqual(report['current_freestream_residual'],2e8)
        result['convergence']['residual_initial']=1e8
        result['convergence']['residual_relative_to_freestream']=.01/1e8
        with self.assertRaisesRegex(ValueError,'freestream residual denominator'):
            audit_restart(self.source,result)

    def test_mixed_contract_and_larger_retry_budget_rejected(self):
        for change in ('mixed','budget','ledger'):
            result=self.continuation()
            if change=='mixed': result['provenance']['seed_initialization']=self.source['provenance']['seed_initialization']
            if change=='budget': result['solver']['options']['nCycles']=9000
            if change=='ledger': result['provenance']['restart']['shared_seed_preparation']['cfd_continuation_seconds']=0.
            with self.subTest(change=change),self.assertRaises(ValueError): audit_restart(self.source,result)

    def test_legacy_qualified_zero_seed_audit_remains_exactly_equal(self):
        seed=self.fixture.seed
        parent=read_json(seed['provenance']['restart']['source_result_path'])
        self.assertEqual(audit_restart(parent,seed),seed['restart_audit'])

    def test_actual_alpha_and_ank_rans_double_output_policy_are_required(self):
        changes = [('useANKSolver',False),('useNKSolver',True),('infChangeCorrection',True),
                   ('equationType','Euler'),('turbulenceModel','SST'),
                   ('writeVolumeSolution',False),('solutionPrecision','single')]
        for key,value in changes:
            result=self.continuation()
            result['solver']['options'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'solver policy'):
                audit_restart(self.source,result)
        result=self.continuation()
        result['solved_alpha_deg']=0.
        with self.assertRaisesRegex(ValueError,'actual solved alpha'):
            audit_restart(self.source,result)

    def test_core_sets_no_inflow_correction_for_seed_target_continuation_only(self):
        # Inspect the constructor boundary; stop BEFORE any solver is created.
        seen=[]
        def capture(*,comm,options):
            seen.append(options)
            raise RuntimeError('stop before solver construction')
        comm=SimpleNamespace(rank=0,Barrier=lambda:None)
        modules={'adflow':SimpleNamespace(ADFLOW=capture),
                 'baseclasses':SimpleNamespace(AeroProblem=object),
                 'mpi4py':SimpleNamespace(MPI=object())}
        request,_=self.validate()
        with patch.dict('sys.modules',modules), self.assertRaisesRegex(RuntimeError,'stop before solver'):
            run_adflow(request,self.fixture.root/'capture-target',comm=comm,restart_file=self.checkpoint)
        self.assertIs(seen[-1]['infChangeCorrection'],False)
        old=self.fixture.seed['provenance']['restart']
        legacy,_=validate_restart_source(old['source_result_path'],self.fixture.source_request.parent,
                                        old['checkpoint_path'],old['checkpoint_sha256'])
        with patch.dict('sys.modules',modules), self.assertRaisesRegex(RuntimeError,'stop before solver'):
            run_adflow(legacy,self.fixture.root/'capture-legacy',comm=comm,restart_file=Path(old['checkpoint_path']))
        self.assertNotIn('infChangeCorrection',seen[-1])


if __name__=='__main__':
    unittest.main()
