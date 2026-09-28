"""Bounded orchestration/report gates. Fixtures are not measured CFD results."""
import copy
from pathlib import Path
import unittest
import numpy as np

import test_cfd_seed as seed_fixture
import test_aerostructural_comparison as comparison_fixture
from mdo_demo.aerostructural import validate_protocol, condition_bundle
from mdo_demo.io import read_json, write_json, sha256_file
from mdo_demo.matched_cfd import canonical_hash
from mdo_demo.seed_policy import (POLICY, SCHEMA, fixed_seed_policy, load_seed_manifest,
                                  project_file, validate_target_policy, continuation_eligible)

ROOT = Path(__file__).resolve().parents[1]


def configuration(manifest_sha='f'*64):
    cfg=read_json(ROOT/'configs/aerostructural_pilot_late_nk_v2.json')
    cfg['cfd_solver']=dict(POLICY)
    cfg['cfd_initialization']={'schema':SCHEMA,'manifest_sha256':manifest_sha,
        'max_continuations':1,'continuation_max_cycles':1200,
        'shared_cost_accounting':'once_per_route_including_verification'}
    return validate_protocol(cfg)


class SharedPolicyTests(unittest.TestCase):
    def test_unchanged_angle_keeps_original_bytes_even_with_different_npz_encoding(self):
        fixture=seed_fixture.FixedSeedTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source=fixture.source_request
        fm=source.parent/'fm_input.npz'
        np.savez_compressed(fm,condition=np.array([0.,.8],dtype=np.float32))
        original=read_json(source)
        original['identity']['fm_input_sha256']=sha256_file(fm)
        original['case_sha256']=canonical_hash(original['identity'])
        write_json(source,original)
        copied=condition_bundle(source,0.,fixture.root/'copied-zero')
        self.assertEqual(read_json(copied)['case_sha256'],original['case_sha256'])
        self.assertEqual((copied.parent/'fm_input.npz').read_bytes(),fm.read_bytes())
        changed=condition_bundle(source,.2,fixture.root/'changed-angle')
        self.assertNotEqual(read_json(changed)['case_sha256'],original['case_sha256'])

    def test_a_different_angle_is_allowed_only_with_frozen_shared_policy(self):
        cfg=configuration()
        validate_target_policy(cfg,{'identity':{'condition':{'alpha_deg':-.3}}})
        for alpha in (-2.01,2.01,float('nan'),True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                validate_target_policy(cfg,{'identity':{'condition':{'alpha_deg':alpha}}})
        cfg.pop('cfd_initialization')
        with self.assertRaises(ValueError): validate_protocol(cfg)

    def test_only_reviewed_budget_and_once_per_route_cost_are_allowed(self):
        cfg=configuration()
        old=validate_protocol(read_json(ROOT/'configs/aerostructural_pilot_late_nk_v2.json'))
        for key in ('design','limits','material','mesh','optimizer','comparison'):
            self.assertEqual(cfg[key],old[key])
        self.assertNotEqual(cfg['protocol_sha256'],old['protocol_sha256'])
        for key,val in [('max_continuations',2),('max_continuations',True),
                        ('continuation_max_cycles',9000),('shared_cost_accounting','ignore')]:
            changed=copy.deepcopy(cfg);changed['cfd_initialization'][key]=val
            with self.subTest(key=key),self.assertRaises(ValueError): validate_protocol(changed)
        for key,val in [('max_cycles',9000),('l2_convergence',1e-7),('mpi_ranks',16)]:
            changed=copy.deepcopy(cfg);changed['cfd_solver'][key]=val
            with self.subTest(key=key),self.assertRaises(ValueError): validate_protocol(changed)

    def test_bad_failure_never_becomes_an_automatic_retry(self):
        fixture=seed_fixture.FixedSeedTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        result=fixture.target_result()
        result['convergence']=fixture.convergence(2e8,3e6,15.)
        result['status']='seed_initialization_unqualified'
        result['seed_audit']={'passed':True,'accepted':False}
        result['internal_iterations_last_solve']=1208
        self.assertTrue(continuation_eligible(result))
        for key,value in [('status','seed_audit_failed'),('internal_iterations_last_solve',20),
                          ('configured_max_internal_iterations',9000)]:
            altered=copy.deepcopy(result);altered[key]=value
            with self.subTest(key=key): self.assertFalse(continuation_eligible(altered))
        for key in ('fatal_failed','finite_coefficients','finite_solved_alpha'):
            altered=copy.deepcopy(result);altered['convergence'][key]=key=='fatal_failed'
            with self.subTest(key=key): self.assertFalse(continuation_eligible(altered))
        result['provenance']['restart']={}
        self.assertFalse(continuation_eligible(result))

    def test_manifest_binds_source_cost_and_all_files_without_changing_namespace(self):
        fixture=seed_fixture.FixedSeedTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root=fixture.root
        cost=root/'cost.json'
        write_json(cost,{'schema':'adflow_shared_seed_host_cost_v1',
                        'seed_result_content_sha256':canonical_hash(fixture.seed),
                        'checkpoint_sha256':sha256_file(fixture.checkpoint),
                        'total_host_seed_preparation_seconds':9652.238})
        export=root/'export.npz';export.write_bytes(b'contract-only placeholder')
        meta=root/'export.json';write_json(meta,{'scope':'not a physical field'})
        files=dict(source_result=fixture.seed_path,source_request=fixture.source_request,
                   checkpoint=fixture.checkpoint,grid_audit=fixture.grid_path,
                   seed_cost_manifest=cost,initial_surface_export=export,initial_export_metadata=meta)
        manifest=root/'manifest.json'
        write_json(manifest,{'schema':SCHEMA,'files':{k:{'path':p.relative_to(root).as_posix(),
                        'sha256':sha256_file(p)} for k,p in files.items()}})
        cfg=configuration(sha256_file(manifest))
        _,loaded,ledger=load_seed_manifest(root,manifest,cfg)
        self.assertEqual(loaded,files)
        self.assertEqual(ledger['shared_seed_preparation_seconds'],9652.238)
        original=fixture.seed_path.read_bytes()
        export.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError,'changed'): load_seed_manifest(root,manifest,cfg)
        self.assertEqual(original,fixture.seed_path.read_bytes())

    def test_manifest_cannot_escape_project(self):
        for path in ('../outside','/tmp/file','C:/outside','a\\file','a/../../outside'):
            with self.subTest(path=path),self.assertRaises(ValueError): project_file(ROOT,path)


def comparison_records():
    records=comparison_fixture.fixture()
    cfg=configuration()
    ledger={'manifest_sha256':cfg['cfd_initialization']['manifest_sha256'],
            'seed_result_content_sha256':'9'*64,'checkpoint_sha256':'8'*64,
            'shared_seed_preparation_seconds':9652.238,
            'accounting':'once_per_route_including_verification'}
    for record in records:
        record.update(protocol=copy.deepcopy(cfg),protocol_sha256=cfg['protocol_sha256'],
                      shared_seed_preparation=copy.deepcopy(ledger),
                      stage_process_wall_seconds=record['total_wall_seconds']+1.)
    for branch in (0,1):
        checked=records[branch+2]['checked'];aero=checked['aerodynamic_evidence']
        aero['fixed_seed_receipt']={
            'schema':'aerostructural_fixed_seed_receipt_v3','accepted':True,'field_audit_passed':True,
            'protocol_sha256':cfg['protocol_sha256'],'initialization_manifest_sha256':ledger['manifest_sha256'],
            'case_sha256':checked['aerodynamic_case_sha256'],'alpha_deg':checked['design']['alpha_deg'],
            'result_content_sha256':aero['provenance']['solver_result_content_sha256'],
            'actual_mpi_ranks':8,'relative_residual':1e-11,'shared_cost_added_to_call':False,
            'mode':'fixed_zero_seed','continuation_count':0,'shared_preparation':copy.deepcopy(ledger),
            'actual_solver_options':{'useANKSolver':True,'useNKSolver':False,'infChangeCorrection':False,
                'equationType':'RANS','turbulenceModel':'SA','nCycles':1200,'L2Convergence':1e-10,
                'L2ConvergenceRel':1e-16,'writeVolumeSolution':True,'solutionPrecision':'double',
                'MGCycle':'sg','ANKSecondOrdSwitchTol':1e-3,'ANKCoupledSwitchTol':1e-16,'nSubiterTurb':10}}
        comparison_fixture.rebind(records,branch)
    return records


class SharedComparisonTests(unittest.TestCase):
    def test_preparation_is_charged_once_per_route_including_final_checks(self):
        result=comparison_fixture.compare(*comparison_records())
        self.assertAlmostEqual(result['rows'][0]['total_wall_seconds'],9652.238+101.+6.2)
        self.assertAlmostEqual(result['rows'][1]['total_wall_seconds'],9652.238+11.+6.2)
        self.assertIsNone(result['equal_quality_speedup'])

    def test_success_label_cannot_hide_wrong_actual_policy_or_seed(self):
        for kind in ('cycles','ranks','seed','count','ledger','receipt'):
            records=comparison_records()
            receipt=records[3]['checked']['aerodynamic_evidence']['fixed_seed_receipt']
            if kind=='cycles': receipt['actual_solver_options']['nCycles']=9000
            if kind=='ranks': receipt['actual_mpi_ranks']=16
            if kind=='seed': receipt['initialization_manifest_sha256']='0'*64
            if kind=='count': receipt['continuation_count']=2
            if kind=='ledger': receipt['shared_preparation']['shared_seed_preparation_seconds']=0.
            if kind=='receipt': receipt['accepted']=False
            with self.subTest(kind=kind),self.assertRaises(ValueError): comparison_fixture.compare(*records)

    def test_missing_full_process_cost_cannot_be_replaced_by_inner_time(self):
        records=comparison_records();records[1].pop('stage_process_wall_seconds')
        comparison_fixture.rebind(records,1)
        with self.assertRaises(ValueError): comparison_fixture.compare(*records)


if __name__=='__main__':
    unittest.main()
