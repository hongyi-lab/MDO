"""State-machine tests with synthetic process receipts, never physical scores."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import test_cfd_seed as seed_fixture
from test_shared_seed_policy import configuration
from mdo_demo.fixed_seed_runtime import FixedSeedCFD
from mdo_demo.aerostructural import PhysicsFailure
from mdo_demo.io import write_json


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=seed_fixture.FixedSeedTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.calls=[];self.codes=[];self.audits=[]
        self.driver=FixedSeedCFD.__new__(FixedSeedCFD)
        self.driver.runtime=SimpleNamespace(project=self.root,protocol=configuration(),code=self.root/'code',
            image_process=self.process,image_call=lambda *a,**k:0.)
        self.driver.path=self.root/'manifest.json'
        self.driver.protocol_path=self.root/'protocol.json'
        self.driver.files={'source_result':self.fixture.seed_path,'source_request':self.fixture.source_request,
            'checkpoint':self.fixture.checkpoint,'grid_audit':self.fixture.grid_path,
            'seed_cost_manifest':self.root/'cost.json', 'initial_surface_export':self.root/'export.npz',
            'initial_export_metadata':self.root/'export.json'}
        self.driver.manifest={'files':{'checkpoint':{'sha256':'a'*64}}}
        self.driver.audit=self.audit
        self.out=self.root/'route';self.out.mkdir()
        self.patcher=patch('mdo_demo.fixed_seed_runtime.load_seed_manifest',return_value=({}, {}, {}))
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def process(self,args,log,**kw):
        self.calls.append(args)
        output=Path(args[args.index('--output')+1]);(output/'solver_surface').mkdir(parents=True)
        (output/'solver_surface/wing_000_vol.cgns').write_bytes(b'not a real field')
        (output/'solver_surface/wing_000_surf.cgns').write_bytes(b'not a real surface')
        write_json(output/'result.json',{'scope':'synthetic execution fixture'})
        return {'exit_code':self.codes.pop(0),'wall_seconds':float(len(self.calls)*10)}

    def audit(self,*a,**kw):
        value=self.audits.pop(0)
        if isinstance(value,Exception): raise value
        return {'accepted':value,'field_audit_passed':True,'continuation_eligible':not value}

    def run_target(self):
        return self.driver.run(self.fixture.target_request,self.out)

    def test_success_uses_immutable_zero_seed_and_no_extra_solve(self):
        self.codes=[0];self.audits=[True]
        result,_,_,receipt=self.run_target()
        self.assertEqual(len(self.calls),1)
        command=self.calls[0]
        self.assertEqual(command[command.index('--checkpoint')+1],self.fixture.checkpoint)
        self.assertEqual(command[command.index('-np')+1],'8')
        self.assertEqual(receipt['host_solver_chain_seconds'],10.)
        self.assertFalse(receipt['shared_cost_added_to_call'])

    def test_one_failed_target_continues_its_own_field_once_and_retains_both_costs(self):
        self.codes=[2,0];self.audits=[False,True]
        result,_,_,receipt=self.run_target()
        self.assertEqual(len(self.calls),2)
        second=self.calls[1]
        self.assertEqual(second[second.index('--source-result')+1],self.out/'fixed_seed/result.json')
        self.assertEqual(second[second.index('--checkpoint')+1],
                         self.out/'fixed_seed/solver_surface/wing_000_vol.cgns')
        self.assertEqual(receipt['host_solver_chain_seconds'],30.)
        self.assertEqual(result,self.out/'continuation/result.json')

    def test_second_failure_stops_without_a_third_attempt(self):
        self.codes=[2,2];self.audits=[False]
        with self.assertRaisesRegex(PhysicsFailure,'sole bounded'): self.run_target()
        self.assertEqual(len(self.calls),2)
        self.assertTrue((self.out/'cfd_process_costs.json').is_file())

    def test_audit_failure_stops_before_a_continuation(self):
        self.codes=[2];self.audits=[ValueError('mismatched source identity')]
        with self.assertRaisesRegex(ValueError,'identity'): self.run_target()
        self.assertEqual(len(self.calls),1)

    def test_success_exit_cannot_hide_unqualified_result(self):
        self.codes=[0];self.audits=[False]
        with self.assertRaisesRegex(PhysicsFailure,'disagree'): self.run_target()
        self.assertEqual(len(self.calls),1)

    def test_initial_cache_does_not_run_cfd_or_charge_seed_per_call(self):
        self.audits=[True]
        _,_,_,receipt=self.driver.run(self.fixture.source_request,self.out)
        self.assertEqual(self.calls,[])
        self.assertEqual(receipt['host_solver_chain_seconds'],0.)
        self.assertFalse(receipt['shared_cost_added_to_call'])


if __name__=='__main__':
    unittest.main()
