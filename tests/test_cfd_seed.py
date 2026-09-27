"""Fixed-seed contract tests only; placeholder files never assert CFD validity."""
import copy
from pathlib import Path
import tempfile
import unittest

from mdo_demo.cfd import convergence_report, solver_options, validate_request
from mdo_demo.cfd_restart import audit_restart, validate_restart_source
from mdo_demo.cfd_seed import audit_seed, validate_seed_source
from mdo_demo.io import read_json, sha256_file, write_json
from mdo_demo.matched_cfd import canonical_hash


class FixedSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.condition = dict(mach=.8, alpha_deg=0., reynolds=20e6, reynolds_length_m=1., temperature_k=300.)
        self.reference = dict(area_m2=1.5, chord_m=1., moment_center_m=[.25, 0., 0.])
        self.source_request = self.bundle("source", 0.)
        self.target_request = self.bundle("target", .20000000298023224)
        source_bundle = read_json(self.source_request)
        parent_out = self.root/'parent'/'solver_surface'
        parent_out.mkdir(parents=True)
        mesh = parent_out.parent/'wing_vol.cgns'
        mesh.write_bytes(b'contract-only mesh placeholder')
        old_checkpoint = parent_out/'wing_000_vol.cgns'
        old_checkpoint.write_bytes(b'contract-only old checkpoint')
        request = validate_request(dict(schema_version=1, problem_id='seed-test', geometry_id='wing',
            geometry_sha256=source_bundle['identity']['surface_sha256'], mesh_path=str(mesh),
            condition=self.condition, reference=self.reference,
            numerics=dict(preset='robust_rans_late_nk', max_cycles=9000)))
        parent = dict(backend='adflow', status='not_converged', case_sha256=source_bundle['case_sha256'],
            volume_case_sha256=request['case_sha256'], identity=request['identity'],
            bundle_identity=source_bundle['identity'], coefficients={'CL':.7,'CD':.03},
            group_coefficients={'mainwing':{'CL':.7,'CD':.03}}, function_groups={'mainwing':'mainwing'},
            solved_alpha_deg=0., convergence=self.convergence(1e8,1e8,.012),
            configured_max_internal_iterations=9000, solver={'options':solver_options(request,parent_out)},
            surface_output_directory=str(parent_out), provenance={},
            timing=dict(total_seconds=9500.,live_mesh_and_cfd_seconds=9568.,mpi_ranks=8))
        parent_path=parent_out.parent/'result.json'
        write_json(parent_path,parent)
        restarted, _=validate_restart_source(parent_path,self.source_request.parent,old_checkpoint,sha256_file(old_checkpoint))
        restarted['numerics'].update(preset='robust_rans_ank_polish',max_cycles=1200)
        seed_out=self.root/'seed'/'solver_surface'
        seed_out.mkdir(parents=True)
        self.checkpoint=seed_out/'wing_000_vol.cgns'
        self.checkpoint.write_bytes(b'contract-only qualified seed placeholder')
        self.seed=copy.deepcopy(parent)
        self.seed.update(status='ok', convergence=self.convergence(1e8,.012,.005),
            solver={'options':solver_options(restarted,seed_out)}, provenance=restarted['provenance'],
            surface_output_directory=str(seed_out), configured_max_internal_iterations=1200,
            timing=dict(total_seconds=79.,refinement_stage_seconds=80.,mpi_ranks=8))
        self.seed['solver']['options']['restartFile']=str(old_checkpoint)
        self.seed['restart_audit']=audit_restart(parent,self.seed)
        self.seed_path=seed_out.parent/'result.json'
        write_json(self.seed_path,self.seed)
        required={name:dict(datatype=4,finite=True,minimum=1.,maximum=2.) for name in
                  ('Density','VelocityX','VelocityY','VelocityZ','Pressure','TurbulentSANuTilde')}
        self.grid=dict(passed=True,status='grid_and_checkpoint_fields_verified',
            mesh_sha256=sha256_file(mesh),checkpoint_sha256=sha256_file(self.checkpoint),
            coordinate_absolute_tolerance_m=1e-10,coordinate_relative_tolerance=0.,max_coordinate_difference_m=0.,
            zone_count=1,total_interior_cells=10,
            zones=[dict(solution=dict(location='CellCenter',required_interior_fields=required,interior_cell_count=10))])
        self.grid_path=self.root/'grid.json'
        write_json(self.grid_path,self.grid)

    @staticmethod
    def convergence(r0,start,final):
        return convergence_report((r0,start,final),tolerance=1e-10,solve_failed=False,fatal_failed=False,
                                  coefficients={'CL':.7,'CD':.03},solved_alpha_deg=0.)

    def bundle(self,name,alpha):
        directory=self.root/name
        directory.mkdir()
        (directory/'wing.xyz').write_bytes(b'identical native surface placeholder')
        (directory/'fm_input.npz').write_bytes(f'placeholder alpha={alpha}'.encode())
        write_json(directory/'native_input.json',dict(geometry={'chords':[1.,1.]},condition={'alpha_deg':alpha,'mach':.8}))
        identity=dict(case_name='wing',surface_sha256=sha256_file(directory/'wing.xyz'),
            fm_input_sha256=sha256_file(directory/'fm_input.npz'),native_input_sha256=sha256_file(directory/'native_input.json'),
            condition={**self.condition,'alpha_deg':alpha},reference=self.reference)
        path=directory/'request.json'
        write_json(path,dict(schema_version=1,identity=identity,case_sha256=canonical_hash(identity),
                            surface_path='wing.xyz',fm_input_path='fm_input.npz',native_input_path='native_input.json'))
        return path

    def validate(self,**kwargs):
        return validate_seed_source(self.seed_path,self.source_request,self.target_request,
            self.checkpoint,sha256_file(self.checkpoint),self.grid_path,**kwargs)

    def target_result(self,**kwargs):
        request,prov=self.validate(**kwargs)
        target=copy.deepcopy(self.seed)
        target.pop('restart_audit')
        target.pop('volume_case_sha256')
        target.update(case_sha256=request['case_sha256'],identity=request['identity'],provenance=request['provenance'],
                      solved_alpha_deg=request['identity']['condition']['alpha_deg'])
        target['solver']={'options':solver_options(request,self.root/'target-output')}
        target['solver']['options'].update(restartFile=str(self.checkpoint.resolve()),infChangeCorrection=False)
        # New alpha legitimately changes BOTH freestream and starting residual.
        target['convergence']=self.convergence(2e8,3e6,.015)
        target['timing']=dict(total_seconds=50.,seed_initialization_stage_seconds=55.,mpi_ranks=8)
        return target

    def test_new_target_uses_clean_provenance_and_actual_float32_alpha(self):
        request,prov=self.validate()
        self.assertNotIn('restart',request['provenance'])
        self.assertEqual(request['identity']['condition']['alpha_deg'],.20000000298023224)
        self.assertNotEqual(prov['source_volume_case_sha256'],prov['target_volume_case_sha256'])
        self.assertEqual(prov['shared_seed_preparation']['cfd_continuation_seconds'],9648.)

    def test_own_freestream_denominator_and_own_large_start_are_valid(self):
        result=self.target_result()
        audit=audit_seed(self.seed,result)
        self.assertTrue(audit['passed'])
        self.assertTrue(audit['accepted'])
        self.assertAlmostEqual(audit['target_final_relative_to_target_freestream'],7.5e-11)
        self.assertFalse(audit['seed_denominator_used_for_acceptance'])
        self.assertEqual(audit['online_call_seconds'],55.)
        self.assertFalse(audit['shared_seed_cost_added_to_call'])

    def test_small_final_relative_to_start_does_not_replace_freestream_criterion(self):
        result=self.target_result()
        result['convergence']=self.convergence(2e8,1e12,.025)
        result['status']='not_converged'
        self.assertFalse(audit_seed(self.seed,result)['accepted'])

    def test_nonalpha_condition_reference_and_geometry_changes_reject(self):
        original=read_json(self.target_request)
        for change in ('mach','reynolds','reference','surface'):
            modified=copy.deepcopy(original)
            if change in ('mach','reynolds'): modified['identity']['condition'][change]*=1.1
            if change=='reference': modified['identity']['reference']['area_m2']*=1.1
            if change=='surface': modified['identity']['surface_sha256']='a'*64
            modified['case_sha256']=canonical_hash(modified['identity'])
            write_json(self.target_request,modified)
            with self.subTest(change=change),self.assertRaises(ValueError): self.validate()
        write_json(self.target_request,original)

    def test_unconverged_seed_and_false_audit_reject(self):
        for change in ('convergence','audit'):
            modified=copy.deepcopy(self.seed)
            if change=='convergence': modified['convergence']['converged']=False
            else: modified['restart_audit']['accepted']=False
            write_json(self.seed_path,modified)
            with self.subTest(change=change),self.assertRaises(ValueError): self.validate()

    def test_wrong_checkpoint_or_incomplete_field_audit_reject(self):
        with self.assertRaisesRegex(ValueError,'checkpoint hash'):
            validate_seed_source(self.seed_path,self.source_request,self.target_request,
                self.checkpoint,'0'*64,self.grid_path)
        del self.grid['zones'][0]['solution']['required_interior_fields']['TurbulentSANuTilde']
        write_json(self.grid_path,self.grid)
        with self.assertRaisesRegex(ValueError,'restart fields'): self.validate()

    def test_actual_solved_alpha_and_policy_must_match_target(self):
        for change in ('alpha','correction','cycles','normalization','ranks'):
            result=self.target_result()
            if change=='alpha': result['solved_alpha_deg']=0.
            if change=='correction': result['solver']['options']['infChangeCorrection']=True
            if change=='cycles': result['solver']['options']['nCycles']=9000
            if change=='normalization': result['convergence']['residual_initial']=float('nan')
            if change=='ranks': result['timing']['mpi_ranks']=16
            with self.subTest(change=change),self.assertRaises(ValueError): audit_seed(self.seed,result)

    def test_shared_host_cost_is_tied_and_not_recharged_per_call(self):
        path=self.root/'host-cost.json'
        host=dict(schema='adflow_shared_seed_host_cost_v1',seed_result_content_sha256=canonical_hash(self.seed),
                  checkpoint_sha256=sha256_file(self.checkpoint),total_host_seed_preparation_seconds=9652.238)
        write_json(path,host)
        audit=audit_seed(self.seed,self.target_result(seed_cost_manifest_path=path))
        self.assertEqual(audit['shared_seed_preparation']['host_preparation_seconds'],9652.238)
        self.assertEqual(audit['online_call_seconds'],55.)
        host['checkpoint_sha256']='0'*64
        write_json(path,host)
        with self.assertRaisesRegex(ValueError,'Host seed cost'): self.validate(seed_cost_manifest_path=path)


if __name__=='__main__':
    unittest.main()
