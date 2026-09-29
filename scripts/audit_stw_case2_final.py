#!/usr/bin/env python3
"""Independent, server-side checks and tabular export for a completed STW run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdo_demo.stw_task import G, atmosphere, mission, read_stw_bdf, sizing_margins, wingbox_volume


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--bdf', type=Path, required=True)
    parser.add_argument('--predecessor', type=Path, action='append', required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    if summary['status'] != 'complete_model_feasible':
        raise RuntimeError('No completed feasible design to audit')
    selected = args.run / 'final_tighter_reanalysis'
    result = json.loads((selected / 'result.json').read_text())
    initial = json.loads((args.initial / 'result.json').read_text())
    panels = json.loads((args.run / 'structure/panels.json').read_text())
    mesh = read_stw_bdf(args.bdf)
    checks = {}
    fields = {}
    for name in ('cruise', 'pullup', 'pushdown'):
        with np.load(selected / f'{name}.npz') as f:
            fields[name] = {k: f[k].copy() for k in f.files}
        f = fields[name]
        checks[name + '_finite'] = all(np.isfinite(a).all() for a in f.values())
        r = result['conditions'][name]
        checks[name + '_coupled'] = bool(r['converged'] and r['displacement_residual_m'] < 5e-5
                                        and abs(r['CL_residual']) < 7.5e-6)
        points = f['points'].reshape(-1, 3)
        forces = f['forces'].reshape(-1, 3)
        nodal = f['loads']
        source = np.r_[forces.sum(axis=0), np.cross(points, forces).sum(axis=0)]
        target = np.r_[nodal[:, :3].sum(axis=0),
                       (np.cross(f['nodes'], nodal[:, :3]) + nodal[:, 3:]).sum(axis=0)]
        checks[name + '_wrench_conservation'] = bool(np.allclose(source, target, rtol=1e-10, atol=1e-6))
        a = np.deg2rad(float(f['alpha']))
        atm = atmosphere(10400 if name == 'cruise' else 0)
        mach = .77 if name == 'cruise' else .4577
        q = .5 * atm['rho'] * (mach * atm['a']) ** 2
        cl = (-source[0] * np.sin(a) + source[2] * np.cos(a)) / (q * 45.5)
        checks[name + '_CL_reintegrated'] = bool(np.isclose(cl, r['CL'], rtol=1e-7, atol=1e-9))
        checks[name + '_same_design'] = bool(np.array_equal(f['design'], fields['cruise']['design']))
        if name != 'cruise':
            checks[name + '_ultimate_safe'] = bool(r['ultimate_failure'] <= 1.0001)
    cruise = fields['cruise']
    computed_mission = mission(result['conditions']['cruise']['mass_box_kg'],
                               result['conditions']['cruise']['CL'], result['conditions']['cruise']['CD'])
    checks['mission_recomputed'] = all(np.isclose(v, result['mission'][k], rtol=1e-12)
                                       for k, v in computed_mission.items())
    volume = wingbox_volume(mesh, cruise['nodes'])
    capacity = 804 * (1.7 * volume + 2.763)
    checks['volume_recomputed'] = bool(np.isclose(volume, result['wingbox_volume_m3'], rtol=1e-12))
    checks['fuel_capacity'] = bool(capacity - computed_mission['total_fuel_kg'] >= -1e-3)
    margins = sizing_margins(cruise['design'], panels)
    checks['sizing_constraints'] = bool(min(margins.values()) >= -1e-8)
    checks['shape_bounds'] = bool(result['geometry_bounds_satisfied'])
    with np.load(args.initial/'cruise.npz') as f:
        original_jig=f['jig'].copy()
    rows=np.arange(len(original_jig)); le_indices=original_jig[:,:,0].argmin(axis=1)
    checks['leading_edge_fixed'] = bool(np.allclose(cruise['jig'][rows,le_indices],original_jig[rows,le_indices],atol=1e-8))
    checks['root_not_twisted'] = bool(np.allclose(cruise['jig'][0,:,:2],original_jig[0,:,:2],atol=1e-8))
    out = args.run / 'audited_delivery'
    out.mkdir(exist_ok=True)
    audit = dict(passed=all(checks.values()), checks=checks, sizing_margins_m=margins,
                 scope='Numerical/constraint consistency of the hybrid model, not independent physical validation',
                 source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [args.run/'summary.json', selected/'result.json', args.bdf,
                                          *selected.glob('*.npz')]})
    write(out/'audit.json', audit)
    if not audit['passed']:
        raise RuntimeError('Failed final consistency audit: ' + str([k for k,v in checks.items() if not v]))
    with (out/'scores.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'unit', 'initial_hybrid', 'optimized_hybrid', 'public_UM_case2', 'meaning'])
        for name, unit, b, f, public, meaning in [
            ('mission_fuel_burn', 'kg', initial['mission']['fuel_burn_kg'], result['mission']['fuel_burn_kg'], 10983.96, 'lower; public value uses a different aerodynamic model'),
            ('wingbox_mass', 'kg', initial['conditions']['cruise']['mass_box_kg'], result['conditions']['cruise']['mass_box_kg'], 704.54, 'mass of one structural wingbox'),
            ('aircraft_lift_to_drag', '1', initial['mission']['L_D'], result['mission']['L_D'], 16.17, 'includes prescribed airframe drag'),
            ('fuel_capacity_margin', 'kg', initial['fuel_capacity_margin_kg'], result['fuel_capacity_margin_kg'], '', 'must be nonnegative; includes reserve'),
            ('pullup_ultimate_failure_KS', '1', initial['conditions']['pullup']['ultimate_failure'], result['conditions']['pullup']['ultimate_failure'], '', 'must be <=1 at 1.5 times limit load'),
            ('pushdown_ultimate_failure_KS', '1', initial['conditions']['pushdown']['ultimate_failure'], result['conditions']['pushdown']['ultimate_failure'], '', 'must be <=1 at 1.5 times limit load'),
            ('model_feasible', 'boolean', initial['feasible_model_only'], result['feasible_model_only'], '', 'does not certify CFD/experimental accuracy'),
        ]:
            writer.writerow([name, unit, b, f, public, meaning])
    with (out/'structural_design.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['panel','panel_thickness_m','stiffener_height_m','stiffener_thickness_m','panel_length_m'])
        for panel in panels:
            idx=panel['dv_start']; writer.writerow([panel['name'], *cruise['design'][idx:idx+4]])
    with (out/'coupled_conditions.csv').open('w', newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['condition','backend','alpha_deg','CL','CD','CL_residual','displacement_residual_m','ultimate_KS','max_displacement_m'])
        for name,r in result['conditions'].items():
            writer.writerow([name,r['aerodynamic_backend'],r['alpha_deg'],r['CL'],r['CD'],r['CL_residual'],r['displacement_residual_m'],r['ultimate_failure'],r['max_deflection_m']])
    predecessors={str(path):json.loads((path/'summary.json').read_text())['wall_seconds'] for path in args.predecessor}
    write(out/'timing.json',dict(predecessor_optimization_seconds=predecessors,
          completion_search_seconds=summary['wall_seconds'],
          total_search_seconds=sum(predecessors.values())+summary['wall_seconds'],
          final_three_condition_analysis_seconds=result['wall_seconds'],
          scope='All listed optimization runs including model and TACS startup. Earlier development/diagnostics excluded and not counted as zero. No CFD speedup claimed.',
          inference_location='SSH server',CPU_threads=1,GPU='RTX A6000 48GB'))
    # Open, editable geometry arrays and a basic OBJ for later rendering.
    for name, vertices in [('initial_wing', np.load(args.initial/'cruise.npz')['jig']),
                           ('optimized_jig_wing', cruise['jig']),('optimized_cruise_wing',cruise['deformed'])]:
        with (out/f'{name}.obj').open('w') as handle:
            handle.write('# STW half wing; x downstream, y semispan, z up; metres\n')
            for xyz in vertices.reshape(-1,3):handle.write('v '+' '.join(f'{x:.12g}' for x in xyz)+'\n')
            ni,nj=vertices.shape[:2]
            for i in range(ni-1):
                for j in range(nj-1):
                    ids=[i*nj+j+1,(i+1)*nj+j+1,(i+1)*nj+j+2,i*nj+j+2]
                    handle.write('f '+' '.join(map(str,ids))+'\n')
    write(out/'result_digest.json',dict(passed=True,mission=result['mission'],
          box_mass_kg=result['conditions']['cruise']['mass_box_kg'],
          fuel_change_percent=100*(result['mission']['fuel_burn_kg']/initial['mission']['fuel_burn_kg']-1),
          fuel_capacity_margin_kg=result['fuel_capacity_margin_kg'],
          ultimate_failure={k:result['conditions'][k]['ultimate_failure'] for k in ['pullup','pushdown']},
          independent_physical_validation=False))
    print(json.dumps(json.loads((out/'result_digest.json').read_text()),indent=2))


if __name__=='__main__':
    main()
