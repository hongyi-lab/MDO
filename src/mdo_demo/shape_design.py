"""Bounded FM shape design on the actual native wing, with rebuilt shell FEA.

Geometry is morphed in physical coordinates, then resampled by the existing
native-to-FM adapter. Fixed reference area preserves the SAME dimensional lift
and drag requirements when planform area changes. No CFD accuracy is implied.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time
import numpy as np
from scipy.integrate import trapezoid

from .aero_contract import read_native_surface, native_mainwing_vertices
from .matched_cfd import (load_bundle, canonical_hash, reference_from_native_blocks,
                          NATIVE_INPUT_FRAME)
from .io import read_json, write_json, sha256_file
from .structural import build_wingbox_mesh, validate_wingbox_thickness

VARIABLES = ('alpha_deg', 'skin_m', 'web_m', 'twist_mid_deg', 'twist_tip_deg',
             'chord_root_scale', 'chord_mid_scale', 'chord_tip_scale',
             'span_scale', 'sweep_delta_deg')
SHAPE_VARIABLES = VARIABLES[3:]


def validate_config(raw):
    cfg = copy.deepcopy(raw)
    if cfg.get('schema') != 'fm_shape_design_v1' or set(cfg['design']) != set(VARIABLES):
        raise ValueError('Explicit ten-variable FM shape protocol required')
    for name in VARIABLES:
        d = cfg['design'][name]
        if not np.isfinite(list(d.values())).all() or not d['lower'] < d['upper']:
            raise ValueError(f'Invalid bounds: {name}')
        if not d['lower'] <= d['initial'] <= d['upper'] or d['scale'] <= 0:
            raise ValueError(f'Invalid initial value or scale: {name}')
        if (name.endswith('_m') or name.endswith('_scale')) and d['lower'] <= 0:
            raise ValueError('Thickness and geometric scales must be positive')
    if cfg['optimizer']['method'] != 'COBYLA' or cfg['optimizer']['max_evaluations'] < 12:
        raise ValueError('Require a finite COBYLA budget')
    if cfg['limits']['volume_ratio_min'] <= 0 or cfg['limits']['area_ratio_min'] <= 0:
        raise ValueError('Positive geometric capacity requirements needed')
    cfg['protocol_sha256'] = canonical_hash({k:v for k,v in cfg.items() if k != 'protocol_sha256'})
    return cfg


def default_design(cfg):
    return {v:float(cfg['design'][v]['initial']) for v in VARIABLES}


def geometry_measures(vertices):
    """Projected planform area and closed-section enclosed geometric volume.

    The tiny open TE gap is closed only for this geometric capacity integral;
    it is NOT added to the aerodynamic load integration surface.
    """
    s = np.asarray(vertices).transpose(1,2,0)
    span = -s[:,:,2].mean(axis=1)
    chord = np.ptp(s[:,:,0], axis=1)
    x,y = s[:,:,0],s[:,:,1]
    section_area = .5*np.abs(np.sum(x*np.roll(y,-1,axis=1)-y*np.roll(x,-1,axis=1),axis=1))
    return {'projected_area_m2':float(trapezoid(chord,span)),
            'enclosed_geometric_volume_m3':float(trapezoid(section_area,span)),
            'structural_span_m':float(span[-1]-span[0]),
            'root_chord_m':float(chord[0]), 'tip_chord_m':float(chord[-1])}


def morph_blocks(blocks, base_vertices, design):
    """Continuous quarter-chord morph, shared by ALL thirteen native patches.

    Positive twist is nose-up: clockwise x/y rotation about the section quarter
    chord. Root twist stays fixed to distinguish twist from global alpha.
    Root z remains fixed; span scale applies outboard from that station.
    """
    if not np.isfinite([design[k] for k in SHAPE_VARIABLES]).all():
        raise ValueError('Nonfinite shape variable')
    scales = [design[k] for k in ('chord_root_scale','chord_mid_scale','chord_tip_scale','span_scale')]
    if min(scales) <= 0 or abs(design['sweep_delta_deg']) >= 30:
        raise ValueError('Invalid shape scales/sweep')
    s = base_vertices.transpose(1,2,0)
    z = s[:,:,2].mean(axis=1)
    eta = (z-z[0])/(z[-1]-z[0])
    # Midpoint of trailing-edge ends and the actual most-upstream point.
    le = s[np.arange(len(s)),s[:,:,0].argmin(axis=1)]
    te = .5*(s[:,0]+s[:,-1])
    pivot = le + .25*(te-le)
    span = z[0]-z[-1]
    base_sweep = np.arctan2(pivot[-1,0]-pivot[0,0],span)
    sweep_shift = np.tan(base_sweep+np.deg2rad(design['sweep_delta_deg']))-np.tan(base_sweep)
    changed = []
    for block in blocks:
        a = np.asarray(block,dtype=float).copy()
        e = (a[2]-z[0])/(z[-1]-z[0])
        px = np.interp(e,eta,pivot[:,0]); py = np.interp(e,eta,pivot[:,1])
        scale = np.interp(e,[0,.5,1],scales[:3])
        theta = np.deg2rad(np.interp(e,[0,.5,1],[0,design['twist_mid_deg'],design['twist_tip_deg']]))
        dx=(a[0]-px)*scale; dy=(a[1]-py)*scale
        a[0]=px+np.cos(theta)*dx+np.sin(theta)*dy + e*span*design['span_scale']*sweep_shift
        a[1]=py-np.sin(theta)*dx+np.cos(theta)*dy
        a[2]=z[0]+(a[2]-z[0])*design['span_scale']
        changed.append(a)
    return changed


def write_surface(path, blocks):
    with Path(path).open('w',encoding='ascii') as f:
        f.write(str(len(blocks))+'\n')
        for b in blocks: f.write(f'{b.shape[2]} {b.shape[1]} 1\n')
        for b in blocks: np.savetxt(f,b.reshape(-1,1),fmt='%.17e')


def make_shape_bundle(base_request, design, output, *, mesh_options):
    started=time.perf_counter()
    base_request=Path(base_request).resolve(); output=Path(output)
    base=load_bundle(base_request)
    blocks=read_native_surface(base_request.parent/base['surface_path'])
    original=native_mainwing_vertices(base_request.parent/base['surface_path'])
    changed=morph_blocks(blocks,original,design)
    output.mkdir(parents=True,exist_ok=False)
    write_surface(output/'wing.xyz',changed)
    native=native_mainwing_vertices(output/'wing.xyz')
    mesh=build_wingbox_mesh(native,**mesh_options)
    validate_wingbox_thickness(mesh,{k:design[k] for k in ('skin_m','web_m')})
    # reference_from_native_blocks accepts the author's (ni,nj,1,3) format.
    author_blocks=[b.transpose(2,1,0)[:,:,None,:].copy() for b in changed]
    sampled,sampling=reference_from_native_blocks(author_blocks)
    centers=.25*(sampled[:,1:,1:]+sampled[:,1:,:-1]+sampled[:,:-1,1:]+sampled[:,:-1,:-1])
    condition=np.array([design['alpha_deg'],base['identity']['condition']['mach']],dtype=np.float32)
    # Constant reference area is deliberate; lift/drag constraints correspond
    # to fixed forces, NOT constant CL on an ever-shrinking physical wing.
    ref=copy.deepcopy(base['identity']['reference'])
    np.savez(output/'fm_input.npz',original_geometry=sampled,geometry=centers.astype(np.float32),
             condition=condition,ref_area=ref['area_m2'])
    definition={'schema':'native_surface_morph_v1','base_surface_sha256':base['identity']['surface_sha256'],
                'design':{k:float(design[k]) for k in VARIABLES},
                'rule':'quarter-chord nose-up twist, chord scaling, root-fixed span and sweep increments',
                'reference_area':'fixed baseline for invariant dimensional task'}
    write_json(output/'native_input.json',definition)
    identity=copy.deepcopy(base['identity'])
    identity.update(case_name='fm-shape-candidate',native_input_sha256=sha256_file(output/'native_input.json'),
                    surface_sha256=sha256_file(output/'wing.xyz'),fm_input_sha256=sha256_file(output/'fm_input.npz'),
                    fm_input_frame=NATIVE_INPUT_FRAME,surface_sampling=sampling,
                    geometry_definition='new morphed native wing; independent CFD validation absent')
    identity['condition']['alpha_deg']=float(condition[0])
    result={'schema_version':1,'case_sha256':canonical_hash(identity),'identity':identity,
            'surface_path':'wing.xyz','fm_input_path':'fm_input.npz','native_input_path':'native_input.json',
            'geometry_measures':geometry_measures(native),
            'baseline_geometry_measures':geometry_measures(original),
            'surface_generation_seconds':time.perf_counter()-started,
            'cfd_executed':False,'matched_speedup_eligible':False,
            'mesh_quality':mesh['metadata']['minimum_scaled_jacobian']}
    write_json(output/'request.json',result)
    return output/'request.json',native,result


def shape_margins(coefficients, structural, measures, base_measures, cfg):
    l=cfg['limits']
    margins={'lift':coefficients['CL']/l['lift_coefficient_min']-1,
             'drag':1-coefficients['CD']/l['drag_coefficient_max'],
             'yield':1-structural['ks_failure']/l['ks_failure_max'],
             'tip_displacement':1-structural['tip_displacement_m']/l['tip_displacement_max_m'],
             'area':measures['projected_area_m2']/base_measures['projected_area_m2']/l['area_ratio_min']-1,
             'volume':measures['enclosed_geometric_volume_m3']/base_measures['enclosed_geometric_volume_m3']/l['volume_ratio_min']-1}
    if not np.isfinite(list(margins.values())).all():raise ValueError('Nonfinite task constraints')
    return {k:float(v) for k,v in margins.items()}
