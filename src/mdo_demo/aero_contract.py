"""Dimensional main-wing loads shared by FM, CFD, and structural solvers.

The contract is a native negative-z half-wing, with physical metres/Newtons.
It excludes the blunt trailing-edge patches and rounded tip for BOTH backends.
The FM's unfamiliar pretraining sampling remains a model-validation limitation;
it does not change the explicitly defined physical load integration domain.
No standard-atmosphere density or inferred dynamic pressure is substituted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .io import read_json, sha256_file, write_json
from .matched_cfd import (ADFLOW_SURFACE_SOURCE_SHA256, BASELINE_TWIST_DEGREES, FRAME_TRANSFORM_VERSION,
                          MODEL_INPUT_FRAME, NATIVE_INPUT_FRAME)

SCHEMA = "physical_mainwing_loads_v1"
MAINWING_PATCHES = [1, 2, 3, 5, 6, 7]
ARRAY_KEYS = ("panel_points_m", "panel_forces_N", "surface_vertices_m")
EXCLUDED_PATCHES = [4, 8, 9, 10, 11, 12, 13]


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array(value, name, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be finite" + (f" with shape {shape}" if shape else ""))
    return result


def _number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def make_request(case_id, geometry_sha256, *, condition, reference, coverage):
    """Bind physical conditions, geometry, reference point and coverage by hash.

    ``dynamic_pressure_pa`` must be supplied from the actual solver's AeroProblem
    or the identical pinned gas/viscosity model. Area is the half-wing reference
    area. No symmetry doubling is performed. ``span_bounds_m`` is [tip, root]
    in the native negative-z frame. Only the full open main-wing is supported.
    """
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id is required")
    if (not isinstance(geometry_sha256, str) or len(geometry_sha256) != 64
            or any(c not in "0123456789abcdef" for c in geometry_sha256)):
        raise ValueError("geometry_sha256 must be the native wing.xyz SHA256")
    names = ("alpha_deg", "mach", "reynolds", "reynolds_length_m", "temperature_k", "dynamic_pressure_pa")
    cond = {k: _number(condition[k], k, positive=(k != "alpha_deg")) for k in names}
    ref = {k: _number(reference[k], k, positive=True) for k in ("area_m2", "chord_m")}
    ref["moment_center_m"] = _array(reference["moment_center_m"], "moment_center_m", (3,)).tolist()
    bounds = _array(coverage["span_bounds_m"], "span_bounds_m", (2,)).tolist()
    if not bounds[0] < bounds[1] <= 0:
        raise ValueError("span_bounds_m must be ordered [negative tip, negative root]")
    if coverage.get("families") != ["mainwing"] or coverage.get("closed_TE") is not False:
        raise ValueError("Only mainwing excluding trailing-edge and tip patches is supported")
    identity = {"case_id": case_id, "geometry_sha256": geometry_sha256, "condition": cond,
                "reference": ref, "coverage": {"families": ["mainwing"], "closed_TE": False,
                "span_bounds_m": bounds, "native_patches": MAINWING_PATCHES.copy(),
                "excluded_native_patches": EXCLUDED_PATCHES.copy(), "complete_aircraft_wall": False},
                "coordinate_frame": "native_x_flow_y_vertical_z_negative_span",
                "units": {"length": "m", "force": "N", "moment": "N*m", "pressure": "Pa"}}
    return {"schema": SCHEMA, "request_sha256": _hash(identity), "identity": identity}


def validate_request(request):
    if request.get("schema") != SCHEMA or request.get("request_sha256") != _hash(request["identity"]):
        raise ValueError("Aerodynamic request schema/hash mismatch")
    ident = request["identity"]
    rebuilt = make_request(ident["case_id"], ident["geometry_sha256"], condition=ident["condition"],
                           reference=ident["reference"], coverage=ident["coverage"])
    if rebuilt != request:
        raise ValueError("Aerodynamic request is not the canonical physical contract")
    return ident


def _coverage_check(vertices, identity):
    vertices = _array(vertices, "surface_vertices_m")
    if vertices.ndim != 3 or vertices.shape[0] != 3 or min(vertices.shape[1:]) < 2:
        raise ValueError("surface_vertices_m must be (3,H+1,W+1)")
    span = vertices[2].mean(axis=1)
    if not np.all(np.diff(span) < 0):
        raise ValueError("Surface rows must run from root to tip in negative z")
    if not np.allclose(vertices[2], span[:, None], atol=1e-8, rtol=1e-8):
        raise ValueError("This contract requires planar constant-z span stations")
    if not np.allclose([span[-1], span[0]], identity["coverage"]["span_bounds_m"], atol=1e-7, rtol=1e-7):
        raise ValueError("Surface does not cover the requested span bounds")
    # A closed ring would silently integrate a different trailing-edge domain.
    if np.max(np.linalg.norm(vertices[:, :, -1] - vertices[:, :, 0], axis=0)) <= 1e-9:
        raise ValueError("Closed trailing edge is outside the open-mainwing contract")
    return vertices


def _quad_geometry(vertices):
    g = vertices.transpose(1, 2, 0)
    p0, p1, p2, p3 = g[:-1, :-1], g[:-1, 1:], g[1:, 1:], g[1:, :-1]
    area_vector = .5 * np.cross(p2-p0, p3-p1)
    area_norm = np.linalg.norm(area_vector, axis=-1)
    triangle_area = .5 * (np.linalg.norm(np.cross(p1-p0, p2-p0), axis=-1)
                           + np.linalg.norm(np.cross(p2-p0, p3-p0), axis=-1))
    if np.any(area_norm <= 1e-16) or not np.isfinite(triangle_area).all():
        raise ValueError("Degenerate surface panel")
    return .25*(p0+p1+p2+p3), area_vector, triangle_area


def _timing(timing):
    result = copy.deepcopy(timing)
    if not isinstance(result, dict) or not isinstance(result.get("scope"), str) or not result["scope"]:
        raise ValueError("Timing requires an explicit measured scope")
    for key, value in result.items():
        if key.endswith("_seconds") and value is not None:
            result[key] = _number(value, key)
            if result[key] < 0:
                raise ValueError("Measured time must not be negative")
    return result


def _assemble(request, vertices, points, forces, *, backend, timing, provenance, limitations):
    ident = validate_request(request)
    vertices = _coverage_check(vertices, ident)
    points, forces = _array(points, "panel_points_m"), _array(forces, "panel_forces_N")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape != forces.shape or not len(points):
        raise ValueError("Nonempty panel_points_m and panel_forces_N must both be (N,3)")
    cond, ref = ident["condition"], ident["reference"]
    total_force = forces.sum(axis=0)
    moment = np.cross(points - ref["moment_center_m"], forces).sum(axis=0)
    alpha = math.radians(cond["alpha_deg"])
    scale = cond["dynamic_pressure_pa"] * ref["area_m2"]
    cm = moment / (scale * ref["chord_m"])
    coefficients = {"CL": float((-total_force[0]*math.sin(alpha)+total_force[1]*math.cos(alpha))/scale),
                    "CD": float((total_force[0]*math.cos(alpha)+total_force[1]*math.sin(alpha))/scale),
                    "CY_span": float(total_force[2]/scale),
                    "CMx": float(cm[0]), "CMy": float(cm[1]), "CMz": float(cm[2]), "CM": float(cm[2])}
    return {"schema": SCHEMA, "request": copy.deepcopy(request), "backend": backend,
            "status": "ok", "interface_compatible": True, "accuracy_validated": False,
            "reference_eligible": False,
            "equal_accuracy_speedup_validated": False,
            "panel_points_m": points.copy(), "panel_forces_N": forces.copy(),
            "surface_vertices_m": vertices.copy(), "moment_center_m": ref["moment_center_m"].copy(),
            "total_force_N": total_force.tolist(), "total_moment_Nm": moment.tolist(),
            "coefficients": coefficients, "coverage": copy.deepcopy(ident["coverage"]),
            "timing": _timing(timing), "provenance": copy.deepcopy(provenance or {}),
            "limitations": list(limitations) + [
                "Mainwing-only loading benchmark: native trailing-edge patches 4/8 and tip patches 9-13 are excluded by BOTH backends; not total-wing loading."],
            "moment_convention": "sum((panel_point_m-reference_point_m) cross panel_force_N); CM=Mz/(q*S*c)",
            "symmetry": "one native negative-z half-wing; no force or area doubling"}


def from_fm_surface(request, sample, fields, *, timing, provenance=None):
    """Decode Cp,150*Cf_tau,300*Cf_z using pinned cfdpost quadrature.

    Geometry, free stream and force vectors undergo the inverse author frame
    transform, then the span reflection. Moments are recomputed in physical
    coordinates (a moment is an axial vector and must not be mirrored as force).
    Pretraining sampling validation is explicitly separate from interface use.
    """
    ident = validate_request(request)
    frame = sample.get("frame_contract", {})
    if (frame.get("version") != FRAME_TRANSFORM_VERSION or frame.get("verified") is not True
            or frame.get("source_frame") != NATIVE_INPUT_FRAME or frame.get("target_frame") != MODEL_INPUT_FRAME
            or frame.get("source_bundle_case_sha256") != ident["case_id"]):
        raise ValueError("Require the verified native bundle frame and matching case identity")
    theta = math.radians(BASELINE_TWIST_DEGREES)
    rotation = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0,0,1.]])
    if not np.allclose(frame["rotation_matrix_native_to_model"], rotation, rtol=0, atol=1e-12):
        raise ValueError("Frame rotation does not match the versioned author transform")
    cond = ident["condition"]
    # These quantities are not input channels of this model. They cannot be
    # changed in the physical request while pretending the FM conditioned on them.
    for key, fixed in (("reynolds", 20_000_000.), ("reynolds_length_m", 1.), ("temperature_k", 300.)):
        if not math.isclose(cond[key], fixed, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"FM cannot represent a changed fixed training condition: {key}")
    if not np.allclose(sample["condition"], [cond["alpha_deg"]+BASELINE_TWIST_DEGREES, cond["mach"]], rtol=1e-7, atol=1e-6):
        raise ValueError("FM input condition differs from physical request")
    if not math.isclose(float(sample["ref_area"]), ident["reference"]["area_m2"], rel_tol=1e-12):
        raise ValueError("Reference area mismatch")
    sampling = sample.get("sampling_contract", {})
    if (sampling.get("version") != "native_open_mainwing_arc_sampling_v1"
            or sampling.get("physical_surface_family") != "mainwing"
            or sampling.get("native_patches") != MAINWING_PATCHES):
        raise ValueError("Unknown FM physical surface coverage")
    field = _array(fields, "fields")
    vertices = _array(sample["original_geometry"], "original_geometry")
    if field.ndim != 3 or field.shape[0] != 3 or vertices.shape != (3,field.shape[1]+1,field.shape[2]+1):
        raise ValueError("Require fields (3,H,W) and physical vertex geometry (3,H+1,W+1)")
    centers = .25*(vertices[:,1:,1:]+vertices[:,1:,:-1]+vertices[:,:-1,1:]+vertices[:,:-1,:-1])
    input_centers = _array(sample["geometry"], "model center geometry", centers.shape)
    if not np.allclose(input_centers, centers, atol=1e-7, rtol=1e-6):
        raise ValueError("FM center inputs do not match integrated surface vertices")
    digest = hashlib.sha256()
    for key in ("original_geometry", "geometry", "condition"):
        value = np.ascontiguousarray(sample[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    if digest.hexdigest() != frame.get("model_input_sha256"):
        raise ValueError("FM model-input hash mismatch")
    points, area_vector, area = _quad_geometry(vertices)
    normal = area_vector / np.linalg.norm(area_vector, axis=-1, keepdims=True)
    g = vertices.transpose(1,2,0)
    tangent = g[:,1:,:2] - g[:,:-1,:2]
    norms = np.linalg.norm(tangent, axis=-1, keepdims=True)
    if np.any(norms <= 1e-16):
        raise ValueError("Degenerate chordwise tangent")
    tangent /= norms
    tangent = .5*(tangent[1:]+tangent[:-1])
    friction = np.concatenate(((field[1]/150.)[...,None]*tangent, (field[2]/300.)[...,None]), axis=-1)
    friction -= np.sum(friction*normal, axis=-1, keepdims=True)*normal
    force = (field[0,...,None]*normal + friction) * area[...,None] * cond["dynamic_pressure_pa"]
    model_to_native = np.diag([1.,1.,-1.]) @ rotation.T
    physical_vertices = np.einsum("ab,bij->aij", model_to_native, vertices)
    physical_points = points.reshape(-1,3) @ model_to_native.T
    physical_force = force.reshape(-1,3) @ model_to_native.T
    if "native_original_geometry" in sample:
        expected = np.array(sample["native_original_geometry"], dtype=float, copy=True)
        expected[2] *= -1
        if not np.allclose(physical_vertices, expected, rtol=1e-10, atol=1e-10):
            raise ValueError("Inverse frame transform disagrees with saved native geometry")
    source = copy.deepcopy(provenance or {})
    source.update(model_input_sha256=frame["model_input_sha256"],
                  decoded_float64_fields_sha256=_array_hash(field))
    output = _assemble(request, physical_vertices, physical_points, physical_force,
                       backend="aerotransformer", timing=timing, provenance=source,
                       limitations=["No independent CFD accuracy or confidence qualification is implied.",
                         "Closed-TE/exact-chord/span-truncation sampling used in pretraining is not reproduced.",
                         "FM panels approximate the same mainwing physical surface with a different tessellation."])
    output["frame_contract"] = copy.deepcopy(frame)
    output["sampling_contract"] = copy.deepcopy(sampling)
    output["quadrature"] = "pinned cfdpost: diagonal unit normal; two-triangle scalar area; projected decoded tangential friction"
    return output


def read_native_surface(path):
    """Read the pinned 13-patch PLOT3D surface without CFD/MPI dependencies."""
    tokens = Path(path).read_text().split()
    blocks = int(tokens[0])
    if blocks != 13:
        raise ValueError("Require the pinned 13-patch native surface")
    dims = np.asarray(tokens[1:1+3*blocks], dtype=int).reshape(blocks,3)
    values = _array(tokens[1+3*blocks:], "PLOT3D coordinates")
    result, offset = [], 0
    for ni,nj,nk in dims:
        if nk != 1 or min(ni,nj) < 2:
            raise ValueError("Expected two-dimensional native surface patches")
        size = int(3*ni*nj)
        result.append(values[offset:offset+size].reshape(3,nj,ni).copy())
        offset += size
    if offset != len(values):
        raise ValueError("Unexpected PLOT3D coordinate count")
    return result


def native_mainwing_vertices(path):
    """Stitch exact native lower/front/upper patches; exclude TE and tip."""
    blocks = read_native_surface(path)
    sides = []
    for start in (0,4):
        lower, front, upper = blocks[start:start+3]
        if not np.allclose(lower[:,:,-1], front[:,:,0], atol=1e-9, rtol=0) or not np.allclose(front[:,:,-1], upper[:,:,0], atol=1e-9, rtol=0):
            raise ValueError("Main-wing patch chord edges do not meet")
        sides.append(np.concatenate((lower[:,:,:-1],front[:,:,:-1],upper),axis=2))
    if not np.allclose(sides[0][:,-1],sides[1][:,0],atol=1e-9,rtol=0):
        raise ValueError("Inner and outer span patches do not meet")
    return np.concatenate((sides[0][:,:-1],sides[1]),axis=1)


def match_native_cells(wing_xyz, centers, zone_ids, zone_info):
    """Bind exported wall cells to native cells with storage-precision evidence.

    CGNS RealSingle coordinates are rounded *vertices*, not rounded centroids.
    Reconstruct that exact conversion from the original PLOT3D surface. The
    original geometric tolerance must still hold after this reconstruction.
    Coordinate tolerances are derived here, never accepted from export metadata.
    """
    from scipy.spatial import cKDTree
    blocks = read_native_surface(wing_xyz)
    original = np.concatenate([_quad_geometry(block)[0].reshape(-1,3) for block in blocks])
    centers = _array(centers,"CGNS wall cell centers",original.shape)
    zone_ids = np.asarray(zone_ids)
    if zone_ids.shape != (len(original),) or zone_ids.dtype.kind not in "iu":
        raise ValueError("Require one integer CGNS zone id per native wall cell")
    tree = cKDTree(original)
    distance,index = tree.query(centers)
    if not np.array_equal(np.sort(index),np.arange(len(original))):
        raise ValueError("CFD wall cells must have a complete one-to-one native cell mapping")
    separation = tree.query(original,k=2)[0][:,1]
    if not np.isfinite(separation).all() or np.any(separation <= 0):
        raise ValueError("Native surface has coincident or invalid cell centers")
    zone_lookup = {int(item["zone"]):item for item in zone_info}
    if len(zone_lookup) != len(zone_info):
        raise ValueError("Duplicate CGNS zone metadata")
    base = 1e-8*max(1.,float(np.ptp(original,axis=0).max()))
    cache, audits = {}, []
    for zone in np.unique(zone_ids):
        info = zone_lookup.get(int(zone),{})
        if info.get("selected_for_body_loads") is not True:
            raise ValueError("CFD cell belongs to a non-body or unrecorded CGNS zone")
        types = info.get("coordinate_storage_types",{})
        if set(types) != {"CoordinateX","CoordinateY","CoordinateZ"}:
            raise ValueError("Missing actual CGNS coordinate storage types")
        kind = tuple(types[key] for key in ("CoordinateX","CoordinateY","CoordinateZ"))
        if any(k not in ("RealSingle","RealDouble") for k in kind):
            raise ValueError("Unsupported CGNS coordinate storage precision")
        if kind not in cache:
            rounded_centers, bounds = [], []
            for block in blocks:
                rounded, roundoff = [], []
                for axis,key in enumerate(kind):
                    dtype = np.float32 if key == "RealSingle" else np.float64
                    stored = block[axis].astype(dtype)
                    rounded.append(stored.astype(float))
                    roundoff.append(.5*np.spacing(np.abs(stored)).astype(float))
                rounded = np.asarray(rounded)
                roundoff = np.asarray(roundoff)
                def averages(a):
                    return .25*(a[:,:-1,:-1]+a[:,1:,:-1]+a[:,:-1,1:]+a[:,1:,1:])
                rounded_centers.append(averages(rounded).reshape(3,-1).T)
                bounds.append(averages(roundoff).reshape(3,-1).T)
            cache[kind] = np.concatenate(rounded_centers),np.concatenate(bounds)
        expected, bound = cache[kind]
        mask = zone_ids == zone
        mapped = index[mask]
        axis_bound = np.max(bound[mapped],axis=0)
        representation_bound = float(np.linalg.norm(axis_bound))
        tolerance = base+representation_bound
        nearest_spacing = float(np.min(separation[mapped]))
        if tolerance >= .1*nearest_spacing:
            raise ValueError("Coordinate precision is insufficient to disambiguate nearby native cells")
        maximum = float(np.max(distance[mask]))
        residual = float(np.max(np.linalg.norm(centers[mask]-expected[mapped],axis=1)))
        if maximum > tolerance or residual > base:
            raise ValueError(f"CGNS zone {zone} differs from native geometry beyond recorded storage rounding: "
                             f"distance={maximum}, bound={tolerance}, rounded residual={residual}, base={base}")
        audits.append({"zone":int(zone),"coordinate_storage_types":types.copy(),
                       "cell_count":int(mask.sum()),"max_original_distance_m":maximum,
                       "max_rounded_reconstruction_error_m":residual,"roundoff_bound_m":representation_bound,
                       "derived_tolerance_m":tolerance,"min_other_native_cell_distance_m":nearest_spacing,
                       "tolerance_to_min_spacing_ratio":tolerance/nearest_spacing})
    return index, {"version":"native_cgns_vertex_storage_match_v1","verified":True,
                   "one_to_one_native_cell_match":True,"native_cell_count":len(original),
                   "base_geometry_tolerance_m":base,
                   "cell_coordinate_match_max_m":float(np.max(distance)),
                   "cell_coordinate_match_tolerance_m":max(z["derived_tolerance_m"] for z in audits),
                   "rounding_reconstruction":"Original PLOT3D vertices cast per coordinate to recorded CGNS storage dtype before centroid averaging",
                   "ambiguity_rule":"derived tolerance < 0.1 * distance to nearest other native cell",
                   "zones":audits}


def from_native_export(request, wing_xyz, export_npz, export_json, solver_result, *,
                       coefficient_audit_atol=1e-5, coefficient_audit_rtol=1e-3,
                       allow_unconverged_interface_audit=False):
    """Reconstruct native dimensional loads from a converged, identity-bound export.

    Native pressure uses the outward face area vector with a minus sign. Skin
    friction is the global traction on the body, scaled by q and face area.
    A solver-family CL/CD/CM comparison is required; failure returns diagnostic
    loads with ``status=load_audit_failed`` and ``interface_compatible=False``.
    These tolerances check integration consistency, not surrogate accuracy.
    Explicit ``allow_unconverged_interface_audit`` can inspect a failed solve's
    force decoding only. Such outputs always remain diagnostic and cannot enter
    a paired comparison or a structural/optimization baseline.
    """
    ident = validate_request(request)
    if not isinstance(allow_unconverged_interface_audit, bool):
        raise ValueError("allow_unconverged_interface_audit must be an explicit boolean")
    meta = read_json(export_json) if isinstance(export_json,(str,Path)) else copy.deepcopy(export_json)
    result = read_json(solver_result) if isinstance(solver_result,(str,Path)) else copy.deepcopy(solver_result)
    if sha256_file(wing_xyz) != ident["geometry_sha256"]:
        raise ValueError("Native surface geometry hash mismatch")
    if meta.get("case_sha256") != ident["case_id"] or result.get("case_sha256") != ident["case_id"]:
        raise ValueError("CFD/export physical case identity mismatch")
    conv = result.get("convergence", {})
    relative = conv.get("residual_relative_to_freestream")
    required = conv.get("required_l2_convergence")
    converged = (result.get("status") == "ok" and conv.get("converged") is True
                 and conv.get("solve_failed") is False and conv.get("fatal_failed") is False
                 and isinstance(relative, (int, float)) and math.isfinite(relative)
                 and isinstance(required, (int, float)) and math.isfinite(required)
                 and 0 < required <= 1e-10 and 0 <= relative <= required)
    if not converged and not allow_unconverged_interface_audit:
        raise ValueError("Unconverged CFD cannot enter the common load interface")
    pressure_audit = meta.get("pressure_convention_audit", {})
    if (meta.get("one_to_one_native_cell_match") is not True or pressure_audit.get("verified") is not True
            or pressure_audit.get("source_sha256") != ADFLOW_SURFACE_SOURCE_SHA256
            or pressure_audit.get("reviewed_source_sha256") != ADFLOW_SURFACE_SOURCE_SHA256
            or meta.get("coordinate_convention") != "native CFD negative-z span"):
        raise ValueError("Require audited native coordinates and pressure convention")
    if sha256_file(export_npz) != meta["output_sha256"]:
        raise ValueError("Native field export hash mismatch")
    source_condition = result["identity"]["condition"]
    if result["identity"].get("geometry_sha256") != ident["geometry_sha256"]:
        raise ValueError("CFD result is associated with another native surface geometry")
    if result.get("function_groups", {}).get("mainwing") != "mainwing":
        raise ValueError("Require explicit mainwing solver family coefficients")
    for name in ("mach", "reynolds", "reynolds_length_m", "temperature_k"):
        if not math.isclose(float(source_condition[name]),ident["condition"][name],rel_tol=1e-10):
            raise ValueError(f"CFD condition mismatch: {name}")
    if not math.isclose(float(result["solved_alpha_deg"]),ident["condition"]["alpha_deg"],rel_tol=0,abs_tol=1e-6):
        raise ValueError("CFD solved angle differs from request")
    if result["identity"]["reference"] != ident["reference"]:
        raise ValueError("CFD reference area/chord/moment point mismatch")
    blocks = read_native_surface(wing_xyz)
    all_points, vectors, patches = [], [], []
    for patch, vertices in enumerate(blocks,1):
        points, area_vector, _ = _quad_geometry(vertices)
        all_points.append(points.reshape(-1,3))
        vectors.append(area_vector.reshape(-1,3))
        patches.extend([patch]*points.shape[0]*points.shape[1])
    points, area_vector, patches = np.concatenate(all_points), np.concatenate(vectors), np.asarray(patches)
    with np.load(export_npz,allow_pickle=False) as archive:
        data = {k:np.array(archive[k],copy=True) for k in ("centers","cp","cf_xyz","native_cell_index","native_patch","mainwing_mask")}
        if "cgns_zone" in archive:
            data["cgns_zone"] = np.array(archive["cgns_zone"],copy=True)
    index = data["native_cell_index"]
    if (index.dtype.kind not in "iu" or index.shape != (len(points),)
            or not np.array_equal(np.sort(index),np.arange(len(points)))):
        raise ValueError("Native cell mapping must be a complete one-to-one permutation")
    if not np.array_equal(data["native_patch"],patches[index]) or not np.array_equal(data["mainwing_mask"],np.isin(patches[index],MAINWING_PATCHES)):
        raise ValueError("Native cell coverage labels disagree with geometry")
    centers = _array(data["centers"],"export centers",points.shape)
    coordinate_audit = None
    if "zones" in meta:
        if "cgns_zone" not in data:
            raise ValueError("CGNS storage metadata requires saved per-cell zone ids")
        verified_index,coordinate_audit = match_native_cells(wing_xyz,centers,data["cgns_zone"],meta["zones"])
        if not np.array_equal(index,verified_index):
            raise ValueError("Saved cell mapping differs from reconstructed geometry match")
    else:
        tolerance = 1e-8*max(1.,float(np.ptp(points,axis=0).max()))
        if np.max(np.linalg.norm(centers-points[index],axis=1)) > tolerance:
            raise ValueError("Exported cell coordinates do not match original geometry")
    cp = _array(data["cp"],"Cp",(len(points),))
    cf = _array(data["cf_xyz"],"Cf_xyz",points.shape)
    selected = np.isin(patches[index],MAINWING_PATCHES)
    vector = area_vector[index][selected]
    force = (-cp[selected,None]*vector + cf[selected]*np.linalg.norm(vector,axis=1,keepdims=True))*ident["condition"]["dynamic_pressure_pa"]
    timing = copy.deepcopy(result["timing"])
    timing["surface_export_seconds"] = meta.get("postprocess_seconds")
    timing["scope"] += "; surface_export_seconds is separately measured and not included in solver total"
    output = _assemble(request,native_mainwing_vertices(wing_xyz),points[index][selected],force,
                       backend="adflow",timing=timing,provenance={"export":meta,"solver":result.get("solver",{}),
                           "solver_result_content_sha256": _hash(result),
                           "solver_result_file_sha256": sha256_file(solver_result) if isinstance(solver_result,(str,Path)) else None,
                           "native_geometry_sha256": ident["geometry_sha256"],
                           "native_field_export_sha256": meta["output_sha256"]},
                       limitations=["Native finite-volume tessellation differs from the FM reference-grid quadrature."])
    truth = result["group_coefficients"]["mainwing"]
    atol, rtol = _number(coefficient_audit_atol,"coefficient_audit_atol"), _number(coefficient_audit_rtol,"coefficient_audit_rtol")
    if min(atol,rtol) < 0:
        raise ValueError("Integration audit tolerances must be nonnegative")
    errors = {k:abs(output["coefficients"][k]-_number(truth[k],k)) for k in ("CL","CD","CMx","CMy","CMz")}
    passed = all(errors[k] <= atol+rtol*abs(truth[k]) for k in errors)
    output["integration_audit"] = {"passed":passed,"absolute_errors":errors,"atol":atol,"rtol":rtol,
                                   "comparison":"same mainwing family; physical reference point; integration consistency only"}
    output["solver_family_coefficients"] = copy.deepcopy(truth)
    output["solver_convergence"] = copy.deepcopy(conv)
    output["coordinate_match_audit"] = coordinate_audit
    output["quadrature"] = "native finite-volume outward diagonal area vector; -Cp*n_area + global body Cf*norm(n_area)"
    if not passed:
        output.update(status="load_audit_failed",interface_compatible=False)
    if not converged:
        output.update(status="diagnostic_only", interface_compatible=False)
        output["limitations"].append("CFD did not meet the frozen 1e-10 convergence requirement. Loads are for interface debugging only.")
    output["reference_eligible"] = bool(passed and converged)
    return output


def assert_same_physical_request(first, second):
    """Require identical geometry, flow, reference and coverage for paired loads."""
    for output in (first,second):
        validate_request(output["request"])
        if output.get("status") != "ok" or not output.get("interface_compatible"):
            raise ValueError("A load output did not pass its interface checks")
    if first["request"]["request_sha256"] != second["request"]["request_sha256"]:
        raise ValueError("Aerodynamic backends do not share the same physical request")
    return True


def save_loads(output, json_path):
    """Write reviewable JSON metadata plus hash-checked, non-pickled NPZ arrays."""
    validate_request(output["request"])
    path = Path(json_path)
    if path.suffix.lower() != ".json":
        raise ValueError("Load archive path must end in .json")
    arrays_path = path.with_suffix(".npz")
    if path.exists() or arrays_path.exists():
        raise ValueError("Choose fresh archive paths to preserve previous experiment records")
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays = {k:_array(output[k],k) for k in ARRAY_KEYS}
    np.savez_compressed(arrays_path,**arrays)
    metadata = {k:copy.deepcopy(v) for k,v in output.items() if k not in ARRAY_KEYS}
    metadata["arrays"] = {"path":arrays_path.name,"sha256":sha256_file(arrays_path)}
    write_json(path,metadata)
    return path


def load_loads(json_path):
    """Read the archive and verify both request identity and numerical integrals."""
    path = Path(json_path)
    output = read_json(path)
    arrays_meta = output.pop("arrays")
    if Path(arrays_meta["path"]).name != arrays_meta["path"]:
        raise ValueError("Archive arrays must be a sibling filename")
    arrays_path = path.parent/arrays_meta["path"]
    if sha256_file(arrays_path) != arrays_meta["sha256"]:
        raise ValueError("Load archive arrays hash mismatch")
    with np.load(arrays_path,allow_pickle=False) as archive:
        for key in ARRAY_KEYS:
            output[key] = np.array(archive[key],copy=True)
    check = _assemble(output["request"],output["surface_vertices_m"],output["panel_points_m"],output["panel_forces_N"],
                      backend=output["backend"],timing=output["timing"],provenance=output["provenance"],limitations=output["limitations"])
    for key in ("total_force_N","total_moment_Nm","moment_center_m"):
        if not np.allclose(output[key],check[key],rtol=1e-12,atol=1e-10):
            raise ValueError(f"Archived {key} is inconsistent with physical panel loads")
    for key,value in check["coefficients"].items():
        if not math.isclose(output["coefficients"][key],value,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("Archived coefficients are inconsistent with physical loads")
    return output
