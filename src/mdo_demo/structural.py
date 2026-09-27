"""Native-frame wingbox shell FEA with TACS and an explicit load-transfer map.

This is a small, unstiffened, isotropic shell wingbox for a one-way integration
pilot. It is not the original XRF1/STW model, a beam substitute, a buckling
analysis, or a completed two-way aeroelastic solver. SI units throughout.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

FRAME = "native_x_flow_y_vertical_z_negative_span"
AXES = "right-handed x streamwise, y vertical, z negative outboard; root row first"
TACS_SOURCE_REFERENCE = "https://smdogroup.github.io/tacs/examples/Example-Plate.html"


def _array(value, name, trailing_shape=None):
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    if trailing_shape is not None and (array.ndim != len(trailing_shape) + 1 or array.shape[1:] != trailing_shape):
        raise ValueError(f"{name} must have shape (N,{','.join(map(str, trailing_shape))})")
    return array


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array, dtype="<f8", order="C")
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _profile_bounds(section, x_values):
    """Intersect the actual closed section polygon with constant-native-x lines.

    Spar fractions refer to the section's native-x chord range. No airfoil,
    thickness ratio, or twist is invented. The two extreme intersections define
    lower/upper skin mid-surfaces of this reduced wingbox layout.
    """
    polygon = np.asarray(section, dtype=float)
    next_vertex = np.roll(polygon, -1, axis=0)
    result = []
    tol = max(float(np.ptp(polygon[:, 0])), 1.0) * 1e-12
    for x in x_values:
        intersections = []
        for first, second in zip(polygon, next_vertex):
            dx = second[0] - first[0]
            if abs(dx) <= tol:
                if abs(x - first[0]) <= tol:
                    intersections.extend((first[1], second[1]))
            else:
                fraction = (x - first[0]) / dx
                if -1e-12 <= fraction <= 1 + 1e-12:
                    intersections.append(first[1] + fraction * (second[1] - first[1]))
        if len(intersections) < 2 or max(intersections) - min(intersections) <= tol:
            raise ValueError("Wingbox skin extraction requires a nonzero section depth at both spars")
        result.append((min(intersections), max(intersections)))
    return np.asarray(result)


def build_wingbox_mesh(surface_vertices_m, *, n_span=12, n_chord=6, n_web=2,
                       front_spar=0.15, rear_spar=0.65, rib_every=3):
    """Build connected CQUAD4 skin/web/rib mid-surfaces from a physical wing.

    Input shape is (3, span_vertices, contour_vertices), with native z strictly
    decreasing outboard. Each section must lie in a constant-z plane. The
    physical root station is preserved (it need not be at z=0).
    The root perimeter is clamped in all six shell DOFs. Internal ribs every
    ``rib_every`` bays and a tip rib share skin/web nodes; there is no root rib.
    ``property_ids`` are 1=skins, 2=spars/ribs; node indices are zero-based.
    """
    for name, value in (("n_span", n_span), ("n_chord", n_chord), ("n_web", n_web), ("rib_every", rib_every)):
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError(f"{name} must be an integer from 1 to 100")
    if not 0 < front_spar < rear_spar < 1:
        raise ValueError("Require 0 < front_spar < rear_spar < 1")
    vertices = _array(surface_vertices_m, "surface_vertices_m")
    if vertices.ndim != 3 or vertices.shape[0] != 3 or vertices.shape[1] < 2 or vertices.shape[2] < 4:
        raise ValueError("surface_vertices_m must have shape (3,H+1,W+1)")
    sections = vertices.transpose(1, 2, 0)
    span_z = sections[:, :, 2].mean(axis=1)
    tolerance = max(float(np.ptp(span_z)), 1.0) * 1e-8
    if span_z[0] > tolerance or not np.all(np.diff(span_z) < 0):
        raise ValueError("Native structural frame requires nonpositive root z and strictly decreasing outboard z")
    if np.max(np.abs(sections[:, :, 2] - span_z[:, None])) > tolerance:
        raise ValueError("Structural section rows must be constant native z")
    source_eta = (span_z-span_z[0]) / (span_z[-1]-span_z[0])
    target_eta = np.linspace(0, 1, n_span + 1)
    sampled = np.empty((n_span + 1, sections.shape[1], 3))
    for contour in range(sections.shape[1]):
        for axis in range(3):
            sampled[:, contour, axis] = np.interp(target_eta, source_eta, sections[:, contour, axis])
    geometry = []
    for section in sampled:
        minimum, maximum = section[:, 0].min(), section[:, 0].max()
        chord = maximum - minimum
        if chord <= tolerance:
            raise ValueError("Section chord must be positive")
        xs = minimum + chord * np.linspace(front_spar, rear_spar, n_chord + 1)
        bounds = _profile_bounds(section, xs)
        geometry.append((xs, bounds, float(section[:, 2].mean())))
    nodes, lookup, node_station = [], {}, []

    def node(i, j, k):
        key = (i, j, k)
        if key not in lookup:
            xs, bounds, z = geometry[i]
            lookup[key] = len(nodes)
            nodes.append([xs[j], bounds[j, 0] + (bounds[j, 1] - bounds[j, 0]) * k / n_web, z])
            node_station.append(i)
        return lookup[key]

    quads, properties = [], []

    def element(corners, prop):
        quads.append([node(*corner) for corner in corners])
        properties.append(prop)

    for i in range(n_span):
        for j in range(n_chord):
            element([(i, j, n_web), (i, j+1, n_web), (i+1, j+1, n_web), (i+1, j, n_web)], 1)
            element([(i, j, 0), (i+1, j, 0), (i+1, j+1, 0), (i, j+1, 0)], 1)
        for k in range(n_web):
            element([(i, 0, k), (i, 0, k+1), (i+1, 0, k+1), (i+1, 0, k)], 2)
            element([(i, n_chord, k), (i+1, n_chord, k), (i+1, n_chord, k+1), (i, n_chord, k+1)], 2)
    rib_stations = sorted(set(range(rib_every, n_span + 1, rib_every)) | {n_span})
    for i in rib_stations:
        for j in range(n_chord):
            for k in range(n_web):
                element([(i, j, k), (i, j, k+1), (i, j+1, k+1), (i, j+1, k)], 2)
    nodes = np.asarray(nodes, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    q = nodes[quads]
    areas = (np.linalg.norm(np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]), axis=1)
             + np.linalg.norm(np.cross(q[:, 2]-q[:, 0], q[:, 3]-q[:, 0]), axis=1)) / 2
    if np.any(areas <= tolerance**2):
        raise ValueError("Degenerate structural shell element")
    # Positive area alone does not reject crossed/folded bilinear quads. Audit
    # orientation at every 2x2 shell integration point against the center.
    center_normal = np.cross((-q[:, 0]+q[:, 1]+q[:, 2]-q[:, 3])/4,
                             (-q[:, 0]-q[:, 1]+q[:, 2]+q[:, 3])/4)
    minimum_scaled_jacobian = 1.0
    for xi in (-1/math.sqrt(3), 1/math.sqrt(3)):
        for eta in (-1/math.sqrt(3), 1/math.sqrt(3)):
            dxi = np.array([-(1-eta), 1-eta, 1+eta, -(1+eta)])/4
            deta = np.array([-(1-xi), -(1+xi), 1+xi, 1-xi])/4
            tangent_xi = np.einsum("n,enj->ej", dxi, q)
            tangent_eta = np.einsum("n,enj->ej", deta, q)
            normal = np.cross(tangent_xi, tangent_eta)
            if np.any(np.sum(normal*center_normal, axis=1) <= tolerance**4):
                raise ValueError("Folded or degenerate structural shell quadrilateral")
            scaled = np.linalg.norm(normal, axis=1) / (np.linalg.norm(tangent_xi, axis=1)*np.linalg.norm(tangent_eta, axis=1))
            minimum_scaled_jacobian = min(minimum_scaled_jacobian, float(scaled.min()))
    station = np.asarray(node_station)
    mesh = {"nodes_m": nodes, "quads": quads, "property_ids": np.asarray(properties, dtype=np.int64),
            "root_node_ids": np.flatnonzero(station == 0), "tip_node_ids": np.flatnonzero(station == n_span),
            "element_areas_m2": areas,
            "metadata": {"coordinate_frame": FRAME, "axes": AXES, "element_type": "CQUAD4/Quad4Shell",
                         "source_surface_sha256": _array_hash(vertices), "n_span": n_span, "n_chord": n_chord,
                         "n_web": n_web, "front_spar": front_spar, "rear_spar": rear_spar,
                         "rib_every": rib_every, "rib_stations": rib_stations,
                         "span_m": float(span_z[0]-span_z[-1]), "root_z_m": float(span_z[0]),
                         "tip_z_m": float(span_z[-1]), "root_boundary": "all six DOFs clamped",
                         "minimum_scaled_jacobian": minimum_scaled_jacobian,
                         "section_rule": "native-x chord fractions; polygon upper/lower intersections",
                         "minimum_box_depth_m": float(min(np.min(b[:, 1]-b[:, 0]) for _, b, _ in geometry)),
                         "minimum_box_width_m": float(min(xs[-1]-xs[0] for xs, _, _ in geometry)),
                         "scope": "single semispan unstiffened isotropic shell box; skin/web/rib mid-surfaces only"}}
    mesh["mesh_sha256"] = _array_hash(nodes, quads, mesh["property_ids"], mesh["root_node_ids"])
    return mesh


def interpolate_point_motion(mesh, mapping, nodal_displacements):
    """Kinematic transpose pair of transfer_point_loads, for virtual-work checks.

    Small rotations only. This function does not deform any aerodynamic mesh or
    implement an aeroelastic iteration.
    """
    motion = _array(nodal_displacements, "nodal_displacements", (6,))
    if motion.shape[0] != len(mesh["nodes_m"]):
        raise ValueError("Nodal displacement count differs from mesh")
    ids, weights = mapping["point_node_ids"], mapping["point_weights"]
    offset = mapping["load_points_m"][:, None] - mesh["nodes_m"][ids]
    local = motion[ids]
    displacement = np.sum(weights[..., None] * (local[..., :3] + np.cross(local[..., 3:], offset)), axis=1)
    rotation = np.sum(weights[..., None] * local[..., 3:], axis=1)
    return displacement, rotation


def transfer_point_loads(mesh, points_m, forces_N, moments_Nm=None, *, reference_point_m=None, neighbors=4):
    """Spread each force over local shell nodes, retaining exact offset couples.

    This prototype uses inverse-distance weights over the nearest k nodes, not
    an experimentally calibrated skin/rib load path. Nodal ordering is generated
    mesh order; entries are [Fx,Fy,Fz,Mx,My,Mz]. Both the wrench and virtual-work
    identity are audited. Local stresses still require mesh/transfer studies.
    """
    nodes = _array(mesh["nodes_m"], "nodes_m", (3,))
    points = _array(points_m, "load_points_m", (3,))
    forces = _array(forces_N, "load_forces_N", (3,))
    moments = np.zeros_like(forces) if moments_Nm is None else _array(moments_Nm, "load_moments_Nm", (3,))
    if forces.shape != points.shape or moments.shape != points.shape:
        raise ValueError("Each load point needs one force and optional moment")
    if mesh["metadata"]["coordinate_frame"] != FRAME:
        raise ValueError("Structural mesh frame differs from native load contract")
    if type(neighbors) is not int or not 1 <= neighbors <= min(16, len(nodes)):
        raise ValueError("neighbors must be an integer from 1 to min(16,node count)")
    reference = np.zeros(3) if reference_point_m is None else _array(reference_point_m, "reference_point_m")
    if reference.shape != (3,):
        raise ValueError("reference_point_m must have shape (3,)")
    tolerance = max(mesh["metadata"]["span_m"], 1.0) * 1e-8
    if points[:, 2].max() > nodes[:, 2].max() + tolerance or points[:, 2].min() < nodes[:, 2].min() - tolerance:
        raise ValueError("Load points exceed the declared native structural semispan")
    ids = np.empty((len(points), neighbors), dtype=np.int64)
    weights = np.empty((len(points), neighbors))
    nodal = np.zeros((len(nodes), 6))
    for start in range(0, len(points), 512):
        stop = min(start + 512, len(points))
        squared = np.sum((points[start:stop, None] - nodes[None])**2, axis=2)
        # Stable ties select ascending generated node index.
        selected = np.argsort(squared, axis=1, kind="stable")[:, :neighbors]
        distance = np.sqrt(np.take_along_axis(squared, selected, axis=1))
        inverse = 1.0 / np.maximum(distance, tolerance * 1e-4)
        local_weights = inverse / inverse.sum(axis=1, keepdims=True)
        coincident = distance[:, 0] <= tolerance * 1e-4
        local_weights[coincident] = 0
        local_weights[coincident, 0] = 1
        ids[start:stop], weights[start:stop] = selected, local_weights
        split_force = local_weights[..., None] * forces[start:stop, None]
        arm = points[start:stop, None] - nodes[selected]
        split_moment = np.cross(arm, split_force) + local_weights[..., None] * moments[start:stop, None]
        for slot in range(neighbors):
            np.add.at(nodal[:, :3], selected[:, slot], split_force[:, slot])
            np.add.at(nodal[:, 3:], selected[:, slot], split_moment[:, slot])
    source_force = forces.sum(axis=0)
    source_moment = (np.cross(points-reference, forces) + moments).sum(axis=0)
    target_force = nodal[:, :3].sum(axis=0)
    target_moment = (np.cross(nodes-reference, nodal[:, :3]) + nodal[:, 3:]).sum(axis=0)
    force_error = float(np.linalg.norm(source_force-target_force))
    moment_error = float(np.linalg.norm(source_moment-target_moment))
    force_scale = max(float(np.linalg.norm(forces, axis=1).sum()), 1.0)
    moment_scale = max(float(np.linalg.norm(np.cross(points-reference, forces)+moments, axis=1).sum()), 1.0)
    mapping = {"nodal_loads": nodal, "point_node_ids": ids, "point_weights": weights,
               "load_points_m": points.copy(), "load_forces_N": forces.copy(), "load_moments_Nm": moments.copy()}
    # Deterministic arbitrary small virtual motion: a diagnostic of the full
    # transfer pair, not a measured physical displacement or convergence test.
    virtual = np.sin(np.arange(len(nodes)*6).reshape(-1, 6)*0.731) * 1e-4
    up, rotation = interpolate_point_motion(mesh, mapping, virtual)
    work_source = float(np.sum(forces*up) + np.sum(moments*rotation))
    work_target = float(np.sum(nodal*virtual))
    work_error = abs(work_source-work_target)
    work_scale = max(float(np.sum(np.abs(forces*up))+np.sum(np.abs(moments*rotation))), 1.0)
    passed = max(force_error/force_scale, moment_error/moment_scale, work_error/work_scale) <= 1e-11
    mapping["audit"] = {"passed": passed, "method": "local inverse-distance nodal force with exact offset couples",
                        "neighbors": neighbors, "reference_point_m": reference.tolist(),
                        "source_total_force_N": source_force.tolist(), "target_total_force_N": target_force.tolist(),
                        "source_total_moment_Nm": source_moment.tolist(), "target_total_moment_Nm": target_moment.tolist(),
                        "force_error_N": force_error, "moment_error_Nm": moment_error,
                        "virtual_work_error_J": work_error, "relative_tolerance": 1e-11,
                        "conservation_scope": "resultant force, moment and linearized virtual work; local stress accuracy unvalidated"}
    if not passed:
        raise RuntimeError("Structural load transfer failed force/moment/virtual-work conservation")
    return mapping


def validate_structure_config(material, thickness):
    material_names = {"rho", "E", "nu", "ys"}
    if not material_names.issubset(material) or set(material)-material_names-{"name"} or set(thickness) != {"skin_m", "web_m"}:
        raise ValueError("Explicit material rho/E/nu/ys (SI) and thickness skin_m/web_m required")
    mat = {key: _positive(material[key], key) for key in ("rho", "E", "ys")}
    nu = material["nu"]
    if isinstance(nu, (bool, np.bool_)) or not math.isfinite(float(nu)) or not -1 < float(nu) < 0.5:
        raise ValueError("nu must lie strictly between -1 and 0.5")
    mat["nu"] = float(nu)
    if "name" in material:
        if not isinstance(material["name"], str):
            raise ValueError("Optional material name must be a string")
        mat["name"] = material["name"]
    thick = {key: _positive(value, key) for key, value in thickness.items()}
    return mat, thick


def validate_wingbox_thickness(mesh, thickness):
    """Reject thicknesses that consume the opposing skin/web midsurface gap.

    These native-section depth/width checks are necessary geometric guards,
    not a general offset-shell collision test or thin-shell validity study.
    Two identical opposing shells occupy one full thickness between their
    midsurfaces, so the remaining gap must stay strictly positive.
    """
    for key, dimension in (("skin_m", "minimum_box_depth_m"),
                           ("web_m", "minimum_box_width_m")):
        value = _positive(thickness[key], key)
        gap = _positive(mesh["metadata"][dimension], dimension)
        if value >= gap:
            raise ValueError(f"{key}={value:g} m must be smaller than {dimension}={gap:g} m; opposing shells would overlap or touch")


def write_wingbox_bdf(mesh, material, thickness, path):
    """Write a standalone NASTRAN geometry/BC deck; callback supplies yield law."""
    mat, thick = validate_structure_config(material, thickness)
    validate_wingbox_thickness(mesh, thick)
    lines = ["SOL 101", "CEND", "SPC = 1", "BEGIN BULK", "$ SI: m, N, kg, Pa; native negative-z semispan",
             f"MAT1,1,{mat['E']:.14e},,{mat['nu']:.14e},{mat['rho']:.14e}",
             "$ Femap Property 1 : SKINS", f"PSHELL,1,1,{thick['skin_m']:.14e},1,,1",
             "$ Femap Property 2 : SPARS_RIBS", f"PSHELL,2,1,{thick['web_m']:.14e},1,,1"]
    for index, xyz in enumerate(mesh["nodes_m"], 1):
        lines.append(f"GRID,{index},,{xyz[0]:.14e},{xyz[1]:.14e},{xyz[2]:.14e}")
    for index, (quad, prop) in enumerate(zip(mesh["quads"], mesh["property_ids"]), 1):
        lines.append(f"CQUAD4,{index},{int(prop)}," + ",".join(str(int(i)+1) for i in quad))
    for index in mesh["root_node_ids"]:
        lines.append(f"SPC1,1,123456,{int(index)+1}")
    lines.append("ENDDATA")
    Path(path).write_text("\n".join(lines)+"\n", encoding="ascii")


def run_tacs_structure(mesh, nodal_loads, material, thickness, output, *, ks_weight=50.0,
                       residual_tolerance=1e-7, transfer_audit=None):
    """Solve a genuine linear-static Quad4Shell TACS model using COMM_SELF.

    Missing TACS/MPI is a dependency failure; no alternative mechanics model is
    substituted. Status=ok means the FE solve passed its residual check, not that
    the design is feasible or validated. KSFailure is yield, not buckling.
    """
    started = time.perf_counter()
    mat, thick = validate_structure_config(material, thickness)
    loads = _array(nodal_loads, "nodal_loads", (6,))
    if len(loads) != len(mesh["nodes_m"]):
        raise ValueError("Nodal load count differs from structural mesh")
    if mesh["metadata"]["coordinate_frame"] != FRAME:
        raise ValueError("Unexpected structural coordinate frame")
    ks_weight = _positive(ks_weight, "ks_weight")
    residual_tolerance = _positive(residual_tolerance, "residual_tolerance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"structure_result.json").exists():
        raise ValueError("A structural result already exists in this directory")
    bdf = output / "wingbox.bdf"
    write_wingbox_bdf(mesh, mat, thick, bdf)
    _json(output/"structure_input.json", {"schema_version": 1, "mesh": mesh["metadata"],
          "mesh_sha256": mesh["mesh_sha256"], "loads_sha256": _array_hash(loads), "material": mat,
          "thickness": thick, "transfer_audit": transfer_audit, "ks_weight": ks_weight,
          "residual_tolerance": residual_tolerance, "scope": "one-way linear-static shell analysis"})
    try:
        try:
            import tacs
            from mpi4py import MPI
            from tacs import TACS, pyTACS, constitutive, elements, functions
        except ImportError as exc:
            raise RuntimeError("TACS with mpi4py is required; use the verified structural runtime. No beam fallback exists.") from exc
        if MPI.COMM_WORLD.Get_size() != 1:
            raise ValueError("This small structural entry must run as one MPI process; COMM_SELF only")
        assembler = pyTACS(str(bdf), comm=MPI.COMM_SELF)

        def callback(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
            prop_id = kwargs.get("propID")
            if prop_id is None:
                # Older pyTACS may omit propID but retains explicit BDF labels.
                label = compDescript.upper()
                prop_id = 1 if "SKINS" in label else 2 if "SPARS_RIBS" in label else None
            if prop_id not in (1, 2):
                raise ValueError(f"Unrecognized wingbox property {prop_id}: {compDescript}")
            properties = constitutive.MaterialProperties(rho=mat["rho"], E=mat["E"],
                                                         nu=mat["nu"], ys=mat["ys"])
            constitutive_model = constitutive.IsoShellConstitutive(properties,
                                      t=thick["skin_m" if prop_id == 1 else "web_m"], tNum=dvNum)
            if any(kind != "CQUAD4" for kind in elemDescripts):
                raise ValueError("Only genuine Quad4Shell elements are supported")
            return [elements.Quad4Shell(None, constitutive_model) for _ in elemDescripts], [1.0]

        assembler.initialize(callback)
        problem = assembler.createStaticProblem("wingbox", options={"L2Convergence": 1e-10,
                         "L2ConvergenceRel": 1e-10, "subSpaceSize": 30, "nRestarts": 10})
        if problem.getVarsPerNode() != 6:
            raise RuntimeError("Wingbox requires six shell DOFs per node")
        problem.addFunction("mass", functions.StructuralMass)
        # The discrete KS upper envelope is explicit and supported by the pinned
        # runtime. It is a von-Mises/yield aggregate, not maximum stress itself.
        problem.addFunction("failure", functions.KSFailure, ksWeight=ks_weight, ftype="discrete")
        problem.addLoadToNodes(list(range(1, len(loads)+1)), loads.astype(TACS.dtype), nastranOrdering=True)
        setup_seconds = time.perf_counter() - started
        solve_start = time.perf_counter()
        solve_return = problem.solve()
        solve_seconds = time.perf_counter() - solve_start
        funcs = {}
        problem.evalFunctions(funcs)
        state = np.asarray(problem.getVariables()).reshape(-1, 6)
        coordinates = np.asarray(problem.getNodes()).reshape(-1, 3)
        if len(coordinates) != len(loads) or not np.isfinite(state).all():
            raise RuntimeError("Invalid TACS state/coordinate arrays")
        if np.iscomplexobj(state) and np.max(np.abs(state.imag)) > 1e-10:
            raise RuntimeError("Unexpected complex physical structural state")
        # pyTACS may reorder nodes. Explicitly match coordinates rather than
        # assuming generated/BDF order equals TACS state-vector order.
        distance = np.sum((mesh["nodes_m"][:, None]-coordinates.real[None])**2, axis=2)
        order = np.argmin(distance, axis=1)
        if len(set(order.tolist())) != len(order) or np.max(distance[np.arange(len(order)), order]) > 1e-16:
            raise RuntimeError("Cannot uniquely map TACS states back to generated structural nodes")
        displacement = state.real[order].copy()
        residual = np.zeros(problem.getNumVariables(), dtype=TACS.dtype)
        problem.getResidual(residual)
        residual_norm = float(np.linalg.norm(residual))
        free_loads = loads.copy()
        free_loads[mesh["root_node_ids"]] = 0.0
        free_load_norm = float(np.linalg.norm(free_loads))
        relative = residual_norm / max(free_load_norm, 1.0)
        mass, failure = float(np.real(funcs["wingbox_mass"])), float(np.real(funcs["wingbox_failure"]))
        if not all(math.isfinite(x) for x in (mass, failure, residual_norm, relative)) or mass <= 0:
            raise RuntimeError("Non-finite or invalid TACS outputs")
        translation_norm = np.linalg.norm(displacement[:, :3], axis=1)
        root_residual = float(np.max(np.abs(displacement[mesh["root_node_ids"]])))
        # Newer TACS returns a boolean; older verified releases return None.
        # In either case the independently evaluated algebraic residual and
        # constrained root displacement are required.
        solver_reported_converged = (bool(solve_return) if isinstance(solve_return, (bool, np.bool_)) else None)
        compliance = float(np.sum(free_loads * displacement))
        if compliance < -1e-10 * max(free_load_norm * float(np.linalg.norm(displacement)), 1.0):
            raise RuntimeError("Negative static compliance indicates invalid stiffness/state/load association")
        converged = (relative <= residual_tolerance and root_residual <= 1e-10
                     and solver_reported_converged is not False)
        np.savez_compressed(output/"structure_arrays.npz", nodes_m=mesh["nodes_m"], quads=mesh["quads"],
                            property_ids=mesh["property_ids"], root_node_ids=mesh["root_node_ids"],
                            tip_node_ids=mesh["tip_node_ids"], nodal_loads=loads,
                            nodal_displacements_and_rotations=displacement)
        problem.writeSolution(outputDir=str(output), baseName="wingbox")
        try:
            version = importlib.metadata.version("tacs")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(tacs, "__version__", "unavailable")
        area_mass = float(np.sum(mesh["element_areas_m2"] * np.where(mesh["property_ids"] == 1,
                                    thick["skin_m"], thick["web_m"])) * mat["rho"])
        result = {"schema_version": 1, "status": "ok" if converged else "not_converged", "backend": "TACS",
                  "scope": "one-way aero-structural integration; linear-static isotropic Quad4Shell wingbox",
                  "mass_kg": mass, "max_displacement_m": float(translation_norm.max()),
                  "tip_displacement_m": float(translation_norm[mesh["tip_node_ids"]].max()),
                  "tip_vertical_displacement_m": float(displacement[mesh["tip_node_ids"], 1].mean()),
                  "ks_failure": failure, "ks_failure_definition": "discrete KS von-Mises/yield failure index; no buckling model",
                  "ks_weight": ks_weight, "solve_seconds": solve_seconds, "setup_seconds": setup_seconds,
                  "total_seconds": time.perf_counter()-started, "timing_scope": "BDF, imports, setup, solve, diagnostics and output; excludes aero and load transfer",
                  "convergence": {"converged": converged, "residual_norm": residual_norm,
                     "relative_residual": relative, "tolerance": residual_tolerance,
                     "normalization": "TACS constrained algebraic residual / max(norm of unconstrained six-DOF load vector,1)",
                     "unconstrained_load_algebraic_norm": free_load_norm,
                     "solver_reported_converged": solver_reported_converged,
                     "root_displacement_rotation_max": root_residual},
                  "compliance_J": compliance, "strain_energy_J": 0.5*compliance,
                  "mesh_sha256": mesh["mesh_sha256"], "loads_sha256": _array_hash(loads),
                  "material": mat, "thickness": thick, "nodes": len(loads), "shell_elements": len(mesh["quads"]),
                  "mesh": mesh["metadata"], "transfer_audit": transfer_audit,
                  "shell_area_mass_estimate_kg": area_mass,
                  "mass_estimate_note": "Triangle-area estimate; TACS bilinear-surface quadrature is authoritative for warped quads",
                  "provenance": {"tacs_version": str(version), "tacs_module": str(tacs.__file__),
                     "structural_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "bdf_sha256": hashlib.sha256(bdf.read_bytes()).hexdigest(), "api_reference": TACS_SOURCE_REFERENCE,
                     "mpi_size": 1, "element": "elements.Quad4Shell", "constitutive": "IsoShellConstitutive"},
                  "limitations": ["No aeroelastic displacement feedback", "No geometric nonlinearities or buckling",
                     "No mesh-convergence or local load-transfer accuracy certification", "No original STW/XRF1 model reproduction",
                     "No structural self-weight/fuel/inertial loads unless explicitly supplied", "No validated MDO total derivatives"]}
        _json(output/"structure_result.json", result)
        return result
    except Exception as exc:
        _json(output/"structure_result.json", {"schema_version": 1, "status": "failed", "backend": "TACS",
              "error": f"{type(exc).__name__}: {exc}", "mesh_sha256": mesh["mesh_sha256"],
              "elapsed_seconds": time.perf_counter()-started, "fallback_used": False})
        raise


def run_structural_case(input_npz, config, output_dir):
    """File adapter used by run_structure.py; no pressure/force conversion here."""
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    allowed = {"schema_version", "coordinate_frame", "material", "thickness", "mesh", "transfer", "solver"}
    if set(config)-allowed or config.get("schema_version") != 1 or config.get("coordinate_frame") != FRAME:
        raise ValueError(f"Explicit schema_version=1 and {FRAME} config required")
    with np.load(input_npz, allow_pickle=False) as source:
        required = {"surface_vertices_m", "load_points_m", "load_forces_N"}
        if not required.issubset(source.files):
            raise ValueError(f"Input NPZ requires {sorted(required)}")
        mesh = build_wingbox_mesh(source["surface_vertices_m"], **config.get("mesh", {}))
        mapping = transfer_point_loads(mesh, source["load_points_m"], source["load_forces_N"],
                         source["load_moments_Nm"] if "load_moments_Nm" in source else None,
                         reference_point_m=source["moment_reference_m"] if "moment_reference_m" in source else None,
                         **config.get("transfer", {}))
    result = run_tacs_structure(mesh, mapping["nodal_loads"], config["material"], config["thickness"],
                               output_dir, transfer_audit=mapping["audit"], **config.get("solver", {}))
    np.savez_compressed(Path(output_dir)/"load_transfer.npz", **{key: value for key, value in mapping.items() if key != "audit"})
    return result
