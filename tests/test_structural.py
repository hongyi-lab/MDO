"""Mechanics/contract tests; real TACS solve checks run only with TACS installed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mdo_demo.structural import (FRAME, build_wingbox_mesh, interpolate_point_motion,
                                run_tacs_structure, transfer_point_loads,
                                validate_structure_config, validate_wingbox_thickness,
                                write_wingbox_bdf)


MATERIAL = {"rho": 2780., "E": 73.1e9, "nu": .33, "ys": 270e6}
THICKNESS = {"skin_m": .002, "web_m": .002}


def rectangular_surface(root_z=-.3, span=2.7, sweep=0., taper=1.):
    # Independent, analytically measurable rectangular closed section.
    contour = np.array([[0., -.05], [1., -.05], [1., .05], [0., .05], [0., -.05]])
    out = np.empty((3, 3, len(contour)))
    for i, eta in enumerate(np.linspace(0, 1, 3)):
        out[0, i] = contour[:, 0]*(1+eta*(taper-1))+eta*sweep
        out[1, i] = contour[:, 1]*(1+eta*(taper-1))
        out[2, i] = root_z-eta*span
    return out


def simple_mesh(**kwargs):
    return build_wingbox_mesh(rectangular_surface(**kwargs), n_span=8, n_chord=4,
                             n_web=2, rib_every=4)


class StructureContractTests(unittest.TestCase):
    def test_mesh_preserves_offset_root_and_has_correct_box_area(self):
        mesh = simple_mesh()
        np.testing.assert_allclose(mesh["nodes_m"][mesh["root_node_ids"], 2], -.3, atol=1e-14)
        np.testing.assert_allclose(mesh["nodes_m"][mesh["tip_node_ids"], 2], -3., atol=1e-14)
        self.assertEqual(mesh["metadata"]["coordinate_frame"], FRAME)
        self.assertAlmostEqual(mesh["metadata"]["span_m"], 2.7)
        # Two skins, two webs, and two box ribs; no invented root plate.
        expected_area = 2*.5*2.7 + 2*.1*2.7 + 2*.5*.1
        self.assertAlmostEqual(mesh["element_areas_m2"].sum(), expected_area, places=12)
        self.assertGreater(mesh["metadata"]["minimum_scaled_jacobian"], .999999)

    def test_mesh_is_connected_and_every_node_is_attached(self):
        mesh = simple_mesh(sweep=1.2, taper=.35)
        quads = mesh["quads"]
        self.assertEqual(set(quads.ravel()), set(range(len(mesh["nodes_m"]))))
        adjacent = [set() for _ in mesh["nodes_m"]]
        for quad in quads:
            for node in quad:
                adjacent[node].update(quad)
        visited, remaining = set(), [0]
        while remaining:
            node = remaining.pop()
            if node not in visited:
                visited.add(node)
                remaining.extend(adjacent[node]-visited)
        self.assertEqual(len(visited), len(mesh["nodes_m"]))
        self.assertGreater(mesh["metadata"]["minimum_scaled_jacobian"], .1)

    def test_surface_frame_and_degeneracy_rejected(self):
        for surface in (rectangular_surface()[:, ::-1], -rectangular_surface(),
                        rectangular_surface(root_z=.3)):
            with self.assertRaises(ValueError):
                build_wingbox_mesh(surface)
        surface = rectangular_surface()
        surface[1] = 0.
        with self.assertRaisesRegex(ValueError, "nonzero section depth"):
            build_wingbox_mesh(surface)
        surface = rectangular_surface()
        surface[2, 1, 1] += .01
        with self.assertRaisesRegex(ValueError, "constant native z"):
            build_wingbox_mesh(surface)

    def test_point_force_and_couple_preserve_global_wrench(self):
        mesh = simple_mesh(sweep=1.2, taper=.5)
        rng = np.random.default_rng(12)
        points = np.column_stack((rng.uniform(0, 2, 50), rng.uniform(-.1, .1, 50),
                                  rng.uniform(-3, -.3, 50)))
        forces = rng.normal(size=(50, 3))*100
        moments = rng.normal(size=(50, 3))*4
        reference = np.array([2.3, -.4, -1.5])
        mapped = transfer_point_loads(mesh, points, forces, moments, reference_point_m=reference)
        loads = mapped["nodal_loads"]
        np.testing.assert_allclose(loads[:, :3].sum(0), forces.sum(0), rtol=1e-13, atol=1e-10)
        np.testing.assert_allclose((np.cross(mesh["nodes_m"]-reference, loads[:, :3])+loads[:, 3:]).sum(0),
                                   (np.cross(points-reference, forces)+moments).sum(0), rtol=1e-13, atol=1e-10)
        self.assertTrue(mapped["audit"]["passed"])

    def test_zero_resultant_force_does_not_erase_pure_couple(self):
        mesh = simple_mesh()
        points = np.array([[.2, 0., -2.7], [.6, 0., -2.7]])
        forces = np.array([[0., 100., 0.], [0., -100., 0.]])
        mapped = transfer_point_loads(mesh, points, forces)
        np.testing.assert_allclose(mapped["audit"]["target_total_force_N"], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(mapped["audit"]["target_total_moment_Nm"], [0, 0, -40], atol=1e-12)

    def test_rigid_body_motion_and_virtual_work_are_exact(self):
        mesh = simple_mesh()
        points = np.array([[.2, .01, -2.], [.7, -.03, -.9]])
        forces = np.array([[10, 40, -3], [2, -5, 8.]])
        moments = np.array([[.3, -.1, .2], [1., 2., -1.]])
        mapped = transfer_point_loads(mesh, points, forces, moments)
        translation, rotation = np.array([.02, -.03, .01]), np.array([.003, -.004, .002])
        motion = np.zeros((len(mesh["nodes_m"]), 6))
        motion[:, :3] = translation+np.cross(rotation, mesh["nodes_m"])
        motion[:, 3:] = rotation
        displacement, returned_rotation = interpolate_point_motion(mesh, mapped, motion)
        np.testing.assert_allclose(displacement, translation+np.cross(rotation, points), atol=1e-15)
        np.testing.assert_allclose(returned_rotation, np.broadcast_to(rotation, (2, 3)), atol=1e-15)
        arbitrary = np.random.default_rng(9).normal(size=motion.shape)*1e-4
        displacement, returned_rotation = interpolate_point_motion(mesh, mapped, arbitrary)
        self.assertAlmostEqual(np.sum(forces*displacement)+np.sum(moments*returned_rotation),
                               np.sum(mapped["nodal_loads"]*arbitrary), places=13)

    def test_loads_outside_actual_root_tip_are_rejected(self):
        mesh = simple_mesh()
        for z in (0., -.2, -3.1):
            with self.assertRaisesRegex(ValueError, "semispan"):
                transfer_point_loads(mesh, [[.4, 0., z]], [[0., 100., 0.]])

    def test_exact_node_load_stays_on_node_and_no_offset_couple(self):
        mesh = simple_mesh()
        node = len(mesh["nodes_m"])-1
        mapped = transfer_point_loads(mesh, mesh["nodes_m"][[node]], [[1., 2., 3.]])
        expected = np.zeros_like(mapped["nodal_loads"])
        expected[node, :3] = [1, 2, 3]
        np.testing.assert_allclose(mapped["nodal_loads"], expected, atol=1e-15)

    def test_translation_of_reference_does_not_change_nodal_loads(self):
        mesh = simple_mesh()
        points, force = [[.4, .02, -2.]], [[3., 20., -1.]]
        a = transfer_point_loads(mesh, points, force)
        offset = np.array([4., -2., .5])
        b = transfer_point_loads(mesh, points, force, reference_point_m=offset)
        np.testing.assert_array_equal(a["nodal_loads"], b["nodal_loads"])
        np.testing.assert_allclose(np.array(a["audit"]["source_total_moment_Nm"])-np.cross(offset, force[0]),
                                   b["audit"]["source_total_moment_Nm"], atol=1e-13)

    def test_material_validation_and_bdf_clamps_only_root(self):
        material, thickness = validate_structure_config({**MATERIAL, "name": "test material"}, THICKNESS)
        self.assertEqual(material["E"], MATERIAL["E"])
        for key, bad in (("rho", 0), ("E", float("nan")), ("nu", .5), ("nu", True), ("ys", -1)):
            with self.assertRaises(ValueError):
                validate_structure_config({**MATERIAL, key: bad}, THICKNESS)
        with self.assertRaises(ValueError):
            validate_structure_config(MATERIAL, {**THICKNESS, "skin_m": 0})
        mesh = simple_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"wingbox.bdf"
            write_wingbox_bdf(mesh, material, thickness, path)
            lines = path.read_text().splitlines()
        constraints = [line.split(",") for line in lines if line.startswith("SPC1")]
        self.assertEqual({int(c[3])-1 for c in constraints}, set(mesh["root_node_ids"]))
        self.assertTrue(all(c[2] == "123456" for c in constraints))
        self.assertEqual(sum(line.startswith("GRID,") for line in lines), len(mesh["nodes_m"]))
        self.assertEqual(sum(line.startswith("CQUAD4,") for line in lines), len(mesh["quads"]))
        # Real fields remain lexical floats, including integer-valued rho/E/xyz.
        self.assertTrue(all("e" in line.split(",")[3] for line in lines if line.startswith("GRID,")))

    def test_opposing_shell_thickness_must_leave_positive_gap(self):
        mesh = simple_mesh(taper=.5)
        self.assertAlmostEqual(mesh["metadata"]["minimum_box_depth_m"], .05)
        self.assertAlmostEqual(mesh["metadata"]["minimum_box_width_m"], .25)
        validate_wingbox_thickness(mesh, THICKNESS)
        for key, limit in (("skin_m", .05), ("web_m", .25)):
            for thickness in (limit, 1.1*limit):
                with self.assertRaisesRegex(ValueError, "overlap or touch"):
                    validate_wingbox_thickness(mesh, {**THICKNESS, key: thickness})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"invalid.bdf"
            with self.assertRaisesRegex(ValueError, "overlap or touch"):
                write_wingbox_bdf(mesh, MATERIAL, {**THICKNESS, "skin_m": .1}, path)
            self.assertFalse(path.exists())


@unittest.skipUnless(importlib.util.find_spec("tacs") is not None, "TACS unavailable; no structural solver mock")
class GenuineTACSPhysicsTests(unittest.TestCase):
    def test_linear_load_response_mass_stiffness_and_clamp(self):
        mesh = simple_mesh()
        loads = transfer_point_loads(mesh, [[.4, 0, -3]], [[0, 100, 0]])["nodal_loads"]
        with tempfile.TemporaryDirectory() as tmp:
            base = run_tacs_structure(mesh, loads, MATERIAL, THICKNESS, Path(tmp)/"base")
            doubled = run_tacs_structure(mesh, 2*loads, MATERIAL, THICKNESS, Path(tmp)/"double_force")
            thicker = run_tacs_structure(mesh, loads, MATERIAL, {k: 2*v for k, v in THICKNESS.items()}, Path(tmp)/"double_thickness")
        for result in (base, doubled, thicker):
            self.assertEqual(result["status"], "ok")
            self.assertLess(result["convergence"]["root_displacement_rotation_max"], 1e-10)
            self.assertGreater(result["strain_energy_J"], 0.)
        expected_mass = (2*.5*2.7+2*.1*2.7+2*.5*.1)*.002*MATERIAL["rho"]
        self.assertAlmostEqual(base["mass_kg"], expected_mass, places=8)
        self.assertAlmostEqual(doubled["mass_kg"], base["mass_kg"], places=10)
        self.assertAlmostEqual(doubled["tip_displacement_m"]/base["tip_displacement_m"], 2., places=7)
        self.assertAlmostEqual(doubled["compliance_J"]/base["compliance_J"], 4., places=7)
        self.assertAlmostEqual(thicker["mass_kg"]/base["mass_kg"], 2., places=8)
        self.assertLess(thicker["tip_displacement_m"], base["tip_displacement_m"])
        # Independent thin-walled Euler-Bernoulli estimate supplies a broad
        # physical-scale guard; shell/rib/shear/Poisson effects remain present.
        t = THICKNESS["skin_m"]
        second_moment = 2*.5*t*(.05**2) + 2*t*(.1**3)/12
        beam_tip = 100*(2.7**3)/(3*MATERIAL["E"]*second_moment)
        self.assertGreater(base["tip_vertical_displacement_m"], .5*beam_tip)
        self.assertLess(base["tip_vertical_displacement_m"], 1.5*beam_tip)


if __name__ == "__main__":
    unittest.main()
