"""Explicit STW-native to released AeroTransformer input contract.

This converts coordinates and units; it does not change the physical case,
clamp angles to training limits, or claim matching training surface coverage.
"""
import numpy as np

CONTRACT_VERSION = "stw_native_to_atsurf_v2"


def encode_condition(*, alpha_deg, mach):
    values = np.array([alpha_deg, mach], dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or mach <= 0:
        raise ValueError("Finite scalar AoA in degrees and positive Mach required")
    encoded = values.astype(np.float32)
    if not np.isfinite(encoded).all():
        raise ValueError("Conditions overflow float32")
    return encoded


def encode_stw_surface(native_vertices_m, *, root_chord_m, reference_half_area_m2):
    """Native (span,chord,xyz) [stream,span,vertical] -> model [x,z,y]."""
    native = np.asarray(native_vertices_m, dtype=np.float64)
    if native.shape != (129, 257, 3) or not np.isfinite(native).all():
        raise ValueError("Expected finite native surface (129,257,3)")
    if not np.isfinite([root_chord_m, reference_half_area_m2]).all() or min(root_chord_m, reference_half_area_m2) <= 0:
        raise ValueError("Positive finite reference length and half-wing area required")
    if not np.allclose(native[:, 0], native[:, -1], rtol=0, atol=1e-10):
        raise ValueError("Expected explicitly closed surface rings")
    vertices = native[..., [0, 2, 1]].transpose(2, 0, 1) / root_chord_m
    centers = .25 * (vertices[:, 1:, 1:] + vertices[:, :-1, 1:]
                     + vertices[:, 1:, :-1] + vertices[:, :-1, :-1])
    return {"original_geometry": vertices, "geometry": centers.astype(np.float32),
            "ref_area": float(reference_half_area_m2 / root_chord_m**2)}


def decode_stw_vertices(model_vertices, *, root_chord_m):
    vertices = np.asarray(model_vertices, dtype=np.float64)
    if vertices.shape != (3, 129, 257) or not np.isfinite(vertices).all() or not np.isfinite(root_chord_m) or root_chord_m <= 0:
        raise ValueError("Invalid model vertices or reference length")
    return vertices.transpose(1, 2, 0)[..., [0, 2, 1]] * root_chord_m


def describe_contract():
    return {"version": CONTRACT_VERSION, "native_axes": ["streamwise", "spanwise", "vertical"],
            "native_length_unit": "m", "model_axes": ["streamwise", "vertical", "spanwise"],
            "model_length_unit": "root chord", "model_geometry": "four-vertex cell centers, float32",
            "condition_order": ["alpha_deg", "Mach"], "condition_dtype": "float32",
            "condition_scaling": "none", "angle_clipping": False, "geometry_rotation_deg": 0.,
            "condition_offset_deg": 0., "reference_area": "half-wing planform area / root_chord_m**2",
            "field_channels": ["Cp", "150*Cf_stream", "300*Cf_span"],
            "scope": "STW untwisted physical root; not a generic CRM frame converter",
            "exact_training_sampling_verified": False,
            "Re_temperature_conditioning": "not encoded by the released model; unchanged"}
