"""Explicit synthetic data for testing report plumbing, never an acceleration result."""

import numpy as np

from .evaluation import evaluate
from .io import write_json
from .report import write_report


class SyntheticDataset:
    manifest = {"source": "Analytic synthetic fixture; not CFD", "seed": 7}

    def __len__(self):
        return 3

    def sample(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        eta, surface = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 2*np.pi, 81), indexing="ij")
        chord = 1 - .4*eta
        x = chord*(.5-.5*np.cos(surface)) + .3*eta
        y = eta*2
        z = .06*chord*np.sin(surface)
        cp = -.6*np.sin(surface)*(1-.3*eta)*(1+.1*index)
        fields = np.stack([cp, np.full_like(cp, .003*150), np.zeros_like(cp)])
        return {"sample_id": f"synthetic-{index}", "shape_id": index,
                "geometry": np.stack([x,z,y]), "condition": np.array([2., .78]),
                "fields": fields, "reference_coefficients": {"CL": .5+.02*index,"CD":.022+.001*index,"CM":-.1}}


class SyntheticPredictor:
    device = "CPU / synthetic"
    provenance = {"model": None, "synthetic": True}

    def predict(self, sample):
        return {"fields": sample["fields"]*1.025,
                "coefficients": {k:v*1.025 for k,v in sample["reference_coefficients"].items()}}


def run_smoke(output):
    report = evaluate(SyntheticDataset(), SyntheticPredictor(), output, limit=3)
    report.update(mode="synthetic_interface_smoke", reference_source="Analytic fixture; not CFD",
                  split_claim="Not a scientific dataset", speedup=None,
                  speedup_note="Synthetic test; timings are not CFD or neural inference measurements")
    write_json(str(output)+"/evaluation.json", report)
    return write_report(report, output)
