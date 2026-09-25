# Model, dataset, and conventions

This demo uses the **released pretrained AeroTransformer** on a small, real
CRMpert subset. It does not invent a neural model or CFD labels, and it does
not claim to reproduce the paper's fine-tuned results or a full MDO run.

## Pinned sources

| Asset | Revision | Source and license |
|---|---|---|
| Paper training/simulation scripts | `3dc350ff69e354d1451eae468c686a368bd6151f` | [tum-pbs/AeroTransformer](https://github.com/tum-pbs/AeroTransformer), Apache-2.0 |
| Model implementation | `ff3abda23e10e1073c07ffd78dad96979e940c77` | [YangYunjia/floGen](https://github.com/YangYunjia/floGen), MIT, with third-party notices retained upstream |
| Surface integration | `c9fb313a4f3f3912a2e1a1aaf56e9e7a338bfa5a` | [YangYunjia/cfdpost](https://github.com/YangYunjia/cfdpost), MIT |
| Weights/configs | `698d70095a00d6f4a25f870e8de0772fc12b68f7` | [thuerey-group/AeroTransformer](https://huggingface.co/thuerey-group/AeroTransformer), model-card MIT |
| CRMpert | `88ece28b846fd1d9870933252db556cf97d30ae0` | [thuerey-group/CRMpert](https://huggingface.co/datasets/thuerey-group/CRMpert), CC-BY-SA-4.0 |

Paper: Yunjia Yang et al., *Towards a Foundation-Model Paradigm for Aerodynamic
Prediction in Three-dimensional Design*, [arXiv:2604.18062](https://arxiv.org/abs/2604.18062).
The released pretrained checkpoints are **not fine-tuned CRMpert folds**. The
pinned model repository contains S/M/L weights, without paper fold assignments
or the corresponding task-fine-tuned checkpoints. Different models mentioned
in the separate `yunplus/AeroTransformer` API are not silently substituted.

## Download size and provenance

From the project root:

```bash
python scripts/fetch_assets.py --model ATsurf_S --samples 8
```

Weights are 4.33 MB (S), 15.16 MB (M), or 58.22 MB (L). A default eight-shape
subset needs about 19 MB of numeric arrays. The full CRMpert arrays would be
about 2.14 GB; no full SuperWing or volumetric files are downloaded.

The downloader reads NPY headers, fetches selected rows with HTTP Range, and
requires a valid `206 Content-Range` response. It fails if Range is ignored.
Its small full downloads, including the index and model weights, are SHA256
checked against the pinned public LFS hashes. Large source arrays are **not**
downloaded whole, so their complete hashes cannot be verified locally: the
manifest explicitly records this distinction and hashes every fetched byte
range plus each assembled local file. HTTPS and immutable source revisions
identify the sliced source. No credentials are needed for these public assets.

Defaults choose the first available condition from evenly spaced shape IDs.
This is a deterministic **convenience subset**, not a representative benchmark
sample and not a paper train/test split. The sample and geometry IDs remain
the IDs of the original release. Use `--sample-ids 0,20,40` to select exact rows,
and `--data-dir assets/CRMpert-other` to preserve an existing subset. Evaluation
is zero-shot; any later fine-tuning must split by **wing shape**, never randomly
mix conditions of the same wing across train/test.

## Exact interface

```python
from mdo_demo.dataset import load_dataset
from mdo_demo.aerotransformer import AeroTransformerPredictor

dataset = load_dataset("assets/CRMpert")
sample = dataset.sample(0)
predictor = AeroTransformerPredictor("assets/AeroTransformer/ATsurf_S", device="cuda:0")
result = predictor.predict(sample)
```

| Quantity | Meaning |
|---|---|
| `geometry` | Float32 cell-center coordinates, `(3,128,256)`, channels x/y/z with z spanwise |
| `original_geometry` | Vertex coordinates, `(3,129,257)`, needed for force integration |
| `condition` | `[angle_of_attack_in_degrees, Mach]` from index columns 2:4 (zero-based) |
| `fields` | `(3,128,256)`: `Cp`, `150*Cf_stream`, `300*Cf_span` |
| `ref_area` | Half-wing reference area from index column 4 |
| `reference_coefficients` | CL/CD/CM integrated on the reference surface, columns 9:12 |
| `solver_coefficients` | CL/CD/CM from the original CFD solver mesh, columns 6:9 |

`geom0.npy` is used as the published cell-center geometry. Upstream training
scripts use a local filename `geom.npy`; the released dataset card calls the
corresponding structured input `geom0.npy`. No additional affine normalization
or rotation is introduced. The direct forward call is the one used in upstream
`training/postprocess.py`: `model(geometry_batch, code=condition_batch)[0]`.
The API path that rotates arbitrary solver meshes by 6.7166 degrees serves a
different coordinate interface; applying it to this dataset would rotate twice.

The result contains normalized `fields`, physical `cp`, `cf_stream`, `cf_span`,
and `coefficients={CL,CD,CM}`. Friction factors are divided out exactly once.
Coefficient integration uses upstream `BasicWing` with `iscentric=True`,
`isnormed=True`, and explicit `normal_factors=(1,150,300)`. This follows the
current implementation despite stale normalization values in some upstream
docstrings. The vertex geometry is never replaced by the cell-center mesh.

Reference-surface and solver-mesh coefficients are distinct. Compare predictions
primarily with the reference-surface labels and report solver values separately.
For pitching moment, the dataset card describes a leading-edge reference while
the pinned `BasicWing.aero_force()` calls `get_moment_2d` with its default
`ref_point=[0.25,0,0]`; some solver scripts also set `xRef=0.25`. The demo preserves
that pinned implementation and reports CM diagnostically. Do not merge those
conventions or advertise solver CM agreement without a reference-point check.
CL/CD are the initial design metrics.

Integrating the eight default released **ground-truth** fields with the pinned
postprocessor gives maximum absolute differences from stored reference labels
of about `8.18e-5` (CL), `6.85e-6` (CD), and `1.25e-3` (CM). The differences persist
in float64; the vertex and cell-center geometry agree to machine precision.
They are a measured integration consistency floor, not neural prediction error.
The source release does not identify the exact historical postprocessor used
to produce every label. Keep this check alongside predictive error, rather than
silently recalibrating labels or claiming bitwise reproduction.

## Runtime, safety, and claims

The model is instantiated from the official JSON architecture and loads the raw
HF tensor state dictionary with `torch.load(..., weights_only=True)` and
`strict=True`. Missing assets, incompatible shapes, or missing dependencies
produce a clear failure; there is no automatic dummy-model fallback.

Prediction timing includes input preparation/device transfer, synchronized
forward execution, output transfer, and CPU coefficient integration. Warm-up
and model loading are reported separately. On CUDA, each boundary synchronizes
the device. Dataset I/O and one-time downloads are outside individual prediction
timing; end-to-end experimental budgets must account for them separately.

Returned provenance includes actual imported source paths, actual Git revisions,
tracked-modification flags, expected source pins, config/weight hashes, device,
and PyTorch version. A non-Git installed package has an unknown Git revision;
it is not reported as verified pinned source.

Surface predictions are **not** a volume-field CFD restart and are **not** an
adjoint implementation. An offline accuracy/inference demo does not establish
CFD speedup, solver convergence, optimization feasibility, or total MDO cost
reduction. Those require matched new CFD runs and a complete optimization study.

Downloaded source trees retain their licenses. Downloaded datasets and derived
subsets stay outside Git; sharing them must retain CRMpert attribution and the
CC-BY-SA-4.0 terms. The project's own license does not relicense these assets.
