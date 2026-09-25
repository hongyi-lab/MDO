# Actual AeroTransformer adaptation

This path updates the real released AeroTransformer state_dict. It is separate
from the synthetic interface test and from any later residual calibration.
It does not introduce a new foundation model or implement neuron selection.

Two fixed-budget baselines are available:

- `head`: freeze the entire backbone and train only the existing
  `final_layer.out_proj` convolution (weights and bias). In the pinned upstream
  architecture this is the final 3×3 projection before unpatchifying the surface.
  Frozen modules stay in evaluation mode, and their weights/buffers are checked
  for exact equality after training. This is a projection-probing baseline.
- `full`: enable all original model parameters and fine-tune the complete model.

The loss is float32 mean squared error on the upstream channels
`[Cp, 150*Cf_stream, 300*Cf_span]`. It is not a drag loss or a gradient-fidelity
guarantee. Both modes use Adam, gradient clipping at norm 1, fixed epochs, no
early stopping and no validation-driven checkpoint selection. A low training
loss is not evidence of held-out accuracy.

## Split contract

The split file uses source sample IDs, not compact subset row numbers:

```json
{
  "schema_version": 1,
  "train_sample_ids": [100, 101],
  "calibration_sample_ids": [200],
  "test_sample_ids": [300]
}
```

These illustrative IDs must be replaced with IDs present in your actual
dataset. The three sets must be disjoint by both sample and geometry. Optional
`train`, `calibration`, `test` arrays declare geometry IDs and are checked against
the sample mapping. An existing `fingerprint` from the split generator is
preserved alongside independently computed normalized-split and file hashes.

Split validation reads metadata only. Training loads field/geometry arrays only
for the declared training cases. Calibration and test cases never influence
fitting, epoch count, checkpoint selection or preprocessing statistics. Decide
hyperparameters before reading test performance. If comparing hyperparameters,
introduce a separate development split rather than reusing the test set.

An explicit list can be provided with `--train-sample-ids`. This fits those IDs
only and makes no claim that other data constitute a held-out test split.

## Run on the A6000

After running the model bootstrap and fetching data/weights:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/train_adapter.py \
  --data assets/CRMpert --checkpoint assets/AeroTransformer/ATsurf_S \
  --split results/experiment/split.json --output results/experiment/adapted_head \
  --device cuda:0 --mode head --epochs 10 --batch-size 1 --lr 1e-4 --seed 7
```

Use `--device cpu` for a small local integration check. The explicit 10-epoch
budget is a starter setting, not a selected best result. For a full-fine-tuning
comparison use a new output directory, `--mode full`, and a budget chosen before
test inspection. Batch size 1 limits activation memory; larger settings need
measurement. CUDA is never silently replaced with CPU when unavailable.

The Python entry point is:

```python
from mdo_demo.adaptation import train_model
manifest = train_model(
    data_dir, base_checkpoint, split_manifest, output,
    device="cuda:0", epochs=10, mode="head", batch_size=1, lr=1e-4, seed=7,
)
```

The output directory must be new or empty. A failed or interrupted directory
should be kept as failure evidence; select a new directory for another run.
Only a completed `manifest.json` and `training_status.json` indicate readiness.

## Artifacts and evaluation

Successful output contains:

- `best_model_weights`: raw tensor state_dict, strictly compatible with the
  original architecture. The filename follows upstream convention; it is the
  final fixed epoch, not a checkpoint selected by test/validation performance.
- `model_config`: byte-for-byte copy of the original architecture config.
- `manifest.json`: base and exported hashes, source revisions, train sample and
  geometry IDs, split and consumed-training-data fingerprints, hyperparameters,
  trainable parameter names/counts, training-loss history, device and timing.
- `training_status.json`: completion or failure marker.

Use the output directory wherever a checkpoint directory is accepted. Run a
separate evaluation restricted to the held-out geometry IDs; do not present an
evaluation over all downloaded samples as a held-out score. The saved training
loss history is the average of losses encountered during each training epoch,
not a final post-training error estimate.

Timing is synchronized for CUDA. `fit_wall_s` covers minibatch assembly,
transfers, forward/backward passes, updates and finite-value checks.
`total_wall_s` also includes reading training data, loading the base model and
exporting the checkpoint; it excludes original CFD label generation. Preserve
both when comparing total cost. Neither metric alone demonstrates inference
acceleration or CFD-verified design quality.

Architecture sources inspected: [AeroTransformer wrapper](https://github.com/YangYunjia/floGen/blob/ff3abda23e10e1073c07ffd78dad96979e940c77/flowvae/app/wing/models.py),
[PDE decoder](https://github.com/YangYunjia/floGen/blob/ff3abda23e10e1073c07ffd78dad96979e940c77/flowvae/base_model/pdet/pde_transformer.py),
[final projection](https://github.com/YangYunjia/floGen/blob/ff3abda23e10e1073c07ffd78dad96979e940c77/flowvae/base_model/pdet/final_layer.py).
