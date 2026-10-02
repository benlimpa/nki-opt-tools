# NKI Optimization Models — v2 Workflow

Reusable process for porting an optimization to v2: build a caller-anchored
performance model calibrated against multiple observations.

## How v2 differs from v1

| | v1 (`nki-opt-models/`) | v2 (`nki-opt-models-v2/`) |
|---|---|---|
| Predict input | knob values only | `KernelProfile` + typed `Knobs` |
| Anchor for `t_before` | hardcoded calibration constants | caller's measured profile |
| Calibration | 1 point per model | ≥2 paired points per knob |
| Constants | hand-derived in source | fit by `fit.py`, stored in `fit.json` |
| Confidence | none | `extrapolation_warning` string |

The two are kept side-by-side so head-to-head experiments via
`benchmark-opt/run-agent-bench-docker.sh` remain possible.

## Layout

```
nki-opt-models-v2/
├── _common.py               KernelProfile, PredictionDelta, load_fit, extrapolation
├── predict_all_v2.py        Entry point: profile.json + knobs.json → predictions.json
├── knob_guide.py            Generates KNOB_GUIDE.md + knobs.json templates
├── KNOB_GUIDE.md            Auto-generated knob extraction guide for the agent
├── WORKFLOW_V2.md           This file
├── tools/
│   └── run_calibration_sweep.py    Profiles a sweep, appends to calibration.csv
└── optNN-name/
    ├── model.py             predict(profile, knobs) → PredictionDelta
    ├── fit.py               Fits calibration.csv → fit.json
    ├── calibration.csv      Multi-point observations
    ├── fit.json             Generated; do not hand-edit
    ├── calibration_plan.yaml  Sweep definition for tools/run_calibration_sweep.py
    ├── sweep_kernels/       Parameterized kernel variants referenced by the plan
    └── README.md            How to extend the calibration
```

## Step 1 — Define the model

Create `optNN-name/model.py`:

- **Module docstring** — pattern, knobs, formula. Cite the
  `nki-optimizations/NN-*/` source.
- **Module-level constants** —
  ```python
  OPTIMIZATION_NUMBER = NN
  OPTIMIZATION_NAME = "Short Name"
  REQUIRED_PROFILE_FIELDS = ("t_total_us", ...)
  KNOB_NAMES: dict[str, str] = {"...": "agent-facing description"}
  ```
- **`Knobs` dataclass** — typed structural fields the agent extracts.
- **`predict(profile, knobs) -> PredictionDelta | None`** — returns `None` if
  applicability fails or required profile fields are `None`. Never falls back
  to hardcoded baselines.
- **`if __name__ == "__main__"` self-test** — load `calibration.csv`, predict
  on each `before` row, print observed vs predicted speedup.

The model must take its baseline anchor from `profile`, not from
`fit.json`. Coefficients in `fit.json` describe *how the bottleneck shifts*
under the optimization, not the absolute baseline latency.

## Step 2 — Seed the calibration

If a single-point observation already exists (typically from
`nki-optimizations/NN-*/{before,after}_profile/summary.txt`), seed
`calibration.csv` with the row schema below. This is enough to ship `fit.py`
and validate end-to-end before extending the sweep.

```
kernel_id,knob_1,knob_2,...,variant,t_total_us,t_active_us,t_tensor_us,t_vector_us,t_scalar_us,t_gpsimd_us,t_dma_us,t_static_dma_us,hbm_read_bytes,hbm_write_bytes,dma_packet_count,static_dma_packet_count,vector_instr_count,tensor_instr_count,matmul_instr_count
```

Two rows per `kernel_id`: one `before` and one `after`.

## Step 3 — Write `fit.py`

`fit.py` reads `calibration.csv`, fits the per-model coefficients, and writes
`fit.json`. The shape of `fit.json` is fixed:

```json
{
  "constants": {"alpha_dma": 1.0, "alpha_overhead": 0.85},
  "knob_range": {"n_tiles": [4, 32]},
  "max_pct_err": 4.1,
  "n_points": 6
}
```

Most models have 1–2 free parameters and use a small least-squares solve.
opt16's `fit.py` is a self-contained 2×2 normal-equations solve — copy it as a
template for one or two coefficients without pulling in numpy.

When only one paired point exists, **pin one coefficient by physical reasoning**
(e.g. `alpha_dma=1.0` — the pipeline does not change DMA active time) and
solve for the rest. Print a `[info] single-point fit: pinned ...` message so
it's obvious that the model is operating below its design coverage.

## Step 4 — Plan the sweep

Open-ended bar: **at least 2 paired observations per knob**. A 1-knob model
needs ≥2 pairs; a 3-knob model needs ≥6. Beyond that, more is better.

Recommended sweep grid:
1. Vary the primary knob over 2–4 distinct values.
2. Add 1–2 off-axis points varying a secondary kernel parameter (TILE_M,
   dtype, M-dim) to detect when the model breaks.

Encode the grid as `calibration_plan.yaml`:

```yaml
- kernel_id: <stable id>
  knobs: {<knob_1>: <value>, <knob_2>: <value>, ...}
  before: sweep_kernels/<id>_before.py
  after:  sweep_kernels/<id>_after.py
  oracle: sweep_kernels/<shared>_oracle.py
```

Drop the kernel variants and oracle into `sweep_kernels/`. Use the existing
`nki-optimizations/NN-*/{before,after}.py` as templates. The oracle can be
shared across variants of the same kernel family.

## Step 5 — Run the sweep

On a Neuron-attached host:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd nki-opt-models-v2/optNN-name
python ../tools/run_calibration_sweep.py --plan calibration_plan.yaml
python fit.py
PYTHONPATH=.. python model.py    # self-test
```

`run_calibration_sweep.py` invokes `nki-profile` per variant, appends rows to
`calibration.csv`, and is idempotent (re-running overwrites existing rows for
the same `(kernel_id, variant)` pair). Run with `--dry-run` first to confirm
the plan resolves.

## Step 6 — Validate

The model's `__main__` self-test should report `err% < max_pct_err` for every
calibration row. If error blows up on off-axis points, the cost formula has
missing terms — revisit Step 1.

End-to-end smoke test:

```bash
python predict_all_v2.py --profile sample_profile.json --knobs sample_knobs.json
```

Should print a `ranked` list with the new opt entry — no exceptions, sensible
speedup, and `extrapolation_warning: null` for in-range knobs.

## Step 7 — Update `KNOB_GUIDE.md`

```bash
python knob_guide.py --write-guide
```

The guide is auto-generated from each model's `KNOB_NAMES`; commit the regen.

## Step 8 — Use it

Run the agent benchmark with the `model_guided_v2` menu only:

```bash
./benchmark-opt/run-agent-bench-docker.sh --menus model_guided_v2
```

In each kernel workspace the agent receives `profile.json`, `KNOB_GUIDE.md`,
and a `knobs.json` template. It fills in the knobs, runs `predict_all_v2.py`,
and uses the ranked predictions to choose an optimization.

## Porting priority

The agent benefits most from these first — port them in order:

1. opt35, opt34, opt31, opt37, opt18 (largest observed speedups)
2. opt03, opt01, opt07, opt24, opt11
3. The remaining 12 models as time permits

Each port is independent. v2 ships when ≥5 models are ported; the rest can land
incrementally.
