#!/usr/bin/env python3
"""
Populate an isolated workspace directory for an external agent benchmark run.

The workspace mirrors exactly what nki-optimizer provides its LLM:
  - Same NKI docs (nki-info.md, distilled-nki-api-spec.md)
  - Same rendered optimization menu (via nki_optimizer's private render functions)
  - For pre-explored: all nki-optimizations/NN-*/README.md files
  - Task description (CLAUDE.md / AGENTS.md / GEMINI.md) with baseline latency

Usage:
    python setup_agent_workspace.py \\
      --kernel-path <path> \\
      --oracle-path <path> \\
      --menu autocomp|pre-explored|no-info \\
      --output-dir <workspace-dir> \\
      [--baseline-latency-ms 1.234]

Prints the workspace path to stdout.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Resolve repo root relative to this script
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
_RESOURCES = _REPO_ROOT / "src" / "triton_nki_transpile" / "resources"
_NKI_OPT_DIR = _REPO_ROOT / "nki-optimizations"

# Add src to path so we can import the package without installing it each time
sys.path.insert(0, str(_REPO_ROOT / "src"))

from triton_nki_transpile.nki_optimizer import (
    _load_autocomp_items,
    _load_pre_explored_menu,
    _render_autocomp_menu,
    _render_pre_explored_menu,
)

TASK_DESCRIPTION_TEMPLATE = """\
# NKI Kernel Optimization Task

You are an expert NKI (Neuron Kernel Interface) performance engineer working on AWS Trainium hardware.
Your task: minimize the execution latency of `kernel.py`{task_objective_suffix}

## Environment

Ensure you activate the NKI PyTorch venv before any profiling or Python commands that require NKI:
```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
```

This workspace also provides two helper executables so Codex can stay within a tight execution policy:
```bash
./python-in-venv ...
./nki-profile --kernel-path kernel.py --oracle-path oracle.py --bench
```

`python-in-venv` activates the Neuron venv and exposes this repository's `src/` on `PYTHONPATH`.
`nki-profile` does the same and invokes `triton_nki_transpile.profile_cli`.
{serialization_warning}
## Files in This Workspace

- `kernel.py` — the NKI kernel to optimize (contains a `@nki.jit`-decorated kernel + wrapper)
- `oracle.py` — correctness test; defines `run_test(wrapper)` which raises on incorrect output
{opt_menu_file_entry}- `nki-docs/nki-info.md` — NKI architecture and programming model reference
- `nki-docs/distilled-nki-api-spec.md` — full NKI API reference (`nki.isa` and `nki.language`)
- `nki-docs/nki-language-guide.rst` — NKI language guide (if present){readme_section}

## Profiling Tool

Use `nki-profile` to measure hardware latency:

```bash
nki-profile --kernel-path <path-to-kernel.py> --oracle-path oracle.py --bench
```

The `--bench` flag profiles using the `bench()` function in `oracle.py`, which runs the
benchmark-representative input shape. Without it, the profiler uses the small correctness-test
shapes from `run_test()` and may report an unrepresentatively low latency.

Output is JSON printed to stdout. On success it includes `latency_ms` plus all hardware metrics
reported by `neuron-profile` (engine utilization, DMA activity, HBM traffic, MFU/MBU estimates, etc.):

```json
{{
  "success": true,
  "latency_ms": 1.234,
  "error": null,
  "total_active_time_ms": 0.891,
  "total_active_time_percent": 72.3,
  "tensor_engine_active_time_percent": 45.1,
  "vector_engine_active_time_percent": 18.6,
  "scalar_engine_active_time_percent": 0.8,
  "gpsimd_engine_active_time_percent": 0.4,
  "dma_active_time_percent": 33.5,
  "hbm_read_bytes": 262144,
  "hbm_write_bytes": 262144,
  "mfu_estimated_percent": 12.3,
  "mbu_estimated_percent": 5.6,
  "neuroncore_cycle_count": 49687,
  "instance_type": "trn1.2xlarge",
  "... (additional neuron-profile metrics)": "..."
}}
```

Key fields for optimization decisions:
- `success: true` means the oracle test passed AND the NEFF compiled successfully.
- `latency_ms` — steady-state hardware latency in ms (`null` if compilation/test failed).
- `error` — failure description when `success` is false.
- `tensor_engine_active_time_percent` — matrix-multiply engine utilization; low = underutilized tensor ops.
- `vector_engine_active_time_percent` — vector/elementwise engine utilization.
- `dma_active_time_percent` — fraction of time on data movement; high = memory-bandwidth bottleneck.
- `hbm_read_bytes` / `hbm_write_bytes` — HBM traffic; reducing these improves memory-bound kernels.
- `mfu_estimated_percent` — model FLOP utilization.
- `mbu_estimated_percent` — memory bandwidth utilization.
- All metrics are omitted when profiling fails or neuron-profile is unavailable.

Profile `kernel.py` first to establish the baseline and examine the engine utilization breakdown before choosing an optimization strategy.

## Task

1. Profile `kernel.py` to verify the **baseline latency** ({baseline_display}).
2. {step2_text}
3. {step3_text}
4. Implement the optimization — write a new version of the kernel.
5. Profile the new version.
6. Write your final best kernel to `optimized_kernel.py` in this directory.
7. Write a brief report (`report.md`) describing:
   - The optimizations you applied and why.
   - A comparison of the baseline and optimized profiles.

## NKI Wrapper Rules

All required imports at the top of the kernel file:
```python
import nki as nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla
```

Implementation requirements:
- The wrapper function MUST be preceded by the comment `# Wrapper Function:`.
- The wrapper function MUST be the LAST function defined.
- Use `device = torch_xla.device()` to get the Neuron device — NEVER `torch.device("cuda")`.
- Do NOT pre-allocate output tensors with `torch.empty(...)` or `torch.zeros(...)` in the wrapper; NKI kernels allocate their own outputs via `nl.ndarray(buffer=nl.shared_hbm)`.
- Preserve the existing kernel function name and wrapper function name exactly.
- The final output MUST be the complete kernel file (all imports + kernel + wrapper).

## Baseline

Wrapper function: `{wrapper_name}`
Baseline latency: **{baseline_display}**
"""


_NKI_OPT_MODELS_DIR = _REPO_ROOT / "nki-opt-models"
_NKI_OPT_MODELS_V2_DIR = _REPO_ROOT / "nki-opt-models-v2"


def _render_approach(approach_mode: str) -> str | None:
    if approach_mode == "pre-explored":
        menu = _load_pre_explored_menu()
        return _render_pre_explored_menu(menu, menu.items)
    elif approach_mode in ("no-info", "model-guided", "model-guided-v2"):
        return None
    else:
        items = _load_autocomp_items()
        return _render_autocomp_menu(items)


def _copy_model_files(workspace: Path) -> None:
    """Copy nki-opt-models scripts into the workspace for model-guided mode."""
    dest = workspace / "nki-opt-models"
    dest.mkdir(exist_ok=True)
    for filename in ["features.py", "predict_all.py", "features_guide.md"]:
        src = _NKI_OPT_MODELS_DIR / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
    # Copy each model.py so predict_all.py can discover and import them
    for subdir in sorted(_NKI_OPT_MODELS_DIR.iterdir()):
        model_path = subdir / "model.py"
        if subdir.is_dir() and model_path.exists():
            dest_subdir = dest / subdir.name
            dest_subdir.mkdir(exist_ok=True)
            shutil.copy2(model_path, dest_subdir / "model.py")


def _copy_model_files_v2(workspace: Path) -> None:
    """Copy nki-opt-models-v2 scripts into the workspace for model-guided-v2 mode."""
    dest = workspace / "nki-opt-models-v2"
    dest.mkdir(exist_ok=True)
    for filename in [
        "_common.py",
        "predict_all_v2.py",
        "knob_guide.py",
        "KNOB_GUIDE.md",
        "OPT38_REFERENCE.md",
    ]:
        src = _NKI_OPT_MODELS_V2_DIR / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
    # Copy each model.py + fit.json so predict_all_v2.py can load them
    for subdir in sorted(_NKI_OPT_MODELS_V2_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_") or subdir.name == "tools":
            continue
        model_path = subdir / "model.py"
        fit_path = subdir / "fit.json"
        if model_path.exists():
            dest_subdir = dest / subdir.name
            dest_subdir.mkdir(exist_ok=True)
            shutil.copy2(model_path, dest_subdir / "model.py")
            if fit_path.exists():
                shutil.copy2(fit_path, dest_subdir / "fit.json")


def _copy_nki_docs(workspace: Path) -> None:
    docs_dir = workspace / "nki-docs"
    docs_dir.mkdir(exist_ok=True)
    for filename in ["nki-info.md", "distilled-nki-api-spec.md", "nki-language-guide.rst"]:
        src = _RESOURCES / filename
        if src.exists():
            shutil.copy2(src, docs_dir / filename)


def _copy_opt_readmes(workspace: Path) -> None:
    opt_dest = workspace / "nki-optimizations"
    opt_dest.mkdir(exist_ok=True)
    for opt_dir in sorted(_NKI_OPT_DIR.iterdir()):
        if not opt_dir.is_dir():
            continue
        readme = opt_dir / "README.md"
        if readme.exists():
            dest_dir = opt_dest / opt_dir.name
            dest_dir.mkdir(exist_ok=True)
            shutil.copy2(readme, dest_dir / "README.md")


def _write_workspace_helpers(workspace: Path) -> None:
    activate_script = workspace / "activate_venv.sh"
    python_wrapper = workspace / "python-in-venv"
    profile_wrapper = workspace / "nki-profile"

    repo_src = _REPO_ROOT / "src"
    nkilib_src = _REPO_ROOT / "nki-library" / "src"
    venv_activate = Path("/opt/aws_neuronx_venv_pytorch_2_9/bin/activate")

    activate_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
source "{venv_activate}"
"""
    )

    # The optional flock branch serializes Neuron Core access across concurrent
    # python invocations in the same container. Without it, codex can launch
    # `./nki-profile --bench` and `./python-in-venv - <<PY ... oracle.run_test PY`
    # in parallel; both try to allocate kernel[2] via torch_xla and the loser
    # gets NRT `Available:0`. See benchmark-opt/agent_features.toml field
    # `nki_lnc_serialize`.
    python_wrapper.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
source "{venv_activate}"
export PYTHONPATH="{repo_src}:{nkilib_src}:${{PYTHONPATH:-}}"
export NEURON_PLATFORM_TARGET_OVERRIDE="${{NEURON_PLATFORM_TARGET_OVERRIDE:-trn1}}"
if [[ "${{NKI_LNC_SERIALIZE:-0}}" == "1" ]]; then
    exec flock /tmp/nki_lnc.lock python "$@"
fi
exec python "$@"
"""
    )

    profile_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/python-in-venv" -m triton_nki_transpile.profile_cli "$@"
"""
    )

    os.chmod(activate_script, 0o755)
    os.chmod(python_wrapper, 0o755)
    os.chmod(profile_wrapper, 0o755)


def _write_output_placeholders(workspace: Path) -> None:
    placeholders = {
        "optimized_kernel.py": "# Write the final optimized kernel here.\n",
        "plan.md": "# Optimization plan\n",
        "report.md": "# Optimization report\n",
    }
    for filename, content in placeholders.items():
        path = workspace / filename
        if not path.exists():
            path.write_text(content)


def _write_codex_rules(workspace: Path) -> None:
    rules_dir = workspace / ".codex" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    config_path = workspace / ".codex" / "config.toml"
    rules_path = rules_dir / "default.rules"
    config_path.write_text(
        """# Project-scoped Codex config for this benchmark workspace.
# Keep this file present so the adjacent `.codex/rules/` directory is loaded as
# a project configuration layer while auth still comes from ~/.codex.
"""
    )
    rules_path.write_text(
        """# Benchmark-specific Codex rules for the isolated optimization workspace.
# These allow only the small set of external commands needed to inspect files,
# activate the Neuron environment, run Python from this workspace, and invoke the profiler.

prefix_rule(
    pattern = ["pwd"],
    decision = "allow",
    justification = "Allow basic workspace inspection",
    match = ["pwd"],
)

prefix_rule(
    pattern = ["ls"],
    decision = "allow",
    justification = "Allow basic workspace inspection",
    match = ["ls", "ls -la"],
)

prefix_rule(
    pattern = ["find"],
    decision = "allow",
    justification = "Allow basic workspace inspection",
    match = ["find . -maxdepth 2 -type f"],
)

prefix_rule(
    pattern = ["rg"],
    decision = "allow",
    justification = "Allow fast read-only search in the benchmark workspace",
    match = ["rg --files .", "rg -n latency kernel.py"],
)

prefix_rule(
    pattern = ["sed", "-n"],
    decision = "allow",
    justification = "Allow bounded file reads",
    match = ["sed -n 1,120p kernel.py", "sed -n 1,200p opt-menu.md"],
)

prefix_rule(
    pattern = ["cat"],
    decision = "allow",
    justification = "Allow file reads",
    match = ["cat AGENTS.md"],
)

prefix_rule(
    pattern = ["head"],
    decision = "allow",
    justification = "Allow bounded file reads",
    match = ["head -n 40 kernel.py"],
)

prefix_rule(
    pattern = ["tail"],
    decision = "allow",
    justification = "Allow bounded file reads",
    match = ["tail -n 40 kernel.py"],
)

prefix_rule(
    pattern = ["source", "/opt/aws_neuronx_venv_pytorch_2_9/bin/activate"],
    decision = "allow",
    justification = "Allow activation of the Neuron Python environment",
    match = ["source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate"],
)

prefix_rule(
    pattern = ["./activate_venv.sh"],
    decision = "allow",
    justification = "Allow activating the local workspace helper for the Neuron environment",
    match = ["./activate_venv.sh"],
)

prefix_rule(
    pattern = ["./python-in-venv"],
    decision = "allow",
    justification = "Allow Python execution inside the workspace with the Neuron venv and repo src on PYTHONPATH",
    match = [
        "./python-in-venv --version",
        "./python-in-venv -m triton_nki_transpile.profile_cli --help",
    ],
)

prefix_rule(
    pattern = ["python-in-venv"],
    decision = "allow",
    justification = "Allow Python execution inside the workspace with the Neuron venv and repo src on PYTHONPATH",
    match = [
        "python-in-venv --version",
        "python-in-venv -m triton_nki_transpile.profile_cli --help",
    ],
)

prefix_rule(
    pattern = [["./nki-profile", "nki-profile"]],
    decision = "allow",
    justification = "Allow the benchmark workspace profiler wrapper",
    match = [
        "./nki-profile --help",
        "nki-profile --help",
        "./nki-profile --kernel-path kernel.py --oracle-path oracle.py --bench",
        "nki-profile --kernel-path kernel.py --oracle-path oracle.py --bench",
    ],
)

prefix_rule(
    pattern = ["python"],
    decision = "forbidden",
    justification = "Use `./python-in-venv` instead of invoking Python directly outside the sandbox.",
    match = ["python -V"],
    not_match = ["./python-in-venv -V"],
)

prefix_rule(
    pattern = ["python3"],
    decision = "forbidden",
    justification = "Use `./python-in-venv` instead of invoking Python directly outside the sandbox.",
    match = ["python3 -V"],
    not_match = ["./python-in-venv -V"],
)
"""
    )


# Per-variant step3 prompt for model-guided-v2. Adding a new variant means
# adding an entry here + a section in approaches.toml — keeps the diff tight.
_SERIALIZATION_WARNING = """
**Run NKI tools sequentially, not in parallel.** `./nki-profile` and `./python-in-venv` both allocate `/dev/neuron0` via `torch_xla`. Launching two of them as concurrent shell tool calls causes the second to fail with NRT `Logical Neuron Core(s) not available - Requested:2 Available:0`. Always wait for one NKI command to finish before starting another.
"""

# Variants whose prompt includes the no-parallel warning. Listed as
# (approach_mode, variant). The harness's `nki_lnc_serialize` agent_features
# field is independent — these two layers compose: variant adds the prompt
# nudge; agent_features adds the flock-based hard guarantee.
_VARIANTS_WITH_SERIAL_WARNING: frozenset[tuple[str, str]] = frozenset({
    ("no-info",         "v2"),
    ("autocomp",        "v2"),
    ("pre-explored",    "v2"),
    ("model-guided-v2", "v4"),
    ("model-guided-v2", "v5"),
    ("model-guided-v2", "v6"),
})


MODEL_GUIDED_V2_STEP3_BY_VARIANT: dict[str, str] = {
    "v1": (
        "Fill in `knobs.json` from `kernel.py` (only structural knobs — profile metrics come from `profile.json`). "
        "Then run the v2 prediction models:\n"
        "   ```bash\n"
        "   ./python-in-venv nki-opt-models-v2/predict_all_v2.py \\\n"
        "       --profile profile.json --knobs knobs.json --output predictions.json\n"
        "   ```\n"
        "   Read `predictions.json` — it lists optimizations ranked by predicted speedup. Each entry "
        "includes `delta_us` (predicted change in profile metrics), `extrapolation_warning` (if the "
        "knob is outside the calibrated range — treat with caution), and `notes`. "
        "Select the highest-speedup applicable optimization without an extrapolation warning, then plan and implement it."
    ),
    "v2": (
        "Fill in `knobs.json` from `kernel.py` (only structural knobs — profile metrics come from `profile.json`). "
        "Then run the v2 prediction models:\n"
        "   ```bash\n"
        "   ./python-in-venv nki-opt-models-v2/predict_all_v2.py \\\n"
        "       --profile profile.json --knobs knobs.json --output predictions.json\n"
        "   ```\n"
        "   Read `predictions.json` — it lists optimizations ranked by predicted speedup. Each entry "
        "includes `delta_us` (predicted change in profile metrics), `extrapolation_warning` (if the "
        "knob is outside the calibrated range — treat with caution), and `notes`. "
        "These models may predict a speedup where there is none, and may incorrectly mark optimizations as inapplicable. "
        "Treat the results with scrutiny. "
        "Select the optimizations you want to apply based on your analysis of the kernel, the profiling data, and the predictions. "
        "Write an optimization plan."
    ),
    # v4 == v3 step3 body (predictor changes unchanged); the v4 variant only
    # adds the no-parallel warning at the Environment-section level via
    # _VARIANTS_WITH_SERIAL_WARNING. Step3 stays the same so we can compare
    # v3 vs v4 cleanly to measure whether the prompt nudge actually moves
    # the NRT-failure rate.
    "v3": (
        "Fill in `knobs.json` from `kernel.py` AND `profile.json` (some knobs in v3 are profile-derived "
        "— see `nki-opt-models-v2/KNOB_GUIDE.md` for which). Then run the v2 prediction models:\n"
        "   ```bash\n"
        "   ./python-in-venv nki-opt-models-v2/predict_all_v2.py \\\n"
        "       --profile profile.json --knobs knobs.json --output predictions.json\n"
        "   ```\n"
        "   Read `predictions.json` — it lists optimizations ranked by predicted speedup. Each entry "
        "includes `delta_us`, `extrapolation_warning`, and `notes`. Treat predictions with scrutiny: "
        "models may predict false speedups or miss applicable opts.\n"
        "\n"
        "   Selection rule (v3): apply **all** opts where the prediction is `applicable=true`, "
        "`speedup > 1.10x`, and there is no `extrapolation_warning` — not just the single top entry. "
        "If two opts conflict (e.g. opt05 and opt07 on the same fusion site), prefer the higher "
        "predicted speedup. Stack the chosen opts in one implementation plan.\n"
        "\n"
        "   Fallback for unmodeled opts: if no opt in `predictions.json` clears the 1.10x bar, consult "
        "`nki-optimizations/optimization-quick-reference.md` and select the most profile-applicable opt "
        "there (it lists profile-metric triggers — `spill_save_bytes > 0`, "
        "`total_active_time_percent < 80%`, etc. — for opts that may lack a v2 model). Treat predicted "
        "speedup as 'unknown' for fallback picks.\n"
        "\n"
        "   Then write the optimization plan."
    ),
}
# Alias v4 → same step3 body as v3. The only difference is the
# no-parallel warning, which is added by the shared template via
# _VARIANTS_WITH_SERIAL_WARNING. Keeping the step3 bodies identical lets us
# attribute any v3-vs-v4 delta entirely to the prompt nudge.
MODEL_GUIDED_V2_STEP3_BY_VARIANT["v4"] = MODEL_GUIDED_V2_STEP3_BY_VARIANT["v3"]

# v5: addresses the four implementation/process defects identified in
# benchmark-opt/reports/mg_v2_lncfix_shortfalls.md:
#   1. opt38 multi-program output-aggregation defect (deck flagged on lora;
#      post-LNC-fix it bit lora, add_rmsnorm_matmul, and adamw). Fix: direct
#      the agent to the verified strided-shard reference before implementing.
#   2. Profile-schema bug funnelled mg_v2 to opt38-only. Fixed at the
#      predictor level (predict_all_v2.py auto-flattens hardware_metrics);
#      the prompt now notes that the previously-hidden traffic/precision
#      opts are visible.
#   3. opt38 over-predicts on DMA-bound kernels. Predictor now applies a
#      DMA-share guard; the prompt warns the agent to read the predictor's
#      DMA-share note and pair opt38 with traffic-reduction on bandwidth-
#      bound kernels rather than relying on sharding alone.
#   4. Agent only verified the bench() shape. The prompt mandates an oracle
#      run (small shape) plus --bench3 (three shapes) before shipping.
MODEL_GUIDED_V2_STEP3_BY_VARIANT["v5"] = (
    "Fill in `knobs.json` from `kernel.py` AND `profile.json` (some knobs in v3+ are profile-derived "
    "— see `nki-opt-models-v2/KNOB_GUIDE.md` for which). Then run the v2 prediction models:\n"
    "   ```bash\n"
    "   ./python-in-venv nki-opt-models-v2/predict_all_v2.py \\\n"
    "       --profile profile.json --knobs knobs.json --output predictions.json\n"
    "   ```\n"
    "   Read `predictions.json` — it lists optimizations ranked by predicted speedup. Each entry "
    "includes `delta_us`, `extrapolation_warning`, and `notes`. Treat predictions with scrutiny: "
    "models may predict false speedups or miss applicable opts.\n"
    "\n"
    "   Selection rule: apply **all** opts where the prediction is `applicable=true`, "
    "`speedup > 1.10x`, and there is no `extrapolation_warning` — not just the single top entry. "
    "If two opts conflict (e.g. opt05 and opt07 on the same fusion site), prefer the higher "
    "predicted speedup. Stack the chosen opts into one implementation plan.\n"
    "\n"
    "   Fallback for unmodeled opts: if no opt in `predictions.json` clears the 1.10x bar, consult "
    "`nki-optimizations/optimization-quick-reference.md` and select the most profile-applicable opt "
    "there (it lists profile-metric triggers — `spill_save_bytes > 0`, "
    "`total_active_time_percent < 80%`, etc. — for opts that may lack a v2 model). Treat predicted "
    "speedup as 'unknown' for fallback picks.\n"
    "\n"
    "   **If you plan to apply opt38 (SPMD launch grid), read "
    "`nki-opt-models-v2/OPT38_REFERENCE.md` first** and use the strided tile-assignment pattern "
    "(`if i_m % NUM_LNCS != LNC_ID: continue`). The contiguous `ceil`-division pattern is a known "
    "bug — it leaves one program with zero tiles when `n_p_tiles < G`, producing a half-output "
    "defect on small shapes. The strided pattern is correct for any tile count (odd, even, or "
    "smaller than `G`), so you do not need a wrapper-level small-shape fallback or an even-tile "
    "parity guard.\n"
    "\n"
    "   **On bandwidth-bound kernels, do not ship opt38 alone.** If `predictions.json`'s opt38 "
    "`notes` mentions a `DMA-share guard` (or if `profile.json` shows `dma_active_time_percent > "
    "50%`), the two NCs will contend for the same HBM bus and the realized speedup will be well "
    "below `G`. Pair opt38 with a traffic-reduction opt (opt23 larger SBUF tile, opt32 multi-level "
    "tiling, opt33 coalesce stores, opt17 SBUF-instead-of-HBM, opt13 lower precision) — these "
    "cut HBM bytes / packets before the two cores share the bus. Sharding without traffic "
    "reduction is rarely sufficient on a DMA-bound kernel.\n"
    "\n"
    "   Then write the optimization plan. Before shipping, **verify on more than just the "
    "`bench()` shape**:\n"
    "   - Run `./nki-profile --kernel-path optimized_kernel.py --oracle-path oracle.py` (no "
    "`--bench`) to confirm the small oracle correctness shape passes — this catches opt38 "
    "uneven-shard bugs that `--bench` would miss.\n"
    "   - Run `./nki-profile --kernel-path optimized_kernel.py --oracle-path oracle.py --bench3` "
    "and confirm a non-trivial speedup at **all three** shapes. The `bench3` scoreboard is what "
    "scores this run; an even-tile parity guard or shape-specific tiling that only wins the "
    "middle shape will halve your geomean.\n"
    "\n"
    "   If the optimization breaks on the small oracle shape but lands on `bench()`, prefer "
    "**fixing the kernel** (e.g. strided assignment, `nl.minimum` bounds guard, sequence "
    "padding) over reverting to baseline. A wrapper-level small-shape fallback is acceptable "
    "as a last resort but should be a documented escape hatch, not the primary path."
)

# v6: identical to v5 except for a strengthened empty-predictor fallback.
# Motivation (compiler-generated benchmark kernels):
# on the compiler-gen set, mg_v2@v5 regressed to 1.00x on `whisper_create_sinusoids`
# and `t5_relative_position_bucket` — kernels where the predictor (and quick-ref)
# had no applicable opt, so the agent declined to act, while the free-form NoInfo
# approach found a 2.66x / 1.41x rewrite. v6 adds a third fallback tier: when
# neither predictions.json nor the quick-reference yields an applicable opt, the
# agent must attempt a free-form optimization of its own rather than shipping the
# baseline. "No model entry" means unmodeled, not already-optimal.
MODEL_GUIDED_V2_STEP3_BY_VARIANT["v6"] = MODEL_GUIDED_V2_STEP3_BY_VARIANT["v5"].replace(
    "   Fallback for unmodeled opts: if no opt in `predictions.json` clears the 1.10x bar, consult "
    "`nki-optimizations/optimization-quick-reference.md` and select the most profile-applicable opt "
    "there (it lists profile-metric triggers — `spill_save_bytes > 0`, "
    "`total_active_time_percent < 80%`, etc. — for opts that may lack a v2 model). Treat predicted "
    "speedup as 'unknown' for fallback picks.\n",
    "   Fallback ladder when the predictor is empty: if no opt in `predictions.json` clears the "
    "1.10x bar, work down this ladder rather than stopping at the baseline:\n"
    "   1. Consult `nki-optimizations/optimization-quick-reference.md` and select the most "
    "profile-applicable opt there (it lists profile-metric triggers — `spill_save_bytes > 0`, "
    "`total_active_time_percent < 80%`, etc. — for opts that may lack a v2 model). Treat predicted "
    "speedup as 'unknown' for fallback picks.\n"
    "   2. **If neither `predictions.json` nor the quick-reference yields an applicable opt, do NOT "
    "ship the baseline unchanged — attempt a free-form optimization of your own.** An empty predictor "
    "means the kernel is *unmodeled*, NOT that it is already optimal. Study `kernel.py` and "
    "`profile.json` directly and rewrite the kernel to cut latency using whatever transformation the "
    "code and profile motivate — for example: eliminate redundant or recomputed work, precompute "
    "constants / index tables / sinusoids once instead of per-call, fuse elementwise chains, "
    "vectorize a scalar or per-element loop, hoist loop-invariant compute out of the program loop, "
    "narrow precision where the oracle tolerates it, or replace an expensive op sequence with a "
    "cheaper algebraic equivalent. Treat this exactly as you would an open-ended 'make it faster' "
    "task.\n"
    "   3. Only return the kernel unchanged if, after a genuine free-form attempt, your `bench3` "
    "measurements show no rewrite beats the baseline. Reverting to baseline is a measured "
    "last resort, never the default when the predictor is silent.\n",
)


def build_workspace(
    kernel_path: Path,
    oracle_path: Path,
    approach_mode: str,
    output_dir: Path,
    baseline_latency_ms: float | None = None,
    docker_mode: bool = False,
    baseline_profile_json: Path | None = None,
    approach_variant: str = "",
) -> Path:
    if not approach_variant:
        raise ValueError(
            "approach_variant is required (e.g. 'v1', 'v2'); no default exists."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy kernel and oracle
    shutil.copy2(kernel_path, output_dir / "kernel.py")
    shutil.copy2(oracle_path, output_dir / "oracle.py")

    # Surface the baseline profile JSON when supplied (model-guided-v2 needs it)
    if baseline_profile_json is not None and baseline_profile_json.exists():
        shutil.copy2(baseline_profile_json, output_dir / "profile.json")

    # NKI docs
    _copy_nki_docs(output_dir)
    _write_workspace_helpers(output_dir)
    _write_output_placeholders(output_dir)
    if not docker_mode:
        _write_codex_rules(output_dir)

    # Optimization menu (the rendered opt-menu.md, when applicable)
    if approach_mode == "model-guided":
        raise ValueError("The legacy v1 performance models are not included; use model-guided-v2 after calibration.")
    menu_text = _render_approach(approach_mode)
    if menu_text is not None:
        (output_dir / "opt-menu.md").write_text(menu_text)

    # Approach-specific template variables
    if approach_mode == "pre-explored":
        _copy_opt_readmes(output_dir)
        readme_section = "\n- `nki-optimizations/` — detailed README for each menu optimization (before/after code, profiling data)"
        task_objective_suffix = " by applying optimizations from `opt-menu.md`."
        opt_menu_file_entry = "- `opt-menu.md` — optimization menu listing available optimizations with applicability criteria\n"
        step2_text = "Read `opt-menu.md` and `nki-docs/nki-info.md` to understand the optimization options."
        step3_text = "Select the most applicable optimization(s) from the menu and write an optimization plan.\n   For the selected optimization, read its README in `nki-optimizations/` for implementation guidance."
    elif approach_mode == "no-info":
        readme_section = ""
        task_objective_suffix = "."
        opt_menu_file_entry = ""
        step2_text = "Read `nki-docs/nki-info.md` to understand the NKI architecture and programming model."
        step3_text = "Select the optimizations you want to apply based on your analysis of the kernel and the profiling data, and write an optimization plan."
    elif approach_mode == "model-guided":
        _copy_model_files(output_dir)
        # Write empty features.json template
        import subprocess
        import sys as _sys
        try:
            result = subprocess.run(
                [_sys.executable, str(output_dir / "nki-opt-models" / "features.py")],
                capture_output=True, text=True
            )
            (output_dir / "features.json").write_text(result.stdout or "{}\n")
        except Exception:
            (output_dir / "features.json").write_text("{}\n")
        readme_section = (
            "\n- `nki-opt-models/features_guide.md` — how to fill in `features.json`"
            "\n- `nki-opt-models/predict_all.py` — runs all performance models against your features"
            "\n- `features.json` — kernel feature template (fill this in before running predictions)"
        )
        task_objective_suffix = " using performance models to predict the best optimizations."
        opt_menu_file_entry = ""
        step2_text = (
            "Read `kernel.py` and the profiler output carefully. "
            "Then read `nki-opt-models/features_guide.md` to understand what kernel features to extract."
        )
        step3_text = (
            "Fill in `features.json` based on the patterns you observe in `kernel.py`. "
            "Then run the prediction models:\n"
            "   ```bash\n"
            "   ./python-in-venv nki-opt-models/predict_all.py --features features.json --output predictions.json\n"
            "   ```\n"
            "   Read `predictions.json` — it lists optimizations ranked by predicted speedup. "
            "Select the most profitable applicable optimization and write an implementation plan."
        )
    elif approach_mode == "model-guided-v2":
        _copy_model_files_v2(output_dir)
        # Write empty knobs.json template via knob_guide.py
        import subprocess
        import sys as _sys
        try:
            result = subprocess.run(
                [_sys.executable, str(output_dir / "nki-opt-models-v2" / "knob_guide.py")],
                capture_output=True, text=True
            )
            (output_dir / "knobs.json").write_text(result.stdout or "{}\n")
        except Exception:
            (output_dir / "knobs.json").write_text("{}\n")
        readme_section = (
            "\n- `nki-opt-models-v2/KNOB_GUIDE.md` — how to fill in `knobs.json` (per-opt knob schemas)"
            "\n- `nki-opt-models-v2/predict_all_v2.py` — runs all v2 performance models"
            "\n- `nki-opt-models-v2/OPT38_REFERENCE.md` — verified strided-shard pattern for opt38 (read before implementing SPMD)"
            "\n- `knobs.json` — kernel knob template (fill this in before running predictions)"
            "\n- `profile.json` — baseline profile (output of `nki-profile`); used as the prediction anchor"
        )
        task_objective_suffix = " using v2 performance models to predict the best optimizations."
        opt_menu_file_entry = ""
        step2_text = (
            "Read `kernel.py` and `profile.json` (the baseline profile). "
            "Then read `nki-opt-models-v2/KNOB_GUIDE.md` to learn the per-opt knob schemas."
        )
        if approach_variant not in MODEL_GUIDED_V2_STEP3_BY_VARIANT:
            raise ValueError(
                f"Unknown approach_variant {approach_variant!r} for model-guided-v2. "
                f"Known: {sorted(MODEL_GUIDED_V2_STEP3_BY_VARIANT.keys())}."
            )
        step3_text = MODEL_GUIDED_V2_STEP3_BY_VARIANT[approach_variant]
    else:
        readme_section = ""
        task_objective_suffix = " by applying optimizations from `opt-menu.md`."
        opt_menu_file_entry = "- `opt-menu.md` — optimization menu listing available optimizations with applicability criteria\n"
        step2_text = "Read `opt-menu.md` and `nki-docs/nki-info.md` to understand the optimization options."
        step3_text = "Select the most applicable optimization(s) from the menu and write an optimization plan."

    # Detect wrapper name from kernel
    try:
        from triton_nki_transpile.utils import extract_wrapper_name
        wrapper_name = extract_wrapper_name(kernel_path.read_text()) or "unknown"
    except Exception:
        wrapper_name = "unknown"

    baseline_display = f"{baseline_latency_ms:.3f} ms" if baseline_latency_ms is not None else "profile kernel.py to measure"

    serialization_warning = (
        "\n" + _SERIALIZATION_WARNING
        if (approach_mode, approach_variant) in _VARIANTS_WITH_SERIAL_WARNING
        else ""
    )

    task_md = TASK_DESCRIPTION_TEMPLATE.format(
        readme_section=readme_section,
        task_objective_suffix=task_objective_suffix,
        opt_menu_file_entry=opt_menu_file_entry,
        step2_text=step2_text,
        step3_text=step3_text,
        wrapper_name=wrapper_name,
        baseline_display=baseline_display,
        serialization_warning=serialization_warning,
    )

    for fname in ["CLAUDE.md", "AGENTS.md", "GEMINI.md"]:
        (output_dir / fname).write_text(task_md)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up an isolated agent workspace for NKI kernel optimization")
    parser.add_argument("--kernel-path", required=True, type=Path)
    parser.add_argument("--oracle-path", required=True, type=Path)
    parser.add_argument("--approach", required=True,
                        choices=["autocomp", "pre-explored", "no-info", "model-guided", "model-guided-v2"],
                        help="Approach (guidance methodology) to use. Maps to approaches.toml.")
    parser.add_argument("--approach-variant", required=True,
                        help="Variant of the approach (e.g. v1, v2). No default exists. "
                             "For model-guided-v2 supported variants: "
                             f"{sorted(MODEL_GUIDED_V2_STEP3_BY_VARIANT.keys())}.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-latency-ms", type=float, default=None)
    parser.add_argument("--baseline-profile-json", type=Path, default=None,
                        help="Path to baseline nki-profile JSON; surfaced as profile.json for model-guided-v2")
    parser.add_argument("--docker-mode", action="store_true",
                        help="Skip writing .codex/rules/ (Docker provides isolation instead)")
    args = parser.parse_args()

    workspace = build_workspace(
        kernel_path=args.kernel_path,
        oracle_path=args.oracle_path,
        approach_mode=args.approach,
        output_dir=args.output_dir,
        baseline_latency_ms=args.baseline_latency_ms,
        docker_mode=args.docker_mode,
        baseline_profile_json=args.baseline_profile_json,
        approach_variant=args.approach_variant,
    )
    print(workspace)


if __name__ == "__main__":
    main()
