#!/usr/bin/env python3
"""Compare original vs. optimized NKI kernels for all (kernel, mode) pairs in a results dir.

Sends both kernel files to the Claude Code CLI and produces a markdown document
describing structural differences, optimizations applied, and performance
trade-offs.

Usage (CLI):
    python compare_kernels.py --results-dir path/to/results/merged [--force]

Importable:
    from compare_kernels import compare_kernels
    md = compare_kernels(run_dir, mode)   # returns markdown str, caches to .md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from triton_nki_transpile.bench_runs import common

ALL_MODES = ["no_info", "autocomp", "pre_explored", "model_guided"]

_COMPARISON_PROMPT = """\
You are comparing two NKI (Neuron Kernel Interface) kernel implementations \
written for AWS Trainium hardware.  The optimized version was produced by an \
AI agent running in **{mode}** mode.

## Measured Result
{speedup_info}

## Original Kernel (`{kernel_name}.py`)
```python
{original_code}
```

## Optimized Kernel (`{mode}_optimized.py`)
```python
{optimized_code}
```

Write a concise markdown comparison document (use `##` headings) covering:
1. **Key structural changes** — what was added, removed, or reorganised
2. **Specific optimizations applied** — highlight the most impactful code \
changes with short inline snippets where helpful
3. **Performance rationale** — why these changes would (or would not) reduce \
latency on Trainium hardware (e.g. reduced HBM traffic, better engine \
utilization, eliminated redundant ops)
4. **Trade-offs or caveats** — e.g. the optimization is specialized to a \
particular input shape, or correctness was narrowed

Keep the document focused and scannable — prefer bullet points over long prose.
"""


# ---------------------------------------------------------------------------
# Claude Code CLI invocation
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, timeout: int = 180) -> str:
    """Invoke the Claude Code CLI non-interactively and return the text result."""
    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--no-session-persistence",
        "-p", prompt,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "The 'claude' CLI was not found. "
            "Ensure Claude Code is installed and available in PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude CLI timed out after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited with code {proc.returncode}:\n{proc.stderr[:600]}"
        )

    if not proc.stdout.strip():
        raise RuntimeError(
            f"claude produced no stdout.\nstderr: {proc.stderr.strip()[:400] or '(empty)'}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.stdout.strip()

    if data.get("is_error"):
        raise RuntimeError(f"claude returned an error: {data.get('result', data)}")

    result = data.get("result", "").strip()
    if not result:
        raise RuntimeError(
            f"claude returned empty result.\nFull output: {proc.stdout[:600]}"
        )
    return result


# ---------------------------------------------------------------------------
# Original kernel lookup
# ---------------------------------------------------------------------------

def find_original_kernel(run_dir: Path, kernel_name: str,
                         source: str | None = None) -> Path | None:
    """Return the path to the original kernel source, or None if not found.

    Pass `source` whenever available (it's recorded in the run's metadata.json)
    so we resolve deterministically via SOURCE_PATHS[source].kernel_dir. The
    `source=None` path searches every known source in order and is vulnerable
    to cross-source name collisions (matmul exists in reference + nkibench;
    cumsum in three sources) — only retained for callers that genuinely lack
    the source.
    """
    if source is not None:
        try:
            p = common.kernel_path_for(source, kernel_name)
        except KeyError:
            return None
        return p if p.exists() else None
    # Legacy fallback — first-match across all sources.
    for src in common.SOURCES:
        p = common.kernel_path_for(src, kernel_name)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Speedup lookup
# ---------------------------------------------------------------------------

def _get_speedup(run_dir: Path, mode: str, kernel_name: str) -> dict[str, Any]:
    results_file = run_dir.parent / "benchmark_results.json"
    if not results_file.exists():
        return {}
    with open(results_file) as f:
        results = json.load(f)
    for entry in results:
        if entry.get("kernel") == kernel_name:
            return {
                "baseline_ms": entry.get("baseline_ms"),
                "optimized_ms": entry.get(f"{mode}_ms"),
                "speedup": entry.get(f"{mode}_speedup"),
            }
    return {}


# ---------------------------------------------------------------------------
# Core function (importable)
# ---------------------------------------------------------------------------

def compare_kernels(
    run_dir: str | Path,
    mode: str,
    force: bool = False,
) -> str:
    """Compare original vs. optimized kernel for one (kernel_dir, mode) pair.

    Returns a markdown string and caches it to ``{mode}_kernel_comparison.md``
    inside *run_dir*.  Skips the LLM call if the cache file already exists and
    *force* is False.
    """
    run_dir = Path(run_dir).resolve()

    # Detect new layout (one run dir per (kernel, menu)) vs legacy (kernel
    # dir holds many menus). New layout uses fixed names; legacy uses
    # `{mode}_*` prefixes.
    source: str | None = None
    if (run_dir / "metadata.json").exists() and (run_dir / "optimized.py").exists():
        output_path = run_dir / "kernel_comparison.md"
        with open(run_dir / "metadata.json") as f:
            md_data = json.load(f)
        kernel_name = md_data.get("kernel") or run_dir.name
        source = md_data.get("source")  # used to disambiguate the original
        optimized_path = run_dir / "optimized.py"
    else:
        output_path = run_dir / f"{mode}_kernel_comparison.md"
        kernel_name = run_dir.name
        optimized_path = run_dir / f"{mode}_optimized.py"

    if output_path.exists() and not force:
        return output_path.read_text()

    if not optimized_path.exists():
        raise FileNotFoundError(f"Optimized kernel not found: {optimized_path}")
    optimized_code = optimized_path.read_text()

    original_path = find_original_kernel(run_dir, kernel_name, source=source)
    if original_path is None:
        searched = (f"SOURCE_PATHS[{source!r}].kernel_dir" if source
                    else ", ".join(common.SOURCES))
        raise FileNotFoundError(
            f"Could not find original kernel '{kernel_name}.py'. Searched: {searched}"
        )
    original_code = original_path.read_text()

    speedup_info = _get_speedup(run_dir, mode, kernel_name)

    if speedup_info.get("speedup") is not None:
        speedup_text = (
            f"**{speedup_info['speedup']:.3f}x speedup** "
            f"(baseline {speedup_info['baseline_ms']} ms → "
            f"optimized {speedup_info['optimized_ms']} ms)"
        )
    else:
        speedup_text = "Speedup data not available."

    prompt = _COMPARISON_PROMPT.format(
        mode=mode,
        kernel_name=kernel_name,
        speedup_info=speedup_text,
        original_code=original_code,
        optimized_code=optimized_code,
    )

    body = _call_claude(prompt)

    # Build a header with provenance metadata
    try:
        rel_original = original_path.relative_to(original_path.parents[2])
    except ValueError:
        rel_original = original_path

    header = (
        f"# Kernel Comparison: `{kernel_name}` — mode `{mode}`\n\n"
        f"**Original source:** `{rel_original}`  \n"
        f"**Optimized kernel:** `{run_dir.name}/{mode}_optimized.py`\n\n"
    )

    markdown = header + body
    output_path.write_text(markdown)
    print(f"  wrote {output_path.relative_to(output_path.parents[3])}", file=sys.stderr)

    return markdown


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def discover_pairs(results_dir: Path) -> list[tuple[Path, str]]:
    """Return all (kernel_dir, mode) pairs found under *results_dir*."""
    pairs = []
    for kernel_dir in sorted(results_dir.iterdir()):
        if not kernel_dir.is_dir():
            continue
        for mode in ALL_MODES:
            if (kernel_dir / f"{mode}_run.jsonl").exists():
                pairs.append((kernel_dir, mode))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the results directory (e.g. results/merged or results/20260422_202622)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if cached comparison files already exist",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        sys.exit(f"Error: {results_dir} is not a directory")

    pairs = discover_pairs(results_dir)
    if not pairs:
        sys.exit(f"No (kernel, mode) pairs found under {results_dir}")

    print(f"Found {len(pairs)} pairs to compare", file=sys.stderr)
    errors = []

    for i, (kernel_dir, mode) in enumerate(pairs, 1):
        label = f"{kernel_dir.name}/{mode}"
        print(f"[{i}/{len(pairs)}] {label} ...", file=sys.stderr)
        try:
            compare_kernels(kernel_dir, mode, force=args.force)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            errors.append((label, str(exc)))

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for label, msg in errors:
            print(f"  {label}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
