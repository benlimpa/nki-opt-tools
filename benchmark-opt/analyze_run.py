#!/usr/bin/env python3
"""Analyze NKI optimization agent runs for all (kernel, mode) pairs in a results directory.

Deterministic metrics are extracted directly from the JSONL event log and
history JSON.  Qualitative facts (optimizations chosen, errors, success
explanation) are extracted by invoking the Claude Code CLI with the agent's
plan, report, and message summaries.

Usage (CLI):
    python analyze_run.py --results-dir path/to/results/merged [--force]

Importable:
    from analyze_run import analyze_run
    result = analyze_run(run_dir, mode)   # returns dict, caches to JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ALL_MODES = ["no_info", "autocomp", "pre_explored", "model_guided"]

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "optimizations_selected": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concise description of each optimization the agent chose to implement",
        },
        "errors_encountered": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concise description of each distinct error or failure the agent ran into",
        },
        "all_errors_resolved": {
            "type": "boolean",
            "description": "Whether all errors were resolved before the run completed",
        },
        "final_optimization_successful": {
            "type": "boolean",
            "description": "Whether the agent produced a kernel that is faster than the baseline",
        },
        "success_explanation": {
            "type": "string",
            "description": "One sentence: did it improve performance, by how much, and why?",
        },
        "approach_summary": {
            "type": "string",
            "description": "1-2 sentences describing the agent's overall strategy",
        },
    },
    "required": [
        "optimizations_selected",
        "errors_encountered",
        "all_errors_resolved",
        "final_optimization_successful",
        "success_explanation",
        "approach_summary",
    ],
}

_ANALYSIS_PROMPT = """\
You are analyzing the output of an AI agent that was trying to optimize an \
NKI (Neuron Kernel Interface) kernel for AWS Trainium hardware.

## Optimization Mode
{mode}

## Agent's Optimization Plan
{plan}

## Agent's Results Report
{report}

## Agent's Messages During the Run
{messages}

## Failed Commands (if any)
{failed_commands}

## Measured Speedup
{speedup_info}

Extract the requested information using the provided JSON schema.
"""


# ---------------------------------------------------------------------------
# Claude Code CLI invocation
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, json_schema: dict | None = None, timeout: int = 180) -> str:
    """Invoke the Claude Code CLI non-interactively and return the text result."""
    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--no-session-persistence",
        "-p", prompt,
    ]
    if json_schema is not None:
        cmd += ["--json-schema", json.dumps(json_schema)]

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

    # --json-schema puts the validated object in "structured_output"; "result" is empty.
    if json_schema is not None and "structured_output" in data:
        return json.dumps(data["structured_output"])

    result = data.get("result", "").strip()
    if not result:
        raise RuntimeError(
            f"claude returned empty result.\nFull output: {proc.stdout[:600]}"
        )
    return result


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def _parse_jsonl(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def extract_jsonl_metrics(jsonl_path: Path) -> dict[str, Any]:
    events = _parse_jsonl(jsonl_path)

    n_turns = sum(1 for e in events if e.get("type") == "turn.started")

    completed_items = [
        e["item"]
        for e in events
        if e.get("type") == "item.completed" and "item" in e
    ]

    commands = [i for i in completed_items if i.get("type") == "command_execution"]
    failed_commands = [
        (i.get("command", "")[:120], i.get("exit_code"))
        for i in commands
        if i.get("exit_code") not in (None, 0)
    ]

    file_changes = [i for i in completed_items if i.get("type") == "file_change"]

    agent_messages = [
        i["text"]
        for i in completed_items
        if i.get("type") == "agent_message" and i.get("text")
    ]

    total_input_tokens = sum(
        e.get("usage", {}).get("input_tokens", 0)
        for e in events
        if e.get("type") == "turn.completed"
    )
    total_output_tokens = sum(
        e.get("usage", {}).get("output_tokens", 0)
        +
        e.get("usage", {}).get("reasoning_output_tokens", 0)
        for e in events
        if e.get("type") == "turn.completed"
    )

    return {
        "n_turns": n_turns,
        "n_commands": len(commands),
        "n_file_changes": len(file_changes),
        "n_failed_commands": len(failed_commands),
        "failed_commands": failed_commands[:10],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "_agent_messages": agent_messages,
    }


# ---------------------------------------------------------------------------
# Speedup lookup
# ---------------------------------------------------------------------------

def _get_speedup(run_dir: Path, mode: str, kernel_name: str) -> dict[str, Any]:
    # New layout: each run is its own directory containing metadata.json with a
    # measurements list. Prefer the latest measurement (last entry).
    metadata_file = run_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            md = json.load(f)
        ms = md.get("measurements") or []
        if ms:
            latest = ms[-1]
            return {
                "baseline_ms":  latest.get("baseline_ms"),
                "optimized_ms": latest.get("optimized_ms"),
                "speedup":      latest.get("speedup"),
            }

    # Legacy layout: parent dir has a benchmark_results.json that aggregates
    # all kernels at the timestamp level.
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
# LLM analysis
# ---------------------------------------------------------------------------

def llm_analyze(
    mode: str,
    plan: str,
    report: str,
    agent_messages: list[str],
    failed_commands: list[tuple[str, Any]],
    speedup_info: dict[str, Any],
) -> dict[str, Any]:
    messages_text = "\n\n".join(
        f"[{i+1}] {msg}" for i, msg in enumerate(agent_messages)
    ) or "(none)"

    failed_text = (
        "\n".join(f"- exit {code}: {cmd}" for cmd, code in failed_commands)
        if failed_commands
        else "(none)"
    )

    if speedup_info.get("speedup") is not None:
        speedup_text = (
            f"{speedup_info['speedup']:.3f}x  "
            f"(baseline {speedup_info['baseline_ms']} ms → "
            f"optimized {speedup_info['optimized_ms']} ms)"
        )
    else:
        speedup_text = "not available"

    prompt = _ANALYSIS_PROMPT.format(
        mode=mode,
        plan=plan or "(no plan.md found)",
        report=report or "(no report.md found)",
        messages=messages_text,
        failed_commands=failed_text,
        speedup_info=speedup_text,
    )

    raw = _call_claude(prompt, json_schema=_ANALYSIS_SCHEMA)

    # The result may be a JSON string (when --json-schema is honoured) or
    # already the parsed text.  Handle both.
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
            )
        return json.loads(text)
    return raw  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Core function (importable)
# ---------------------------------------------------------------------------

def analyze_run(
    run_dir: str | Path,
    mode: str,
    force: bool = False,
) -> dict[str, Any]:
    """Analyze one (kernel_dir, mode) pair.

    Returns the combined metrics + LLM analysis dict, and caches the result
    to ``{mode}_analysis.json`` inside *run_dir*.  Skips the LLM call if the
    cache file already exists and *force* is False.
    """
    run_dir = Path(run_dir).resolve()

    # Locate per-mode files. New layout (one run dir per (kernel, menu)) stores
    # them at fixed names; legacy layout (kernel dir contains many menus) used
    # `{mode}_*` prefixes. Prefer new-layout names if present.
    new_layout = (run_dir / "run.jsonl").exists() and (run_dir / "metadata.json").exists()
    if new_layout:
        output_path  = run_dir / "analysis.json"
        jsonl_path   = run_dir / "run.jsonl"
        history_path = run_dir / "history.json"
        workspace    = run_dir / "workspace"
        with open(run_dir / "metadata.json") as f:
            kernel_name = json.load(f).get("kernel") or run_dir.name
    else:
        output_path  = run_dir / f"{mode}_analysis.json"
        jsonl_path   = run_dir / f"{mode}_run.jsonl"
        history_path = run_dir / f"{mode}_optimized.history.json"
        workspace    = run_dir / f"{mode}_workspace"
        kernel_name  = run_dir.name

    if output_path.exists() and not force:
        with open(output_path) as f:
            return json.load(f)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL log not found: {jsonl_path}")

    metrics = extract_jsonl_metrics(jsonl_path)
    agent_messages = metrics.pop("_agent_messages")

    n_opt_iterations = 0
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        n_opt_iterations = max(0, len(history) - 1)

    speedup_info = _get_speedup(run_dir, mode, kernel_name)

    plan = (workspace / "plan.md").read_text() if (workspace / "plan.md").exists() else ""
    report = (workspace / "report.md").read_text() if (workspace / "report.md").exists() else ""

    llm_result = llm_analyze(
        mode=mode,
        plan=plan,
        report=report,
        agent_messages=agent_messages,
        failed_commands=metrics["failed_commands"],
        speedup_info=speedup_info,
    )

    result = {
        "kernel": kernel_name,
        "mode": mode,
        "metrics": {
            **metrics,
            "n_optimization_iterations": n_opt_iterations,
        },
        "speedup": speedup_info,
        "analysis": llm_result,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {output_path.relative_to(output_path.parents[3])}", file=sys.stderr)

    return result


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
        help="Re-run even if cached analysis files already exist",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        sys.exit(f"Error: {results_dir} is not a directory")

    pairs = discover_pairs(results_dir)
    if not pairs:
        sys.exit(f"No (kernel, mode) pairs found under {results_dir}")

    print(f"Found {len(pairs)} pairs to analyze", file=sys.stderr)
    all_results = []
    errors = []

    for i, (kernel_dir, mode) in enumerate(pairs, 1):
        label = f"{kernel_dir.name}/{mode}"
        print(f"[{i}/{len(pairs)}] {label} ...", file=sys.stderr)
        try:
            result = analyze_run(kernel_dir, mode, force=args.force)
            all_results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            errors.append((label, str(exc)))

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for label, msg in errors:
            print(f"  {label}: {msg}", file=sys.stderr)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    _main()
