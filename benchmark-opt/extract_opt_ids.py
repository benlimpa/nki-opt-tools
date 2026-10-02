#!/usr/bin/env python3
"""Stage 1 of opt-ID extraction: regex candidates.

For each `runs/<id>/` directory, scan `workspace/report.md` (and the agent's
selection chatter) for any opt-ID citation, and write the raw matches to
`runs/<id>/opt_ids.json` as `candidates_regex`.

This stage deliberately makes NO judgement about which of those opts were
actually *implemented* — the agent often discusses several and ships one.
That decision is made by `classify_opt_ids.py` (LLM, subscription-billed).

Output schema:

  {
    "run_id": "...",
    "mode": "model_guided_v2",
    "kernel": "cumsum",
    "source": "nl_to_nki",
    "candidates_regex": [38, 31],         # raw opt-IDs mentioned in report.md
    "regex_evidence": ["opt38", "#31"],    # surface forms for audit
    "implemented_opt_ids": null,           # filled by classify_opt_ids.py
    "rationale": null,                     # filled by classify_opt_ids.py
    "method": null,                        # filled by classify_opt_ids.py
    "warnings": []
  }

The `#NN` pattern requires NO whitespace after `#` so it doesn't catch
markdown headings (`# 1. Intro`).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"

OPT_PATTERN = re.compile(r"opt[\s_-]*0*(\d{1,3})", re.IGNORECASE)
HASH_PATTERN = re.compile(r"#0*(\d{1,3})\b")

CATALOG_NUMBERS = {1, 3, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21,
                   24, 29, 30, 31, 32, 33, 34, 35, 37, 38}


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


def extract_candidates(report_text: str) -> tuple[list[int], list[str]]:
    """Return (opt_ids, surface_forms) for every catalog opt mentioned in report.md."""
    body = re.sub(r"```.*?```", "", report_text, flags=re.DOTALL)

    pairs: list[tuple[int, str]] = []
    for m in OPT_PATTERN.finditer(body):
        pairs.append((int(m.group(1)), m.group(0)))
    for m in HASH_PATTERN.finditer(body):
        pairs.append((int(m.group(1)), m.group(0)))

    seen_n: set[int] = set()
    nums: list[int] = []
    for n, _ in pairs:
        if n in CATALOG_NUMBERS and n not in seen_n:
            seen_n.add(n)
            nums.append(n)

    seen_e: set[str] = set()
    evidence: list[str] = []
    for n, s in pairs:
        if n in CATALOG_NUMBERS and s not in seen_e:
            seen_e.add(s)
            evidence.append(s)

    return nums, evidence


def process_run(run_dir: Path, *, force: bool) -> dict | None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())

    out_path = run_dir / "opt_ids.json"
    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        # If implemented_opt_ids is filled (classification done), preserve it.
        if existing.get("implemented_opt_ids") is not None:
            return existing

    report = _read(run_dir / "workspace" / "report.md")
    warnings: list[str] = []
    if report is None:
        candidates: list[int] = []
        evidence: list[str] = []
        warnings.append("no workspace/report.md")
    else:
        candidates, evidence = extract_candidates(report)

    record = {
        "run_id": metadata["run_id"],
        "mode": metadata.get("menu"),
        "kernel": metadata.get("kernel"),
        "source": metadata.get("source"),
        "candidates_regex": candidates,
        "regex_evidence": evidence,
        "implemented_opt_ids": None,
        "rationale": None,
        "method": None,
        "warnings": warnings,
    }
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-features", default="codex_v2_opt38")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run", help="single run_id")
    args = ap.parse_args()

    if args.run:
        run_dirs = [RUNS_DIR / args.run]
    else:
        run_dirs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir())

    written = 0
    no_report = 0
    no_candidates = 0
    for run_dir in run_dirs:
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        metadata = json.loads(meta_path.read_text())
        if metadata.get("agent_features") != args.agent_features:
            continue
        if metadata.get("model") != args.model:
            continue
        if metadata.get("cheat"):
            continue
        rec = process_run(run_dir, force=args.force)
        if rec is None:
            continue
        written += 1
        if "no workspace/report.md" in rec["warnings"]:
            no_report += 1
        elif not rec["candidates_regex"]:
            no_candidates += 1

    print(f"runs processed:        {written}")
    print(f"  with regex hits:     {written - no_report - no_candidates}")
    print(f"  no regex hits:       {no_candidates}")
    print(f"  no report.md:        {no_report}")


if __name__ == "__main__":
    main()
