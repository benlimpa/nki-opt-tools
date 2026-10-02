"""Convert one timestamped results dir from run-agent-bench-docker.sh into runs/.

Used as a post-run step by the harness script. Each (kernel, menu) cell in the
results dir becomes one run in benchmark-opt/runs/<run_id>/.

CLI:
    bench-ingest \\
        --results-dir benchmark-opt/agent_bench_docker_version_0_1_18_codex_gpt_5_4_reference/results/20260507_215815 \\
        --agent codex --model gpt-5.4 --source reference \\
        --agent-features codex_v2_opt38 \\
        --measurement-method bench3_geomean \\
        --package-version 0.1.18

This complements migrate.py (which handles legacy bulk migration). Idempotent.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import common
from .common import RUNS_DIR, REPO_ROOT, LEGACY_CHEAT_BLACKLIST, LEGACY_CHEAT_REASON
from .migrate import find_kernel_menu_files, apply_one, _ts_dir_to_iso


def _next_replicate(source: str, kernel: str, approach: str, approach_variant: str,
                    model: str, agent_features: str) -> int:
    """Find the next free replicate index for a cell by scanning runs/."""
    from .common import RunMetadata
    seen = 0
    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            md_path = d / "metadata.json"
            if not md_path.exists():
                continue
            try:
                md = RunMetadata.from_path(d)
            except Exception:
                continue
            if (md.source == source and md.kernel == kernel
                    and md.approach == approach
                    and md.approach_variant == approach_variant
                    and md.model == model and md.agent_features == agent_features):
                seen = max(seen, md.replicate)
    return seen + 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, type=Path,
                   help="Path to results/<timestamp>/ produced by the harness.")
    p.add_argument("--agent", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True, help="kernel-source label (reference, nl_to_nki, ...)")
    p.add_argument("--agent-features", required=True)
    p.add_argument("--measurement-method", required=True,
                   help="The method the harness measured under. Used only as a default tag for the new measurement record.")
    p.add_argument("--package-version", default=None)
    p.add_argument("--docker-image", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--no-git", action="store_true",
                   help="Use shutil.move instead of git mv (default for fresh, untracked output dirs).")
    p.add_argument("--approach-variants", required=True,
                   help="Comma-separated <approach>=<variant> pairs pinning the variant per approach "
                        "(e.g. 'no_info=v1,autocomp=v1,pre_explored=v1,model_guided_v2=v2'). "
                        "Every approach discovered in --results-dir must appear here — there is "
                        "no default variant.")
    args = p.parse_args(argv)

    # Parse --approach-variants into a dict.
    variant_overrides: dict[str, str] = {}
    for pair in args.approach_variants.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"--approach-variants: malformed entry {pair!r}", file=sys.stderr)
            return 1
        k, v = pair.split("=", 1)
        variant_overrides[k.strip()] = v.strip()

    common.require_agent_features(args.agent_features)
    common.require_measurement_method(args.measurement_method)

    if not args.results_dir.exists():
        print(f"results-dir does not exist: {args.results_dir}", file=sys.stderr)
        return 1

    ts_iso = _ts_dir_to_iso(args.results_dir.name)
    cells = find_kernel_menu_files(args.results_dir)
    if not cells:
        print(f"No (kernel, approach) cells found in {args.results_dir}.")
        return 0

    approaches_reg = common.load_approaches()

    print(f"Ingesting {len(cells)} cells from {args.results_dir.relative_to(REPO_ROOT)}")
    n_ok = n_err = 0
    for kernel, approach, _files in cells:
        if approach not in approaches_reg:
            print(f"  ! skipping unknown approach {approach!r} for kernel {kernel}", file=sys.stderr)
            n_err += 1
            continue
        if approach not in variant_overrides:
            print(
                f"  ! approach {approach!r} (kernel {kernel}) has no variant in "
                f"--approach-variants. Every approach must be pinned; no default exists.",
                file=sys.stderr,
            )
            n_err += 1
            continue
        approach_variant = variant_overrides[approach]
        approaches_reg[approach].require_variant(approach_variant)

        rep = _next_replicate(args.source, kernel, approach, approach_variant,
                              args.model, args.agent_features)
        run_id = common.make_run_id(
            timestamp=ts_iso, source=args.source, kernel=kernel,
            approach=approach, agent=args.agent, model=args.model,
            approach_variant=approach_variant,
        )
        # If a run with that exact id already exists, suffix with -repN to disambiguate.
        if (RUNS_DIR / run_id).exists():
            run_id = f"{run_id}__rep{rep}"

        cheat = (args.source, kernel, approach) in LEGACY_CHEAT_BLACKLIST
        row = {
            "legacy_dir":       args.results_dir.parent.parent.name,  # the agent_bench_docker_*/ basename
            "timestamp_dir":    args.results_dir.name,
            "kernel":           kernel,
            "approach":         approach,
            "approach_variant": approach_variant,
            "agent":            args.agent,
            "model":            args.model,
            "source":           args.source,
            "version":          args.package_version,
            "agent_features":   args.agent_features,
            "replicate":        rep,
            "run_id":           run_id,
            "cheat":            cheat,
            "cheat_reason":     LEGACY_CHEAT_REASON if cheat else None,
        }
        try:
            apply_one(row, use_git_mv=not args.no_git, verbose=False)
            n_ok += 1
            print(f"  + {run_id}  (replicate {rep})")
        except Exception as e:
            n_err += 1
            print(f"  ! {run_id}: {e}")

    print(f"\nDone. {n_ok} ingested, {n_err} errors.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
