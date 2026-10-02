"""Thin CLI exposing the per-source path helpers in common.py to shell scripts.

Installed as the `bench-paths` console script. Each subcommand prints exactly
one value (or a newline-separated list) to stdout. On error, prints to stderr
and exits non-zero — so shell call sites can fail fast.

Usage:
    bench-paths kernel-dir <source>
    bench-paths oracle-dir <source>
    bench-paths kernel <source> <kernel>             # kernel .py path
    bench-paths oracle <source> <kernel>             # oracle .py path (strict)
    bench-paths baseline-category <source>
    bench-paths baseline-profile <source> <kernel>   # cached baseline json path
    bench-paths list-sources                          # one source name per line
    bench-paths list-kernels <source>                 # active (registry-filtered)
    bench-paths list-kernels --all <source>           # include skip_default
"""
from __future__ import annotations

import argparse
import sys

from . import common


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench-paths", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def _src(name): return sub.add_parser(name).add_argument("source")
    def _src_kernel(name):
        sp = sub.add_parser(name)
        sp.add_argument("source")
        sp.add_argument("kernel")

    _src("kernel-dir")
    _src("oracle-dir")
    _src("baseline-category")
    _src_kernel("kernel")
    _src_kernel("oracle")
    _src_kernel("baseline-profile")
    sub.add_parser("list-sources")
    lk = sub.add_parser("list-kernels")
    lk.add_argument("source")
    lk.add_argument("--all", action="store_true",
                    help="include kernels flagged skip_default in kernels.toml")

    args = p.parse_args(argv)

    try:
        if args.cmd == "kernel-dir":
            print(common.kernel_dir_for(args.source))
        elif args.cmd == "oracle-dir":
            print(common.oracle_dir_for(args.source))
        elif args.cmd == "baseline-category":
            print(common.baseline_category_for(args.source))
        elif args.cmd == "kernel":
            print(common.kernel_path_for(args.source, args.kernel))
        elif args.cmd == "oracle":
            print(common.oracle_path_for(args.source, args.kernel))
        elif args.cmd == "baseline-profile":
            print(common.baseline_profile_path_for(args.source, args.kernel))
        elif args.cmd == "list-sources":
            for s in common.SOURCES:
                print(s)
        elif args.cmd == "list-kernels":
            for k in common.kernels_for_source(args.source,
                                               include_skipped=args.all):
                print(k)
        else:  # pragma: no cover — argparse enforces required subcommand
            p.error(f"unknown subcommand {args.cmd!r}")
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"bench-paths: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
