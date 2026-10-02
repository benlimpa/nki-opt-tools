#!/usr/bin/env python3
"""
benchmark-opt/memory-watchdog.py

Monitors total system memory usage and kills the top memory-consuming
processes when usage exceeds a configured threshold. Intended to guard
against runaway profiler subprocesses that occasionally leak and freeze
the host.

Strategy:
  - Sample /proc/meminfo every --interval seconds.
  - When (MemTotal - MemAvailable) >= --threshold-gb, list user processes
    sorted by RSS, and SIGTERM (then SIGKILL after --grace-seconds) the
    top processes whose combined RSS is at least --kill-fraction of the
    over-threshold excess (or up to --max-kills processes, whichever
    comes first).
  - Never kills PID 1, the watchdog itself, the parent shell, sshd,
    systemd, dockerd, or anything in --exclude-pids / --exclude-names.

Usage:
    python3 benchmark-opt/memory-watchdog.py \
        --threshold-gb 22 --interval 5 --log /tmp/memwatch.log

Run it in the background alongside long profiling jobs:
    nohup python3 benchmark-opt/memory-watchdog.py \
        --threshold-gb 22 > /tmp/memwatch.log 2>&1 &
"""
import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Allowlist: only processes whose cmdline contains one of these substrings
# are eligible to be killed. This is intentionally narrow so the watchdog
# can never take out claude, sshd, dockerd, the parent shell, etc.
DEFAULT_KILL_PATTERNS = [
    "/tmp/nki_opt_profile_",   # the profiler tempdir for each iteration
    "profile_kernel.py",        # the inner profiler script
    "neuron-profile",           # Neuron profiling CLI
    "neuronx-cc",               # Neuron compiler (can leak during profile)
    "neff",                     # NEFF post-processing
]

# Hard exclusions — never touch these even if a pattern matches accidentally.
HARD_EXCLUDE_NAMES = {
    "systemd", "init", "sshd", "dockerd", "containerd", "snapd",
    "claude", "memory-watchdog.py", "tmux", "screen",
    "bash", "fish", "zsh", "node", "amazon-ssm-agent", "multipathd",
}


def _read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if not parts:
                continue
            info[key] = int(parts[0]) * 1024  # kB → bytes
    return info


def _used_bytes():
    info = _read_meminfo()
    total = info["MemTotal"]
    avail = info.get("MemAvailable", info["MemFree"] + info.get("Cached", 0))
    return total - avail, total


def _list_processes():
    """Return list of (pid, rss_bytes, comm, cmdline)."""
    procs = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/status") as f:
                status = f.read()
            rss_kb = 0
            comm = ""
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("Name:"):
                    comm = line.split(maxsplit=1)[1].strip()
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().decode(errors="replace").replace("\x00", " ").strip()
            except OSError:
                cmdline = comm
            procs.append((pid, rss_kb * 1024, comm, cmdline))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return procs


def _is_killable(pid, comm, cmdline, kill_patterns, exclude_pids,
                 exclude_names, self_ancestors):
    """A process is killable iff:
       - it is not the watchdog itself / an ancestor / pid 1,
       - it is not in the hard-exclude list,
       - its cmdline matches at least one allowlist pattern.
    """
    if pid in exclude_pids or pid in self_ancestors or pid == 1:
        return False
    for name in exclude_names:
        if name and (name == comm or name in cmdline):
            return False
    for pat in kill_patterns:
        if pat and pat in cmdline:
            return True
    return False


def _ancestors(pid):
    chain = set()
    cur = pid
    while cur and cur != 0 and cur != 1:
        chain.add(cur)
        try:
            with open(f"/proc/{cur}/status") as f:
                ppid = 0
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
            cur = ppid
        except (FileNotFoundError, ProcessLookupError):
            break
    return chain


def _kill(pid, sig):
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def _log(msg, log_path=None):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def watch(args):
    threshold = int(args.threshold_gb * 1024**3)
    self_ancestors = _ancestors(os.getpid())
    exclude_pids = set(args.exclude_pids or [])
    exclude_names = set(HARD_EXCLUDE_NAMES) | set(args.exclude_names or [])
    kill_patterns = list(DEFAULT_KILL_PATTERNS) + list(args.kill_patterns or [])
    log_path = args.log

    _log(
        f"watchdog started: threshold={args.threshold_gb:.1f} GB, "
        f"interval={args.interval}s, max_kills={args.max_kills}, "
        f"grace={args.grace_seconds}s",
        log_path,
    )
    _log(f"  kill patterns: {kill_patterns}", log_path)
    _log(f"  hard excludes: {sorted(exclude_names)}", log_path)

    while True:
        used, total = _used_bytes()
        used_gb = used / 1024**3
        total_gb = total / 1024**3

        if used >= threshold:
            _log(
                f"OVER THRESHOLD: {used_gb:.2f} GB / {total_gb:.2f} GB used "
                f"(threshold {args.threshold_gb} GB) — scanning processes",
                log_path,
            )
            procs = _list_processes()
            procs.sort(key=lambda p: p[1], reverse=True)

            killable = [
                p for p in procs
                if _is_killable(
                    p[0], p[2], p[3], kill_patterns,
                    exclude_pids, exclude_names, self_ancestors,
                )
                and p[1] > 100 * 1024 * 1024  # at least 100 MB
            ]

            if not killable:
                _log(
                    "  no killable profile processes found above 100 MB — "
                    "leaving non-allowlisted processes alone",
                    log_path,
                )
            else:
                excess = used - threshold
                cumulative = 0
                victims = []
                for pid, rss, comm, cmdline in killable:
                    if len(victims) >= args.max_kills:
                        break
                    victims.append((pid, rss, comm, cmdline))
                    cumulative += rss
                    if cumulative >= max(excess, args.min_freed_gb * 1024**3):
                        break

                for pid, rss, comm, cmdline in victims:
                    _log(
                        f"  KILL pid={pid} rss={rss/1024**3:.2f}GB "
                        f"comm={comm!r} cmd={cmdline[:120]!r}",
                        log_path,
                    )
                    _kill(pid, signal.SIGTERM)

                if args.grace_seconds > 0:
                    time.sleep(args.grace_seconds)
                    for pid, rss, comm, _ in victims:
                        if Path(f"/proc/{pid}").exists():
                            _log(f"  SIGKILL pid={pid} (still alive after SIGTERM)", log_path)
                            _kill(pid, signal.SIGKILL)

            time.sleep(max(args.interval, 5))
        else:
            time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold-gb", type=float, default=22.0,
                        help="Trigger when (MemTotal - MemAvailable) >= this many GB (default: 22)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Polling interval in seconds (default: 5)")
    parser.add_argument("--max-kills", type=int, default=2,
                        help="Maximum processes to kill per trigger (default: 2)")
    parser.add_argument("--min-freed-gb", type=float, default=4.0,
                        help="Keep selecting victims until at least this much RSS is queued for kill (default: 4)")
    parser.add_argument("--grace-seconds", type=float, default=8.0,
                        help="Seconds to wait between SIGTERM and SIGKILL (default: 8)")
    parser.add_argument("--exclude-pids", type=int, nargs="*", default=None,
                        help="Additional PIDs to never kill")
    parser.add_argument("--exclude-names", nargs="*", default=None,
                        help="Additional process name/cmdline substrings to never kill")
    parser.add_argument("--kill-patterns", nargs="*", default=None,
                        help="Additional cmdline substrings to add to the kill allowlist "
                             "(in addition to the built-in profiler patterns)")
    parser.add_argument("--log", type=str, default=None,
                        help="Append events to this file in addition to stdout")
    args = parser.parse_args()

    try:
        watch(args)
    except KeyboardInterrupt:
        _log("watchdog stopped (KeyboardInterrupt)", args.log)
        sys.exit(0)


if __name__ == "__main__":
    main()
