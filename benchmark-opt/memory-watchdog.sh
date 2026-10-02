#!/usr/bin/env bash
# memory-watchdog.sh — kill runaway NKI profiling processes before they OOM the host.
#
# Adapted from benchmark-opt/NKIBenchB2/sentinel.sh, which has worked reliably
# in the past. Key differences from the original Python watchdog:
#   - Uses pgrep -f / ps tree expansion (sees children that the Python version
#     missed because it walked /proc by VmRSS only and never followed PPid down).
#   - Kills the entire process tree (and process group) with SIGKILL, not just
#     individual leaked PIDs.
#   - Two independent triggers: low MemAvailable AND per-process max age.
#
# Usage:
#   ./memory-watchdog.sh &                  # defaults: 4 GB floor, 8-min cap
#   MEM_FLOOR_MB=4500 MAX_AGE_SEC=900 ./memory-watchdog.sh &
#
#   kill %1                                 # to stop it
#
# Logs persistently to benchmark-opt/logs/memwatch-<TS>.log; symlinks
# memwatch-latest.log to the current run.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"
RUN_TS=$(date '+%Y%m%d-%H%M%S')
LOG="$LOGS_DIR/memwatch-${RUN_TS}.log"
ln -sf "memwatch-${RUN_TS}.log" "$LOGS_DIR/memwatch-latest.log" 2>/dev/null || true

# Tunables — override via env.
MEM_FLOOR_MB="${MEM_FLOOR_MB:-4000}"     # kill if MemAvailable drops below 4 GB
MAX_AGE_SEC="${MAX_AGE_SEC:-900}"        # kill any matched process older than 15 min
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-30}"

# Allowlist patterns — only processes whose cmdline matches at least one of
# these are eligible to be killed. Hard-coded so a typo can never take out
# claude / sshd / dockerd / the parent shell.
PATTERNS=(
    "nki_opt_profile"
    "neuron-profile"
    "profile_kernel.py"
    "profile_cli"
    "triton_nki_transpile.profile"
    "neuronx-cc"
)

# Hard-deny: even if a pattern accidentally matches one of these, do not kill.
HARD_DENY=(
    "claude" "sshd" "dockerd" "containerd" "systemd" "tmux" "screen"
    "memory-watchdog" "amazon-ssm-agent" "snapd"
)

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

log "=== memory-watchdog started ==="
log "  LOG=$LOG"
log "  MEM_FLOOR=${MEM_FLOOR_MB}MB  MAX_AGE=${MAX_AGE_SEC}s  CHECK_INTERVAL=${CHECK_INTERVAL}s"
log "  patterns: ${PATTERNS[*]}"
log "  hard-deny: ${HARD_DENY[*]}"

avail_mb() { awk '/^MemAvailable:/ { printf "%d", $2/1024 }' /proc/meminfo; }
total_mb() { awk '/^MemTotal:/     { printf "%d", $2/1024 }' /proc/meminfo; }

full_cmdline() {
    local pid=$1
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | sed 's/ $//' || echo "(cmdline unavailable)"
}

is_hard_deny() {
    local cmd=$1 deny
    for deny in "${HARD_DENY[@]}"; do
        [[ "$cmd" == *"$deny"* ]] && return 0
    done
    return 1
}

# All PIDs whose cmdline matches any PATTERNS entry, with hard-deny applied.
pids_matching() {
    local pat pid cmd
    local -a pids=()
    for pat in "${PATTERNS[@]}"; do
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            cmd=$(full_cmdline "$pid")
            is_hard_deny "$cmd" && continue
            pids+=("$pid")
        done < <(pgrep -f "$pat" 2>/dev/null || true)
    done
    printf '%s\n' "${pids[@]+"${pids[@]}"}" | sort -u
}

# BFS of a PID and all descendants.
all_descendants() {
    local -a queue=("$@")
    local -a result=()
    local pid child
    while [[ ${#queue[@]} -gt 0 ]]; do
        pid="${queue[0]}"
        queue=("${queue[@]:1}")
        [[ -z "$pid" ]] && continue
        result+=("$pid")
        for child in $(ps --ppid "$pid" -o pid= 2>/dev/null | tr -d ' ' || true); do
            [[ -n "$child" ]] && queue+=("$child")
        done
    done
    printf '%s\n' "${result[@]+"${result[@]}"}" | sort -u
}

process_age_sec() {
    local elapsed
    elapsed=$(ps -o etimes= -p "$1" 2>/dev/null | tr -d ' ') || true
    printf '%s' "${elapsed:-0}"
}

kill_pids() {
    local reason=$1; shift
    local -a roots=("$@")
    [[ ${#roots[@]} -eq 0 ]] && return

    local -a all_pids=()
    mapfile -t all_pids < <(all_descendants "${roots[@]}")

    log "KILLING [reason: $reason] — ${#all_pids[@]} process(es) (roots: ${roots[*]})"

    local pid cmd age ppid pgid
    for pid in "${all_pids[@]}"; do
        cmd=$(full_cmdline "$pid")
        if is_hard_deny "$cmd"; then
            log "  SKIP (hard-deny) pid=$pid cmd=$cmd"
            continue
        fi
        age=$(process_age_sec "$pid")
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || ppid="?"
        pgid=$(ps -o pgid=  -p "$pid" 2>/dev/null | tr -d ' ') || pgid="?"
        log "  KILL pid=$pid ppid=$ppid pgid=$pgid age=${age}s"
        log "       cmd=$cmd"
        kill -9 "$pid" 2>/dev/null || true
        [[ "$pgid" =~ ^[0-9]+$ ]] && kill -9 "-${pgid}" 2>/dev/null || true
    done

    # Final broad sweep so any process matching a pattern is caught even if
    # we missed it during the tree walk.
    local pat
    for pat in "${PATTERNS[@]}"; do
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            cmd=$(full_cmdline "$pid")
            is_hard_deny "$cmd" && continue
            log "  sweep KILL pid=$pid pat=$pat cmd=$cmd"
            kill -9 "$pid" 2>/dev/null || true
        done < <(pgrep -f "$pat" 2>/dev/null || true)
    done

    log "  sweep complete  mem_after=$(avail_mb)MB"
    sleep 3
}

trap 'log "=== memory-watchdog exiting ==="; exit 0' INT TERM

last_heartbeat=0

while true; do
    sleep "$CHECK_INTERVAL"

    now=$(date '+%s')
    avail=$(avail_mb)
    total=$(total_mb)
    mapfile -t pids < <(pids_matching)

    if (( now - last_heartbeat >= HEARTBEAT_INTERVAL )); then
        log "heartbeat  mem=${avail}/${total}MB  tracked_pids=${#pids[@]}"
        last_heartbeat=$now
    fi

    # (a) Memory floor.
    if [[ "$avail" -lt "$MEM_FLOOR_MB" ]] && [[ ${#pids[@]} -gt 0 ]]; then
        log "ALERT: mem=${avail}/${total}MB < floor ${MEM_FLOOR_MB}MB"
        kill_pids "low-memory (${avail}MB)" "${pids[@]}"
        continue
    fi

    # (b) Per-process age cap.
    for pid in "${pids[@]}"; do
        age=$(process_age_sec "$pid")
        if [[ "$age" -gt "$MAX_AGE_SEC" ]]; then
            log "ALERT: pid=$pid age=${age}s > max ${MAX_AGE_SEC}s"
            log "       cmd=$(full_cmdline "$pid")"
            kill_pids "timeout (${age}s)" "${pids[@]}"
            break
        fi
    done
done
