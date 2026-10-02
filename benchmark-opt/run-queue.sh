#!/usr/bin/env bash
# benchmark-opt/run-queue.sh
#
# Drive run-agent-bench-docker.sh from a queue file produced by
# `bench-status --runqueue-out FILE`.
#
# The queue file is plain text: section headers
# `[<name> / <af> / <method> / <model> / <agent> / <reasoning_effort>]` group
# tab-separated `source<TAB>kernel<TAB>approach@variant` lines. Lines starting
# with `#` are skipped. `name` is the matrix.toml cell's unique short id and is
# only used for display here; af/method/model/agent/reasoning are forwarded to
# the runner as --agent-features / --measurement-method / --model / --agents /
# --reasoning-effort.
#
# Each (section, source, kernel) group becomes one or more invocations of
# run-agent-bench-docker.sh. Within a (section, source, kernel) group we
# partition specs into batches where each approach name appears at most once
# (the underlying script can't run two variants of the same approach in one
# invocation — they'd collide on on-disk workspace/log/history filenames).
# In the common case of one variant per approach, partitioning yields a single
# batch per kernel.
#
# Usage:
#   ./run-queue.sh QUEUE_FILE [extra flags...]
#
# Extra flags are forwarded to run-agent-bench-docker.sh, e.g.
#   ./run-queue.sh queue.txt --reasoning-effort high --use-codex-api-key
#
# Options:
#   --dry-run    Print the run-agent-bench-docker.sh invocations without
#                executing them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run-agent-bench-docker.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 QUEUE_FILE [extra flags forwarded to run-agent-bench-docker.sh]" >&2
    exit 1
fi

QUEUE_FILE="$1"; shift

DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -f "$QUEUE_FILE" ]]; then
    echo "Queue file not found: $QUEUE_FILE" >&2
    exit 1
fi

# Bash 4+ required for associative arrays.
if [[ -z "${BASH_VERSINFO+x}" ]] || (( BASH_VERSINFO[0] < 4 )); then
    echo "This script needs bash 4+ (associative arrays). Current: $BASH_VERSION" >&2
    exit 1
fi

# Parse the queue into entries:
#   `name|af|method|model|agent|reasoning|source|kernel|approach@variant`
ENTRIES=()
SECTION_NAME=""; SECTION_AF=""; SECTION_METHOD=""; SECTION_MODEL=""
SECTION_AGENT=""; SECTION_REASONING=""

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"            # strip inline comments
    line="${line#"${line%%[![:space:]]*}"}"  # ltrim
    line="${line%"${line##*[![:space:]]}"}"  # rtrim
    [[ -z "$line" ]] && continue

    # Section header: `[name / af / method / model / agent / reasoning]`. Split
    # on '/' and trim each field. None of these fields contain '/'.
    if [[ "$line" =~ ^\[(.+)\]$ ]]; then
        IFS='/' read -ra _hdr <<< "${BASH_REMATCH[1]}"
        for j in "${!_hdr[@]}"; do
            t="${_hdr[$j]}"
            t="${t#"${t%%[![:space:]]*}"}"; t="${t%"${t##*[![:space:]]}"}"
            _hdr[$j]="$t"
        done
        if (( ${#_hdr[@]} != 6 )); then
            echo "Malformed section header (need [name / af / method / model / agent / reasoning], got ${#_hdr[@]} fields): '$raw_line'" >&2
            exit 1
        fi
        SECTION_NAME="${_hdr[0]}"
        SECTION_AF="${_hdr[1]}"
        SECTION_METHOD="${_hdr[2]}"
        SECTION_MODEL="${_hdr[3]}"
        SECTION_AGENT="${_hdr[4]}"
        SECTION_REASONING="${_hdr[5]}"
        if [[ -z "$SECTION_NAME" || -z "$SECTION_AF" || -z "$SECTION_METHOD" \
              || -z "$SECTION_MODEL" || -z "$SECTION_AGENT" \
              || -z "$SECTION_REASONING" ]]; then
            echo "Malformed section header (empty field): '$raw_line'" >&2
            exit 1
        fi
        continue
    fi

    if [[ -z "$SECTION_NAME" ]]; then
        echo "Queue line before any section header: '$raw_line'" >&2
        exit 1
    fi

    # source<TAB>kernel<TAB>approach@variant (also tolerate spaces)
    read -r src kernel spec rest <<< "$line"
    if [[ -z "$src" || -z "$kernel" || -z "$spec" ]]; then
        echo "Malformed queue line: '$raw_line'" >&2
        exit 1
    fi
    if [[ "$spec" != *"@"* ]]; then
        echo "Queue entry missing @variant: '$raw_line' (every approach must be pinned, e.g. 'model_guided_v2@v2')" >&2
        exit 1
    fi
    ENTRIES+=("$SECTION_NAME|$SECTION_AF|$SECTION_METHOD|$SECTION_MODEL|$SECTION_AGENT|$SECTION_REASONING|$src|$kernel|$spec")
done < "$QUEUE_FILE"

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
    echo "Queue contains no active entries."
    exit 0
fi

# Group by (name|af|method|model|agent|reasoning|src|kernel) → list of approach@variant specs.
declare -A SPECS_BY_KEY
declare -a KEY_ORDER=()    # preserve first-seen order for stable output

for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r name af method model agent reasoning src kernel spec <<< "$entry"
    key="$name|$af|$method|$model|$agent|$reasoning|$src|$kernel"
    if [[ -z "${SPECS_BY_KEY[$key]:-}" ]]; then
        SPECS_BY_KEY[$key]="$spec"
        KEY_ORDER+=("$key")
    else
        SPECS_BY_KEY[$key]="${SPECS_BY_KEY[$key]} $spec"
    fi
done

# Partition a group's spec list into batches where each approach name appears
# at most once. Greedy: each spec goes into the lowest-indexed batch where its
# approach name isn't already taken. Sets global BATCHES (array of CSV strings)
# as the result. Worst case: # batches = max count of any single approach.
_partition_batches() {
    local specs_str="$1"
    local -a specs
    read -ra specs <<< "$specs_str"

    BATCHES=()
    local -a batch_approaches=()  # parallel to BATCHES: space-separated approach names already used in that batch
    for spec in "${specs[@]}"; do
        local name="${spec%@*}"
        local placed=0
        local i
        for ((i = 0; i < ${#BATCHES[@]}; i++)); do
            case " ${batch_approaches[$i]} " in
                *" $name "*) ;;
                *)
                    BATCHES[$i]="${BATCHES[$i]},$spec"
                    batch_approaches[$i]="${batch_approaches[$i]} $name"
                    placed=1
                    break
                    ;;
            esac
        done
        if [[ $placed -eq 0 ]]; then
            BATCHES+=("$spec")
            batch_approaches+=(" $name ")
        fi
    done
}

# Count total invocations up front (= sum of batch counts across groups).
TOTAL_INVOCATIONS=0
for key in "${KEY_ORDER[@]}"; do
    _partition_batches "${SPECS_BY_KEY[$key]}"
    TOTAL_INVOCATIONS=$((TOTAL_INVOCATIONS + ${#BATCHES[@]}))
done

echo "Queue: ${#ENTRIES[@]} entries → ${#KEY_ORDER[@]} (kernel × cohort) groups → $TOTAL_INVOCATIONS invocation(s)"
echo ""

i=0
for key in "${KEY_ORDER[@]}"; do
    IFS='|' read -r name af method model agent reasoning src kernel <<< "$key"
    _partition_batches "${SPECS_BY_KEY[$key]}"

    local_total=${#BATCHES[@]}
    for ((b = 0; b < local_total; b++)); do
        i=$((i + 1))
        approaches="${BATCHES[$b]}"

        cmd=(
            "$RUNNER"
            --source "$src"
            --kernels "$kernel"
            --agent-features "$af"
            --measurement-method "$method"
            --model "$model"
            --approaches "$approaches"
            --agents "$agent"
            --reasoning-effort "$reasoning"
        )
        if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
            cmd+=("${EXTRA_ARGS[@]}")
        fi

        if (( local_total > 1 )); then
            tag="$name :: $src/$kernel (batch $((b + 1))/$local_total)"
        else
            tag="$name :: $src/$kernel"
        fi
        echo "[$i/$TOTAL_INVOCATIONS] $tag  approaches=$approaches  af=$af  method=$method  model=$model  agent=$agent  reasoning=$reasoning"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            printf '  %q ' "${cmd[@]}"; echo
        else
            "${cmd[@]}"
        fi
    done
done

echo ""
echo "Queue complete."
