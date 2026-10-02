#!/usr/bin/env bash
# benchmark-opt/run-agent-bench-docker.sh
#
# Each agent runs inside a Docker container for
# true OS-level isolation. The host venv and source tree are bind-mounted
# (read-only); only the per-run workspace is writable from inside the container.
#
# Prerequisites:
#   1. Run ./benchmark-opt/test-docker-sandbox.sh to verify all layers work.
#   2. For codex: the sandbox image must include node/npm/codex (see Step 5 of
#      test-docker-sandbox.sh for the one-time build command).
#
# Usage:
#   ./benchmark-opt/run-agent-bench-docker.sh [OPTIONS]
#
# Options:
#   --source SLUG             Pick a known kernel-set by name: one of
#                             reference, nl_to_nki, nki_library, nkibench.
#                             Sets KERNEL_DIR, ORACLE_DIR, and SUFFIX_SLUG.
#                             Individual --kernel-dir / --oracle-dir override.
#   --kernel-dir DIR          Directory containing NKI kernel .py files
#                             (default: benchmark-opt/nl-to-nki)
#   --oracle-dir DIR          Directory containing oracle test logic files
#                             (default: src/.../kernels/tests/logic)
#   --output-dir DIR          Results output directory (default: auto-generated)
#   --kernels LIST            Comma-separated kernel names (default: see KERNELS)
#   --agents LIST             Comma-separated agents: claude,codex (default: claude)
#   --model MODEL             Model name passed to the agent CLI
#   --reasoning-effort LVL    Codex reasoning effort: low, medium, high, xhigh
#   --sandbox-image IMAGE     Docker image to use for codex (default: nki-bench-sandbox:latest)
#   --approaches LIST         Comma-separated approaches to run. Each entry
#                             MUST be `name@variant` — there is no default
#                             variant. Known approaches: autocomp, pre_explored,
#                             no_info, model_guided, model_guided_v2; known
#                             variants are listed in approaches.toml. Hyphens
#                             in names are accepted. Duplicate (name, variant)
#                             pairs are rejected.
#                             Default: see APPROACHES in CONFIGURATION.
#   --use-codex-api-key       Authenticate codex via CODEX_API_KEY env var instead of auth files
#   --no-codex-api-key        Authenticate codex via ~/.codex/auth.json (default)
#   --agent-features TAG      Tag from agent_features.toml to record on the new runs.
#                             Auto-derived from (agent, package_version) if omitted.
#   --measurement-method TAG  Measurement-method tag for the post-run profile data
#                             (default: bench3_geomean — the harness profiler).
#   --no-ingest               Skip the post-run ingest into benchmark-opt/runs/.
#   -h, --help                Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9"
NEURON_VENV_ACTIVATE="$NEURON_VENV/bin/activate"
NKILIB_SRC="$REPO_ROOT/nki-library/src"

if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
fi

if [[ ! -f "$NEURON_VENV_ACTIVATE" ]]; then
    echo "[ERROR] Required Neuron venv not found: $NEURON_VENV_ACTIVATE" >&2
    exit 1
fi

# ============================================================
# CONFIGURATION
# ============================================================

### NL To NKI kernels
# KERNELS=(
    # "cumsum"
    # "layernorm"
    # "rms"
    # "sparsemax"
# )
# KERNEL_DIR="$REPO_ROOT/benchmark-opt/nl-to-nki"
# ORACLE_DIR="$REPO_ROOT/src/triton_nki_transpile/kernels/tests/logic"
# SUFFIX_SLUG="nl_to_nki"

### Reference kernels (partially derived from NKI library / Partially hand-written)
# KERNELS=(
    # "cascaded_max"
    # "crossentropy"
    # "matmul"
    # "rope"
    # "dynamic_elementwise_add"
# )
# KERNEL_DIR="$REPO_ROOT/benchmark-opt/reference"
# ORACLE_DIR="$REPO_ROOT/src/triton_nki_transpile/kernels/tests/logic"
# SUFFIX_SLUG="reference"

### NKI Library Kernels
# KERNELS=(
    # "cumsum"
    # "dynamic_elementwise_add"
#     "rope"
# )
# KERNEL_DIR="$REPO_ROOT/benchmark-opt/nki-library/kernels"
# ORACLE_DIR="$REPO_ROOT/benchmark-opt/nki-library/oracles"
# SUFFIX_SLUG="nki_library"

### NKIBench Kernels (Translated to NKI Beta 2)
# KERNELS=(
    # "adamw"
    # "add_rmsnorm_matmul"
    # "bmm"
    # "gqa_full"   # profile timed out
    # "lora"
    # "mamba"
    # "matmul"
    # "matmul_add_rmsnorm"
    # "rmsnorm_matmul"

    # "rope_single_freq_apply"
    # "silu"
    # "swiglu"
    # "transpose_matmul"
# )
# KERNEL_DIR="$REPO_ROOT/benchmark-opt/NKIBenchB2/kernels"
# ORACLE_DIR="$REPO_ROOT/benchmark-opt/NKIBenchB2/oracles"
# SUFFIX_SLUG="nkibench"



RUN_PREFIX="agent_bench_docker_"

AGENT_NAMES=(
    # "claude"
    "codex"
)
USE_CODEX_API_KEY=0


# MODEL_NAME="claude-sonnet-4-6"
# MODEL_NAME="claude-haiku-4-5"
MODEL_NAME="gpt-5.4"
REASONING_EFFORT="medium"

# Approaches to run, comma-separated. Each entry MUST be "name@variant" — no
# default exists. Override with --approaches.
APPROACHES="no_info@v1,autocomp@v1,pre_explored@v1,model_guided_v2@v2"
ALL_KNOWN_APPROACHES="autocomp pre_explored no_info model_guided model_guided_v2"

# Base neuron image; used for claude (which only needs the venv).
# For codex the --sandbox-image override (nki-bench-sandbox:latest) is used
# because that image also has node/npm/codex pre-installed.
BASE_DOCKER_IMAGE="public.ecr.aws/neuron/pytorch-inference-neuronx:2.9.0-neuronx-py312-sdk2.28.0-ubuntu24.04"
SANDBOX_IMAGE="nki-bench-sandbox:latest"   # for codex; see test-docker-sandbox.sh step 5

# ============================================================
# HELPERS (defined above CLI parsing because --source resolution calls
# _bench_paths during the arg-parse loop below).
# ============================================================
_repo_python() {
    (
        source "$NEURON_VENV_ACTIVATE"
        python "$@"
    )
}

# Query bench-paths (installed as a console script by `pip install -e .`).
# Wraps it in the venv so it works regardless of the caller's environment.
_bench_paths() {
    (
        source "$NEURON_VENV_ACTIVATE"
        bench-paths "$@"
    )
}

# Run nki-profile with nki-library/src on PYTHONPATH so that oracle.py files that
# import from nkilib_src.* work correctly on the host (inside Docker the workspace's
# python-in-venv script already sets PYTHONPATH via setup_agent_workspace.py).
_run_profile() {
    (
        source "$NEURON_VENV_ACTIVATE"
        PYTHONPATH="$NKILIB_SRC:$REPO_ROOT/src:${PYTHONPATH:-}" \
        nki-profile "$@"
    )
}

# ============================================================
# CLI OVERRIDES
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE_SLUG="$2"; shift 2
            # Resolve per-source paths from the canonical registry in
            # triton_nki_transpile.bench_runs.common.SOURCE_PATHS — see the
            # `bench-paths` CLI. An unknown source makes _bench_paths exit 1,
            # which propagates here under `set -e`.
            KERNEL_DIR="$(_bench_paths kernel-dir        "$SOURCE_SLUG")"
            ORACLE_DIR="$(_bench_paths oracle-dir        "$SOURCE_SLUG")"
            BASELINE_CATEGORY="$(_bench_paths baseline-category "$SOURCE_SLUG")"
            SUFFIX_SLUG="$SOURCE_SLUG"
            ;;
        --kernel-dir)       KERNEL_DIR="$2";            shift 2 ;;
        --oracle-dir)       ORACLE_DIR="$2";            shift 2 ;;
        --output-dir)       OUTPUT_DIR_OVERRIDE="$2";   shift 2 ;;
        --kernels)          IFS=',' read -ra KERNELS <<< "$2"; shift 2 ;;
        --agents)           IFS=',' read -ra AGENT_NAMES <<< "$2"; shift 2 ;;
        --model)            MODEL_NAME="$2";            shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2";      shift 2 ;;
        --sandbox-image)    SANDBOX_IMAGE="$2";         shift 2 ;;
        --approaches)       APPROACHES="$2";            shift 2 ;;
        --use-codex-api-key) USE_CODEX_API_KEY=1;       shift ;;
        --no-codex-api-key)  USE_CODEX_API_KEY=0;       shift ;;
        --agent-features)    AGENT_FEATURES_TAG="$2";   shift 2 ;;
        --measurement-method) MEASUREMENT_METHOD_TAG="$2"; shift 2 ;;
        --no-ingest)         SKIP_INGEST=1;             shift ;;
        -h|--help)
            grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Parse and validate the approach list. Every entry MUST be "name@variant".
# Hyphens are normalized to underscores within the name segment.
APPROACHES="${APPROACHES//-/_}"
IFS=',' read -ra APPROACH_SPEC_LIST <<< "$APPROACHES"

# APPROACH_LIST: bare approach names (used as the file-prefix for workspace
# dirs, optimized.py, history.json, etc. — only one variant per approach per
# invocation, since these on-disk names don't carry the variant).
# VARIANT_FOR: assoc array, approach_name -> variant.
declare -a APPROACH_LIST=()
declare -A VARIANT_FOR=()
INGEST_VARIANT_OVERRIDES=""   # comma-separated "name=variant" pairs passed to ingest
for spec in "${APPROACH_SPEC_LIST[@]}"; do
    if [[ "$spec" != *"@"* ]]; then
        echo "Approach spec '$spec' is missing a variant. Use 'name@variant' (e.g. 'model_guided_v2@v2'); no default exists." >&2
        exit 1
    fi
    approach="${spec%@*}"
    variant="${spec#*@}"
    if [[ -z "$approach" || -z "$variant" ]]; then
        echo "Malformed approach spec: '$spec'" >&2
        exit 1
    fi
    case " $ALL_KNOWN_APPROACHES " in
        *" $approach "*) ;;
        *) echo "Unknown approach: '$approach'. Known: $ALL_KNOWN_APPROACHES" >&2; exit 1 ;;
    esac
    if [[ -n "${VARIANT_FOR[$approach]:-}" ]]; then
        echo "Duplicate approach '$approach' in --approaches list (only one variant per approach per invocation; on-disk filenames don't encode the variant)." >&2
        exit 1
    fi
    APPROACH_LIST+=("$approach")
    VARIANT_FOR["$approach"]="$variant"
    if [[ -n "$INGEST_VARIANT_OVERRIDES" ]]; then
        INGEST_VARIANT_OVERRIDES="${INGEST_VARIANT_OVERRIDES},${approach}=${variant}"
    else
        INGEST_VARIANT_OVERRIDES="${approach}=${variant}"
    fi
done

# ============================================================
# PATHS
# ============================================================
PACKAGE_VERSION=$(grep '^version' "$REPO_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
VERSION_SLUG="version_$(echo "$PACKAGE_VERSION" | tr '.' '_')"
AGENTS_SLUG=$(IFS='_'; echo "${AGENT_NAMES[*]}")
MODEL_SLUG=$(echo "$MODEL_NAME" | tr '.-' '_' | tr '/:' '_')
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_DIR_OVERRIDE:-$SCRIPT_DIR/${RUN_PREFIX}${VERSION_SLUG}_${AGENTS_SLUG}_${MODEL_SLUG}_${SUFFIX_SLUG}}"
OUTPUT_DIR="$RUN_DIR/results/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# ============================================================
# CREDENTIAL SETUP — copies (not mounts) of host credential files
# ============================================================
# Each agent gets its own home dir mounted at CONTAINER_HOME.
# Only the specific credential files are copied — not sessions, history, or logs.
TMP_CREDS="$(mktemp -d)"
TMP_CLAUDE_HOME="$TMP_CREDS/claude_home"   # mounted as $CONTAINER_HOME for claude runs
TMP_CODEX_HOME="$TMP_CREDS/codex_home"    # mounted as $CONTAINER_HOME for codex runs
mkdir -p "$TMP_CLAUDE_HOME/.claude" "$TMP_CODEX_HOME/.codex"

cleanup_creds() {
    if [[ -d "$TMP_CREDS" ]]; then rm -rf "$TMP_CREDS"; fi
}
trap cleanup_creds EXIT

_prepare_claude_creds() {
    local src="$HOME/.claude/.credentials.json"
    if [[ ! -f "$src" ]]; then
        echo "[ERROR] ~/.claude/.credentials.json not found" >&2
        return 1
    fi
    cp "$src" "$TMP_CLAUDE_HOME/.claude/.credentials.json"
    chmod 600 "$TMP_CLAUDE_HOME/.claude/.credentials.json"
}

_prepare_codex_creds() {
    local auth_src="$HOME/.codex/auth.json"
    local id_src="$HOME/.codex/installation_id"
    if [[ ! -f "$auth_src" ]]; then
        echo "[ERROR] ~/.codex/auth.json not found" >&2
        return 1
    fi
    cp "$auth_src" "$TMP_CODEX_HOME/.codex/auth.json"
    chmod 600 "$TMP_CODEX_HOME/.codex/auth.json"
    if [[ -f "$id_src" ]]; then
        cp "$id_src" "$TMP_CODEX_HOME/.codex/installation_id"
    fi
    # Disable codex's internal bubblewrap sandbox — Docker already provides isolation.
    # The --sandbox CLI flag is not honoured by `codex exec`; config.toml is the override.
#     cat > "$TMP_CODEX_HOME/.codex/config.toml" <<'EOF'
# [sandbox]
# policy = "danger-full-access"
# EOF
}

# Find the real claude binary (resolve symlink)
CLAUDE_BIN=""
if command -v claude >/dev/null 2>&1; then
    CLAUDE_BIN="$(realpath "$(command -v claude)")"
fi

CONFIG_LOG="$RUN_DIR/config.txt"
cat > "$CONFIG_LOG" <<EOF
Run Configuration
=================
Date:            $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Package version: $PACKAGE_VERSION
Mode:            docker
Base image:      $BASE_DOCKER_IMAGE
Sandbox image:   $SANDBOX_IMAGE
Agents:          ${AGENT_NAMES[*]}
Model:           $MODEL_NAME
Reasoning:       $REASONING_EFFORT
Run dir:         $RUN_DIR
Output dir:      $OUTPUT_DIR
Kernels:
$(printf '  - %s\n' "${KERNELS[@]}")
EOF

echo "============================================================"
echo "NKI Agent Benchmark (Docker)"
echo "Timestamp:    $TIMESTAMP"
echo "Version:      $PACKAGE_VERSION"
echo "Agents:       ${AGENT_NAMES[*]}"
echo "Model:        $MODEL_NAME"
echo "Reasoning:    $REASONING_EFFORT"
echo "Kernels:      ${KERNELS[*]}"
echo "Run dir:      $RUN_DIR"
echo "Output:       $OUTPUT_DIR"
echo "============================================================"
echo ""

RESULTS_JSON="$OUTPUT_DIR/benchmark_results.json"
RESULTS_CSV="$OUTPUT_DIR/benchmark_results.csv"
echo "[]" > "$RESULTS_JSON"

# Base docker run command — the image uses a custom entrypoint so we must override it.
# Run as the current user (not root) so:
#   - claude --permission-mode bypassPermissions is allowed (blocked for root)
#   - workspace files are owned by us, not root
#
# The image sets TEMP=/home/model-server/tmp (a SageMaker default). Claude Code's Bash sandbox
# uses os.tmpdir() which reads TMPDIR/TMP/TEMP env vars. We override TMPDIR=/tmp so the
# sandbox writes its temp dirs to /tmp which is always writable.
CONTAINER_HOME="/tmp/agent_home"

DR="docker run --rm --entrypoint bash --user $(id -u):$(id -g) -e HOME=$CONTAINER_HOME -e TMPDIR=/tmp"

# Honor `nki_lnc_serialize = true` on the active agent_features tag — exports
# NKI_LNC_SERIALIZE=1 into the container, which the workspace's python-in-venv
# shim picks up and wraps `python` in flock(/tmp/nki_lnc.lock). Prevents the
# concurrent-shell-tool race during initialization of /dev/neuron0.
# Off-by-default — only the
# `*_lncfix` tags currently opt in. AGENT_FEATURES_TAG must be set before
# this runs (matrix-driven flow always sets it via --agent-features).
_LNC_ENV=""
if [[ -n "${AGENT_FEATURES_TAG:-}" ]]; then
    if python3 -c "
import sys, tomllib, pathlib
r = tomllib.loads(pathlib.Path('$SCRIPT_DIR/agent_features.toml').read_text())
sys.exit(0 if r.get('$AGENT_FEATURES_TAG', {}).get('nki_lnc_serialize') is True else 1)
" 2>/dev/null; then
        _LNC_ENV="-e NKI_LNC_SERIALIZE=1"
        echo "[lnc] $AGENT_FEATURES_TAG has nki_lnc_serialize=true; exporting NKI_LNC_SERIALIZE=1 into container"
    fi
fi

# Consistency guard: when an agent_features tag is given explicitly, the agent(s)
# we launch must match the agent recorded in that tag. Otherwise the run is
# ingested under a tag that misdescribes it, and graphectory later parses it with
# the wrong adapter. (Auto-derived tags — no --agent-features — can't mismatch:
# they're built from AGENT_NAMES[0] at ingest time below.)
if [[ -n "${AGENT_FEATURES_TAG:-}" ]]; then
    _TAG_AGENT="$(python3 -c "
import sys, tomllib, pathlib
r = tomllib.loads(pathlib.Path('$SCRIPT_DIR/agent_features.toml').read_text())
t = r.get('$AGENT_FEATURES_TAG')
if t is None: sys.exit(3)
print(t.get('agent', ''))
" 2>/dev/null)" || { echo "[ERROR] agent_features tag '$AGENT_FEATURES_TAG' not found in agent_features.toml" >&2; exit 1; }
    for _a in "${AGENT_NAMES[@]}"; do
        if [[ "$_a" != "$_TAG_AGENT" ]]; then
            echo "[ERROR] agent mismatch: launching agent '$_a' but agent_features tag '$AGENT_FEATURES_TAG' declares agent='$_TAG_AGENT'." >&2
            echo "        Use --agents $_TAG_AGENT (or choose a tag whose agent matches) so the run is ingested and parsed correctly." >&2
            exit 1
        fi
    done
fi

# Common docker volume mounts for NKI access (venv + source).
# nki-library/src is mounted so oracle.py files can import nkilib_src.* inside the container.
# The workspace is mounted separately per invocation at its host path.
_nki_mounts() {
    echo "-v $NEURON_VENV:$NEURON_VENV:ro"
    echo "-v $REPO_ROOT/src:/workspace/nki-opt-tools/src:ro"
    echo "-v $NKILIB_SRC:/workspace/nki-opt-tools/nki-library/src:ro"
}

# NeuronCore allocation probe.
#
# Why this exists: opt38 SPMD requires `kernel[2]`. If the host driver leaks state
# from a prior container (OOM, watchdog kill), subsequent containers can see
# `Available:0` from NRT and silently ship baseline (Case 1 in
# reports/mg_v2_giveup_traces.md). Whether a given container can *allocate* cores
# is decided per `docker run` invocation, so the check must run inside the agent's
# own container — not a sibling container with the same image. Also, `neuron-ls`
# only reports static topology; only an actual `kernel[2]` launch exercises the
# `nrt_allocate_neuron_cores` path where the real failure occurs.
#
# Implementation: a small NKI kernel in benchmark-opt/neuron_probe.py launches
# `kernel[2]`. We mount it RO into every agent container and exec it as the first
# thing in the agent's bash `-c` command. Exit 99 if it fails — the approach loop
# catches that and offers an interactive retry.
NEURON_PROBE_HOST_PATH="$SCRIPT_DIR/neuron_probe.py"
NEURON_PROBE_CONTAINER_PATH="/tmp/neuron_probe.py"
if [[ ! -f "$NEURON_PROBE_HOST_PATH" ]]; then
    echo "[ERROR] Probe script missing at $NEURON_PROBE_HOST_PATH" >&2
    exit 1
fi


# ============================================================
# AGENT INVOCATION HELPERS (Docker-wrapped)
# ============================================================

_invoke_claude() {
    local workspace="$1"
    local log_file="$2"
    local model="$3"

    if [[ -z "$CLAUDE_BIN" ]]; then
        echo "[ERROR] claude not found in PATH" | tee -a "$log_file"
        return 1
    fi
    _prepare_claude_creds

    $DR \
        --device=/dev/neuron0 \
        $_LNC_ENV \
        $(_nki_mounts) \
        -v "$workspace:$workspace:rw" \
        -v "$CLAUDE_BIN:/usr/local/bin/claude:ro" \
        -v "$TMP_CLAUDE_HOME:$CONTAINER_HOME:rw" \
        -v "$NEURON_PROBE_HOST_PATH:$NEURON_PROBE_CONTAINER_PATH:ro" \
        -w "$workspace" \
        "$BASE_DOCKER_IMAGE" \
        -c "source $NEURON_VENV/bin/activate && python3 $NEURON_PROBE_CONTAINER_PATH || exit 99
            claude \
                --print \
                --model '$model' \
                --permission-mode bypassPermissions \
                --output-format stream-json \
                --verbose \
                --include-partial-messages \
                --no-session-persistence \
                'Read CLAUDE.md for your task instructions, then complete the NKI kernel optimization task described there.'" \
        >> "$log_file" 2>&1
}

_invoke_codex() {
    local workspace="$1"
    local log_file="$2"
    local model="$3"
    local reasoning_effort="$4"

    # Verify the sandbox image (with node/npm/codex) exists
    if ! docker image inspect "$SANDBOX_IMAGE" >/dev/null 2>&1; then
        echo "[ERROR] Sandbox image '$SANDBOX_IMAGE' not found." | tee -a "$log_file"
        echo "[ERROR] Build it first: ./benchmark-opt/build-sandbox-image.sh" | tee -a "$log_file"
        return 1
    fi

    local codex_auth_envvar=""
    if [[ "${USE_CODEX_API_KEY:-0}" -eq 1 ]]; then
        if [[ -z "${CODEX_API_KEY:-}" ]]; then
            echo "[ERROR] USE_CODEX_API_KEY=1 but CODEX_API_KEY is not set" | tee -a "$log_file"
            return 1
        fi
        codex_auth_envvar="CODEX_API_KEY=$CODEX_API_KEY"
        # Write sandbox config only (no auth files needed)
#         cat > "$TMP_CODEX_HOME/.codex/config.toml" <<'EOF'
# [sandbox]
# policy = "danger-full-access"
# EOF
    else
        _prepare_codex_creds
    fi

    $DR \
        --device=/dev/neuron0 \
        --security-opt seccomp=unconfined \
        --cap-add SYS_ADMIN \
        $_LNC_ENV \
        $(_nki_mounts) \
        -v "$workspace:$workspace:rw" \
        -v "$TMP_CODEX_HOME:$CONTAINER_HOME:rw" \
        -v "$NEURON_PROBE_HOST_PATH:$NEURON_PROBE_CONTAINER_PATH:ro" \
        -w "$workspace" \
        "$SANDBOX_IMAGE" \
        -c "
            source $NEURON_VENV/bin/activate && python3 $NEURON_PROBE_CONTAINER_PATH || exit 99
            PATH=\"$workspace:\$PATH\"
            $codex_auth_envvar \
            codex exec \
                --json \
                --model '$model' \
                --dangerously-bypass-approvals-and-sandbox \
                -c 'model_reasoning_effort=\"$reasoning_effort\"' \
                --skip-git-repo-check \
                'Read AGENTS.md for your task instructions, then complete the NKI kernel optimization task described there.'
        " \
        >> "$log_file" 2>&1
}

_run_agent() {
    local agent="$1"
    local workspace="$2"
    local log_file="$3"
    local model="$4"
    local reasoning_effort="$5"
    case "$agent" in
        claude) _invoke_claude "$workspace" "$log_file" "$model" ;;
        codex)  _invoke_codex  "$workspace" "$log_file" "$model" "$reasoning_effort" ;;
        *) echo "[ERROR] Unknown agent: $agent" | tee -a "$log_file"; return 1 ;;
    esac
}

# ============================================================
# WRITE SYNTHETIC HISTORY (compatible with compare-results.py)
# ============================================================
_write_history() {
    local history_file="$1"
    local kernel_file="$2"
    local opt_file="$3"
    local baseline_json="$4"
    local result_json="$5"

    _repo_python - <<PYEOF
import json, pathlib

baseline = json.loads(r"""$baseline_json""")
result = json.loads(r"""$result_json""")
kernel_code = pathlib.Path(r"$kernel_file").read_text()

final_lat = result.get("latency_ms")
base_lat = baseline.get("latency_ms")
accepted = result.get("success", False) and final_lat is not None and (base_lat is None or final_lat < base_lat)

opt_code = ""
opt_path = r"$opt_file"
if opt_path and pathlib.Path(opt_path).exists():
    opt_code = pathlib.Path(opt_path).read_text()

history = [
    {"iteration": 0, "plan": None, "code": kernel_code,
     "profile_result": baseline, "accepted": True},
    {"iteration": 1, "plan": "(agent run)", "code": opt_code or kernel_code,
     "profile_result": result, "accepted": accepted},
]
pathlib.Path(r"$history_file").write_text(json.dumps(history, indent=2))
PYEOF
}

# ============================================================
# MAIN LOOP: agent × kernel × approach
# ============================================================
# Note: NeuronCore visibility is verified per-invocation *inside* each agent's
# container by the probe script mounted at $NEURON_PROBE_CONTAINER_PATH. If the
# probe fails (cores not visible, or stale driver state holding them), the agent
# is never invoked and exits 99 — the approach loop below will record that as an
# "agent invocation failed" run rather than a silent baseline ship.
for AGENT in "${AGENT_NAMES[@]}"; do
    echo "Agent: $AGENT (docker)"
    echo ""

    for KERNEL_NAME in "${KERNELS[@]}"; do
        KERNEL_FILE="$KERNEL_DIR/${KERNEL_NAME}.py"
        if [[ ! -f "$KERNEL_FILE" ]]; then
            echo "[SKIP] $KERNEL_NAME: kernel file not found at $KERNEL_FILE"
            continue
        fi
        # Oracle naming is per-source-deterministic (_logic.py vs .py), so we
        # ask bench-paths rather than guess. A missing oracle prints to stderr
        # and exits non-zero; capture both stdout and exit status.
        if ORACLE_FILE="$(_bench_paths oracle "$SOURCE_SLUG" "$KERNEL_NAME" 2>/dev/null)"; then
            :
        else
            echo "[SKIP] $KERNEL_NAME: oracle not found (per SOURCE_PATHS[$SOURCE_SLUG].oracle_suffix)"
            continue
        fi

        echo "------------------------------------------------------------"
        echo "Agent: $AGENT  Kernel: $KERNEL_NAME"
        echo "  kernel: $KERNEL_FILE"
        echo "  oracle: $ORACLE_FILE"

        KERNEL_OUTPUT_DIR="$OUTPUT_DIR/$KERNEL_NAME"
        mkdir -p "$KERNEL_OUTPUT_DIR"

        # Load cached baseline profile (produced by profile-baselines.py).
        # Baselines are deterministic, so we don't re-run them per benchmark.
        if [[ -z "${BASELINE_CATEGORY:-}" ]]; then
            echo "[ERROR] No baseline category set — use --source so the runner knows which baseline-profiles/<category>/ to read from." >&2
            exit 1
        fi
        BASELINE_CACHE="$REPO_ROOT/benchmark-opt/baseline-profiles/$BASELINE_CATEGORY/$KERNEL_NAME.json"
        if [[ ! -f "$BASELINE_CACHE" ]]; then
            echo "[ERROR] No cached baseline at $BASELINE_CACHE. Run benchmark-opt/profile-baselines.py to generate it." >&2
            exit 1
        fi
        # Extract the inner profile_result (matches the live profiler's output
        # shape, which downstream consumers expect).
        BASELINE_JSON_FILE="$KERNEL_OUTPUT_DIR/baseline_profile.json"
        BASELINE_LATENCY=$(_repo_python -c "
import json, sys
from pathlib import Path
d = json.loads(Path('$BASELINE_CACHE').read_text())
result = d.get('profile_result') or {}
Path('$BASELINE_JSON_FILE').write_text(json.dumps(result))
if result.get('success') and result.get('latency_ms') is not None:
    print(f\"{result['latency_ms']:.3f}\")
else:
    print('FAILED')
")
        BASELINE_JSON=$(cat "$BASELINE_JSON_FILE")
        echo "  Baseline latency (cached): $BASELINE_LATENCY ms"

        for APPROACH_SLUG in "${APPROACH_LIST[@]}"; do
            APPROACH="${APPROACH_SLUG//_/-}"
            VARIANT="${VARIANT_FOR[$APPROACH_SLUG]}"  # always set; arg-parser rejects empty
            APPROACH_DISPLAY="${APPROACH}@${VARIANT}"
            WORKSPACE="$KERNEL_OUTPUT_DIR/${APPROACH_SLUG}_workspace"
            LOG_FILE="$KERNEL_OUTPUT_DIR/${APPROACH_SLUG}_run.jsonl"
            OUT_KERNEL="$KERNEL_OUTPUT_DIR/${APPROACH_SLUG}_optimized.py"
            HISTORY_FILE="$KERNEL_OUTPUT_DIR/${APPROACH_SLUG}_optimized.history.json"

            echo ""
            echo "  [$APPROACH_DISPLAY] Setting up workspace..."
            _repo_python "$SCRIPT_DIR/setup_agent_workspace.py" \
                --kernel-path "$KERNEL_FILE" \
                --oracle-path "$ORACLE_FILE" \
                --approach "$APPROACH" \
                --approach-variant "$VARIANT" \
                --output-dir "$WORKSPACE" \
                --baseline-latency-ms "$BASELINE_LATENCY" \
                --baseline-profile-json "$BASELINE_JSON_FILE" \
                --docker-mode \
                > /dev/null

            echo "  [$APPROACH_DISPLAY] Running $AGENT agent (docker)..."
            : > "$LOG_FILE"

            # Retry loop: rc=99 from the agent container means the in-container
            # NeuronCore probe failed (couldn't allocate 2 LNCs). That's almost
            # always recoverable by clearing stuck driver state on the host, so
            # prompt the user before declaring the iteration lost.
            AGENT_RC=0
            while true; do
                if _run_agent "$AGENT" "$WORKSPACE" "$LOG_FILE" "$MODEL_NAME" "$REASONING_EFFORT"; then
                    AGENT_RC=0
                    break
                fi
                AGENT_RC=$?
                if [[ $AGENT_RC -ne 99 ]]; then break; fi

                echo ""
                echo "  [$APPROACH_DISPLAY] NeuronCore probe failed inside the agent container (rc=99)."
                echo "  ----- last 20 lines of $LOG_FILE -----" >&2
                tail -20 "$LOG_FILE" 2>/dev/null | sed 's/^/    /' >&2 || true
                echo "  ----------------------------------------" >&2
                echo "  Likely cause: stuck driver state from a prior run."
                echo "  Suggested fixes before retrying:"
                echo "    - Run 'neuron-ls' on the host; check for orphan processes"
                echo "    - 'sudo modprobe -r neuron && sudo modprobe neuron' to reload the driver"
                echo "    - or reboot the instance"
                echo ""
                if [[ -r /dev/tty ]]; then
                    printf "  Retry this iteration? [y]es / [s]kip / [a]bort run: "
                    read -r ANS </dev/tty
                else
                    echo "  (no controlling tty; defaulting to 'skip')"
                    ANS="s"
                fi
                case "$ANS" in
                    [yY]*)
                        echo "  Retrying..."
                        : > "$LOG_FILE"
                        continue
                        ;;
                    [aA]*)
                        echo "  Aborting entire run."
                        exit 1
                        ;;
                    *)
                        echo "  Skipping this iteration."
                        break
                        ;;
                esac
            done

            if [[ $AGENT_RC -ne 0 ]]; then
                if [[ $AGENT_RC -eq 99 ]]; then
                    echo "  [$APPROACH_DISPLAY] Skipped: neuron-core probe failed. See $LOG_FILE"
                    RESULT_JSON='{"success":false,"latency_ms":null,"error":"neuron core probe failed"}'
                else
                    echo "  [$APPROACH_DISPLAY] Agent invocation failed (rc=$AGENT_RC). See $LOG_FILE"
                    RESULT_JSON='{"success":false,"latency_ms":null,"error":"agent invocation failed"}'
                fi
            elif [[ ! -s "$WORKSPACE/optimized_kernel.py" ]] || \
                 grep -q "^# Write the final optimized kernel here" "$WORKSPACE/optimized_kernel.py" 2>/dev/null; then
                echo "  [$APPROACH_DISPLAY] Warning: agent did not write optimized_kernel.py"
                RESULT_JSON='{"success":false,"latency_ms":null,"error":"agent did not produce optimized_kernel.py"}'
            else
                cp "$WORKSPACE/optimized_kernel.py" "$OUT_KERNEL"
                echo "  [$APPROACH_DISPLAY] Profiling agent output..."
                RESULT_JSON=$(
                    _run_profile \
                        --kernel-path "$WORKSPACE/optimized_kernel.py" \
                        --oracle-path "$ORACLE_FILE" \
                        --bench3 \
                        2>/dev/null
                ) || RESULT_JSON='{"success":false,"latency_ms":null,"error":"profiler error"}'
            fi

            RESULT_LATENCY=$(echo "$RESULT_JSON" | _repo_python -c "
import sys, json
d = json.loads(sys.stdin.read())
if d.get('success') and d.get('latency_ms') is not None:
    print(f\"{d['latency_ms']:.3f}\")
else:
    print('FAILED')
")
            echo "  [$APPROACH_DISPLAY] Result latency: $RESULT_LATENCY ms"

            _write_history "$HISTORY_FILE" "$KERNEL_FILE" \
                "${OUT_KERNEL:-}" "$BASELINE_JSON" "$RESULT_JSON"

            echo "  [$APPROACH_DISPLAY] Done. Log: $LOG_FILE | History: $HISTORY_FILE"
        done

        echo ""
    done

    echo ""
done

echo "============================================================"
echo "All runs complete. Comparing results..."
echo ""

_repo_python "$SCRIPT_DIR/compare-results.py" \
    --results-dir "$OUTPUT_DIR" \
    --output-json "$RESULTS_JSON" \
    --output-csv "$RESULTS_CSV"

echo ""
echo "Results JSON: $RESULTS_JSON"
echo "Results CSV:  $RESULTS_CSV"
echo "============================================================"

# ============================================================
# INGEST INTO benchmark-opt/runs/
# ============================================================
# Convert this timestamp's per-(kernel,approach) outputs into the flat runs/<run_id>/
# layout consumed by `bench-manifest`/`bench-report`/`bench-status`. Skipped with
# --no-ingest. agent_features is auto-derived from PACKAGE_VERSION + the first
# agent in AGENT_NAMES; override with --agent-features TAG. The default
# measurement_method is bench3_geomean since the harness's profiler uses bench3.
# Approach variants (always pinned via --approaches name@variant) are
# forwarded via --approach-variants. All approaches must have an explicit
# variant — there is no default.
SKIP_INGEST="${SKIP_INGEST:-0}"
if [[ "$SKIP_INGEST" -eq 0 ]]; then
    if [[ -z "${SUFFIX_SLUG:-}" ]]; then
        echo "[ingest] WARNING: SUFFIX_SLUG is empty — skipping ingest. Set it in the kernel-set block above."
    else
        INGEST_AGENT="${AGENT_NAMES[0]}"
        # Auto-derive agent_features from (agent, version) unless overridden.
        if [[ -z "${AGENT_FEATURES_TAG:-}" ]]; then
            case "$PACKAGE_VERSION" in
                0.1.14|0.1.15|0.1.16) AF_SUFFIX="v1_opt37" ;;
                *) AF_SUFFIX="v2_opt38" ;;
            esac
            AGENT_FEATURES_TAG="${INGEST_AGENT}_${AF_SUFFIX}"
        fi
        MEASUREMENT_METHOD_TAG="${MEASUREMENT_METHOD_TAG:-bench3_geomean}"
        echo ""
        echo "[ingest] -> benchmark-opt/runs/  agent_features=$AGENT_FEATURES_TAG  measurement_method=$MEASUREMENT_METHOD_TAG  approach_variants=$INGEST_VARIANT_OVERRIDES"
        ( cd "$SCRIPT_DIR" && source "$NEURON_VENV_ACTIVATE" && bench-ingest \
            --results-dir "$OUTPUT_DIR" \
            --agent "$INGEST_AGENT" \
            --model "$MODEL_NAME" \
            --source "$SUFFIX_SLUG" \
            --agent-features "$AGENT_FEATURES_TAG" \
            --measurement-method "$MEASUREMENT_METHOD_TAG" \
            --package-version "$PACKAGE_VERSION" \
            --reasoning-effort "$REASONING_EFFORT" \
            --approach-variants "$INGEST_VARIANT_OVERRIDES" \
            --no-git ) || echo "[ingest] failed — see above. Re-run with: bench-ingest --results-dir $OUTPUT_DIR ..."
    fi
fi
