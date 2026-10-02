"""Shared helpers for the bench_runs package.

Keep this module dependency-light (stdlib + tomllib) so it imports cleanly even
when the Neuron venv isn't active. Heavier deps (pandas, the profiler) live in
the modules that need them.
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Repo-root paths anchored off this file so commands can be run from anywhere.
# Layout: <REPO_ROOT>/src/triton_nki_transpile/bench_runs/common.py
PKG_DIR        = Path(__file__).resolve().parent
REPO_ROOT      = PKG_DIR.parents[2]
BENCHMARK_OPT  = REPO_ROOT / "benchmark-opt"

RUNS_DIR             = BENCHMARK_OPT / "runs"
REPORTS_DIR          = BENCHMARK_OPT / "reports"
SNAPSHOTS_DIR        = REPORTS_DIR / "snapshots"
MANIFEST_PATH        = BENCHMARK_OPT / "manifest.csv"

AGENT_FEATURES_TOML  = BENCHMARK_OPT / "agent_features.toml"
MEASUREMENT_METHODS_TOML = BENCHMARK_OPT / "measurement_methods.toml"
APPROACHES_TOML      = BENCHMARK_OPT / "approaches.toml"
MATRIX_TOML          = BENCHMARK_OPT / "matrix.toml"
KERNELS_TOML         = BENCHMARK_OPT / "kernels.toml"

# The approaches tracked historically. Order matches the canonical CSV.
# (Renamed from MENUS on 2026-05-18 — see approaches.toml header.)
APPROACHES = ["no_info", "autocomp", "pre_explored", "model_guided", "model_guided_v2"]

# Display labels (column headers in reports). Keyed by approach name; for
# non-default variants the report appends "@<variant>" to the label.
APPROACH_LABELS = {
    "no_info":         "NoInfo",
    "autocomp":        "AutoComp",
    "pre_explored":    "PreExplored",
    "model_guided":    "ModelGuided",
    "model_guided_v2": "ModelGuidedV2",
}

# The kernel-source directories the project tracks.
SOURCES = ["reference", "nl_to_nki", "nki_library", "nkibench", "compiler_gen"]

# Per-source filesystem layout. SINGLE source of truth — every other location
# (shell scripts via `bench-paths`, profile-baselines.py, measure.py, …) reads
# from here. Paths are relative to REPO_ROOT.
#
# Fields:
#   kernel_dir        — directory holding the kernel <name>.py files
#   oracle_dir        — directory holding the oracle for each kernel
#   oracle_suffix     — appended to <kernel> to form the oracle filename;
#                       per-source-deterministic so no _logic.py/.py fallback
#                       is ever needed (oracle_path_for is strict).
#   baseline_category — sub-directory under benchmark-opt/baseline-profiles/
#                       where cached baseline profiles for this source live.
#                       (For `reference` this is "logic", not "reference",
#                       because reference kernels are byte-identical to the
#                       logic/ originals.)
SOURCE_PATHS: dict[str, dict[str, str]] = {
    "reference":   {"kernel_dir":        "benchmark-opt/reference",
                    "oracle_dir":        "src/triton_nki_transpile/kernels/tests/logic",
                    "oracle_suffix":     "_logic.py",
                    "baseline_category": "logic"},
    "nl_to_nki":   {"kernel_dir":        "benchmark-opt/nl-to-nki",
                    "oracle_dir":        "src/triton_nki_transpile/kernels/tests/logic",
                    "oracle_suffix":     "_logic.py",
                    "baseline_category": "nl_to_nki"},
    "nki_library": {"kernel_dir":        "benchmark-opt/nki-library/kernels",
                    "oracle_dir":        "benchmark-opt/nki-library/oracles",
                    "oracle_suffix":     ".py",
                    "baseline_category": "nki_library"},
    "nkibench":    {"kernel_dir":        "benchmark-opt/NKIBenchB2/kernels",
                    "oracle_dir":        "benchmark-opt/NKIBenchB2/oracles",
                    "oracle_suffix":     ".py",
                    "baseline_category": "nkibench"},
    "compiler_gen": {"kernel_dir":        "benchmark-opt/CompilerGenBenchB2/kernels",
                     "oracle_dir":        "benchmark-opt/CompilerGenBenchB2/oracles",
                     "oracle_suffix":     ".py",
                     "baseline_category": "compiler_gen"},
}

BASELINE_PROFILES_DIR = BENCHMARK_OPT / "baseline-profiles"

# Directly lifted from analyze_experiments.ipynb's BLACKLIST_KERNELS. Used by
# migrate.py to seed the cheat flag on legacy runs that match.
# Tuple is (source, kernel, approach) — pre-2026-05-18 this was (..., menu).
LEGACY_CHEAT_BLACKLIST: list[tuple[str, str, str]] = [
    ("nl_to_nki", "sparsemax", "pre_explored"),
    ("nl_to_nki", "sparsemax", "no_info"),
    ("nl_to_nki", "sparsemax", "model_guided"),
]
LEGACY_CHEAT_REASON = "dma_copy identity / pytorch offload of core computation"

# ---------- Registry loaders ----------

def load_agent_features() -> dict[str, dict]:
    return tomllib.loads(AGENT_FEATURES_TOML.read_text())

def load_measurement_methods() -> dict[str, dict]:
    return tomllib.loads(MEASUREMENT_METHODS_TOML.read_text())

def load_matrix() -> dict:
    return tomllib.loads(MATRIX_TOML.read_text())


# Cell names appear in queue headers, CSV filenames, and CLI args, so restrict
# them to a filesystem/shell-safe alphabet. Underscores and hyphens are fine.
_CELL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def load_matrix_cells() -> tuple[list[dict], dict]:
    """Return `(cells, defaults)` from matrix.toml with each cell's `name`
    field validated. `defaults` is the top-level table (default_target_runs,
    default_reasoning_effort, ...).

    Every cell MUST declare a unique `name` (snake_case-ish slug — letters,
    digits, `_`, `-`; must start with a letter or digit). The name is used as
    the cell's identifier in queue headers, CSV filenames, and `--cell` flags.
    """
    matrix = load_matrix()
    cells = list(matrix.get("cell", []))
    defaults = {k: v for k, v in matrix.items() if k != "cell"}

    seen: dict[str, int] = {}
    for i, cell in enumerate(cells):
        name = cell.get("name")
        if not name:
            raise ValueError(
                f"matrix.toml: [[cell]] #{i} is missing the required `name` "
                f"field. Every cell needs a unique short identifier."
            )
        if not isinstance(name, str) or not _CELL_NAME_RE.match(name):
            raise ValueError(
                f"matrix.toml: [[cell]] name={name!r} is not a valid identifier "
                f"(expected [A-Za-z0-9][A-Za-z0-9_-]*)."
            )
        if name in seen:
            raise ValueError(
                f"matrix.toml: duplicate cell name {name!r} "
                f"(first at index {seen[name]}, again at index {i})."
            )
        seen[name] = i
    return cells, defaults


def find_cell_by_name(name: str) -> dict:
    """Return the matrix cell with this `name`. Raises KeyError if not found."""
    cells, _ = load_matrix_cells()
    for cell in cells:
        if cell["name"] == name:
            return cell
    known = sorted(c["name"] for c in cells)
    raise KeyError(f"No matrix.toml cell named {name!r}. Known: {known}")

def require_agent_features(tag: str) -> dict:
    reg = load_agent_features()
    if tag not in reg:
        raise KeyError(
            f"Unknown agent_features tag: {tag!r}. "
            f"Known: {sorted(reg.keys())}. "
            f"Add it to {AGENT_FEATURES_TOML.relative_to(REPO_ROOT)} first."
        )
    return reg[tag]

def require_measurement_method(tag: str) -> dict:
    reg = load_measurement_methods()
    if tag not in reg:
        raise KeyError(
            f"Unknown measurement_method tag: {tag!r}. "
            f"Known: {sorted(reg.keys())}. "
            f"Add it to {MEASUREMENT_METHODS_TOML.relative_to(REPO_ROOT)} first."
        )
    return reg[tag]

# ---------- Per-source path helpers ----------
#
# Single resolver per artifact type. Strict per-source naming — `oracle_suffix`
# in SOURCE_PATHS gives the deterministic suffix, no `_logic.py`/`.py` fallback
# is performed. Shell scripts get at these through the `bench-paths` CLI.

def _require_source(source: str) -> dict[str, str]:
    if source not in SOURCE_PATHS:
        raise KeyError(f"Unknown source: {source!r}. Known: {SOURCES}")
    return SOURCE_PATHS[source]

def kernel_dir_for(source: str) -> Path:
    return REPO_ROOT / _require_source(source)["kernel_dir"]

def oracle_dir_for(source: str) -> Path:
    return REPO_ROOT / _require_source(source)["oracle_dir"]

def baseline_category_for(source: str) -> str:
    return _require_source(source)["baseline_category"]

def kernel_path_for(source: str, kernel: str) -> Path:
    """Absolute path to the kernel `.py` file. Does not check existence; use
    `list_kernels_on_disk(source)` if you need to know what's actually present."""
    return kernel_dir_for(source) / f"{kernel}.py"

def oracle_path_for(source: str, kernel: str, *, must_exist: bool = True) -> Path:
    """Absolute path to the kernel's oracle file. Strict: uses the source's
    `oracle_suffix` (no `_logic.py` vs `.py` fallback). Raises FileNotFoundError
    when `must_exist` and the file is missing — that's almost always a bug."""
    sp = _require_source(source)
    path = REPO_ROOT / sp["oracle_dir"] / f"{kernel}{sp['oracle_suffix']}"
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"oracle for ({source!r}, {kernel!r}) not at {path.relative_to(REPO_ROOT)}"
        )
    return path

def baseline_profile_path_for(source: str, kernel: str) -> Path:
    """Path to the cached baseline profile JSON for (source, kernel).
    Layout: benchmark-opt/baseline-profiles/<baseline_category>/<kernel>.json."""
    return BASELINE_PROFILES_DIR / baseline_category_for(source) / f"{kernel}.json"

def baseline_profile_is_usable(source: str, kernel: str) -> bool:
    """True iff the cached baseline JSON exists AND records a successful profile.

    The runner reads `profile_result.success` and `profile_result.latency_ms`
    to compute speedup; a failed-baseline file (kernel/oracle bug, or marked
    `skipped: true`) is useless and should be treated the same as a missing
    file. Use this instead of `baseline_profile_path_for(...).exists()` for
    gating decisions (pre-flight checks, queue exclusion)."""
    path = baseline_profile_path_for(source, kernel)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("skipped"):
        return False
    pr = data.get("profile_result") or {}
    return bool(pr.get("success")) and pr.get("latency_ms") is not None


# ---------- Kernels registry ----------
#
# Truth for *which kernels exist* lives on disk under SOURCE_PATHS[src].kernel_dir.
# kernels.toml is an optional overlay providing skip_default / tags / notes.

def list_kernels_on_disk(source: str) -> list[str]:
    """Sorted kernel names — stems of `*.py` files under the source's kernel dir.
    Files whose name starts with `_` (e.g. `__init__.py`) are ignored."""
    kdir = kernel_dir_for(source)
    if not kdir.is_dir():
        return []
    return sorted(p.stem for p in kdir.glob("*.py") if not p.name.startswith("_"))


def load_kernels_registry() -> dict[str, dict[str, dict]]:
    """Parse kernels.toml into {source: {kernel: attrs}}. Empty dict if missing.

    Validates that every (source, kernel) referenced actually exists on disk so
    typos are caught at load time (status.py / runner startup)."""
    if not KERNELS_TOML.exists():
        return {}
    raw = tomllib.loads(KERNELS_TOML.read_text())
    out: dict[str, dict[str, dict]] = {}
    for src, body in raw.items():
        if src not in SOURCES:
            raise ValueError(
                f"kernels.toml: unknown source {src!r}. Known: {SOURCES}"
            )
        if not isinstance(body, dict):
            raise ValueError(
                f"kernels.toml: [{src}] must be a table of [{src}.<kernel>] entries"
            )
        disk = set(list_kernels_on_disk(src))
        for kernel, attrs in body.items():
            if not isinstance(attrs, dict):
                raise ValueError(
                    f"kernels.toml: [{src}.{kernel}] must be a table"
                )
            if kernel not in disk:
                raise ValueError(
                    f"kernels.toml: [{src}.{kernel}] not found on disk under "
                    f"{SOURCE_PATHS[src]['kernel_dir']}/{kernel}.py"
                )
            # An entry referencing a kernel without an oracle is almost always
            # a stale entry — surface it here, not at runtime.
            oracle = oracle_path_for(src, kernel, must_exist=False)
            if not oracle.exists():
                raise ValueError(
                    f"kernels.toml: [{src}.{kernel}] has no oracle file at "
                    f"{oracle.relative_to(REPO_ROOT)}"
                )
        out[src] = body
    return out


def kernels_for_source(source: str,
                       registry: dict | None = None,
                       *, include_skipped: bool = False) -> list[str]:
    """Kernels available for `source` after applying the registry.

    Returns disk kernels minus those flagged `skip_default = true`. Set
    `include_skipped=True` to ignore the flag (e.g. for an audit listing)."""
    disk = list_kernels_on_disk(source)
    if include_skipped:
        return disk
    if registry is None:
        registry = load_kernels_registry()
    reg_src = registry.get(source, {})
    return [k for k in disk if not reg_src.get(k, {}).get("skip_default", False)]


def resolve_cell_kernels(spec,
                         sources: list[str],
                         registry: dict | None = None) -> dict[str, list[str]]:
    """Resolve a matrix-cell `kernels` value into a {source: [kernel, ...]} map.

    Supported spec shapes:
      - `"all"` — all on-disk kernels per source, minus `skip_default = true`.
      - `list[str]` — same names applied to every source; each name MUST exist
        on disk in every listed source (use the dict form to vary by source).
        `skip_default` is bypassed (explicit listing is opt-in).
      - `dict[str, spec]` — per-source spec. The dict's keys must equal `sources`
        exactly; each value is itself an "all" or a list. `skip_default` applies
        only within sub-specs that resolve as "all".
    """
    if registry is None:
        registry = load_kernels_registry()

    if spec == "all":
        return {s: kernels_for_source(s, registry) for s in sources}

    if isinstance(spec, list):
        out: dict[str, list[str]] = {}
        for s in sources:
            disk = set(list_kernels_on_disk(s))
            missing = [k for k in spec if k not in disk]
            if missing:
                raise ValueError(
                    f"kernels list references {missing!r} which don't exist in "
                    f"source {s!r}. Use a per-source dict (kernels = {{ {s} = [...] }}) "
                    f"to vary kernels per source."
                )
            out[s] = list(spec)
        return out

    if isinstance(spec, dict):
        if set(spec.keys()) != set(sources):
            extra = sorted(set(spec.keys()) - set(sources))
            missing = sorted(set(sources) - set(spec.keys()))
            raise ValueError(
                f"kernels dict keys must match the cell's `sources` exactly. "
                f"extra={extra}, missing={missing}. sources={sources}"
            )
        out = {}
        for s, sub in spec.items():
            sub_map = resolve_cell_kernels(sub, [s], registry)
            out[s] = sub_map[s]
        return out

    raise TypeError(
        f"Unsupported kernels spec type {type(spec).__name__!r}; "
        f'use "all", [list], or a {{source: spec}} dict.'
    )


# ---------- Approaches registry ----------

@dataclass
class Approach:
    """One row of approaches.toml: an approach name + its known variants.

    There is no default_variant — every reference is explicit `name@variant`.
    """
    name: str
    variants: dict[str, dict]  # variant name -> {description, ...}

    def require_variant(self, variant: str) -> dict:
        if variant not in self.variants:
            raise KeyError(
                f"Unknown variant {variant!r} of approach {self.name!r}. "
                f"Known: {sorted(self.variants.keys())}. "
                f"Add it to {APPROACHES_TOML.relative_to(REPO_ROOT)} first."
            )
        return self.variants[variant]


def load_approaches() -> dict[str, Approach]:
    """Parse approaches.toml into {name: Approach}.

    Each top-level table is an approach; nested tables are its variants.
    Empty approach bodies (no variants) are rejected.
    """
    raw = tomllib.loads(APPROACHES_TOML.read_text())
    out: dict[str, Approach] = {}
    for name, body in raw.items():
        variants = {k: v for k, v in body.items() if isinstance(v, dict)}
        if not variants:
            raise ValueError(
                f"approaches.toml: [{name}] has no variants. Add at least "
                f"[{name}.v1] with a description."
            )
        out[name] = Approach(name=name, variants=variants)
    return out


def require_approach(name: str, variant: str) -> tuple[Approach, str]:
    """Look up an approach and validate the variant. Both arguments required."""
    if not variant:
        raise ValueError(
            f"variant is required for approach {name!r} — no default exists."
        )
    reg = load_approaches()
    if name not in reg:
        raise KeyError(
            f"Unknown approach: {name!r}. "
            f"Known: {sorted(reg.keys())}. "
            f"Add it to {APPROACHES_TOML.relative_to(REPO_ROOT)} first."
        )
    approach = reg[name]
    approach.require_variant(variant)
    return approach, variant


def parse_approach_spec(spec: str) -> tuple[str, str]:
    """Parse an approach spec into (name, variant). Both REQUIRED.

    "model_guided_v2@v2" -> ("model_guided_v2", "v2")

    Bare names (no `@`) are rejected — every reference must be explicit.
    """
    if "@" not in spec:
        raise ValueError(
            f"Approach spec {spec!r} is missing a variant. Use 'name@variant' "
            f"(e.g. 'model_guided_v2@v2'); no default exists."
        )
    name, variant = spec.split("@", 1)
    if not name or not variant:
        raise ValueError(f"Malformed approach spec: {spec!r}")
    return name, variant


def format_approach_spec(name: str, variant: str) -> str:
    """Canonical display form. Always includes @variant."""
    if not variant:
        raise ValueError(f"format_approach_spec: empty variant for approach {name!r}")
    return f"{name}@{variant}"


def resolve_approach_specs(specs: list[str]) -> list[tuple[str, str, str]]:
    """Turn matrix `approaches = [...]` entries into (name, variant, display).

    `display` is the canonical form used as a column/queue/bucket key, always
    `name@variant`. Raises:
      - ValueError on bare names (no `@`)
      - KeyError on unknown approach or variant
      - ValueError on duplicate (name, variant) pairs within `specs`
    """
    reg = load_approaches()
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for spec in specs:
        name, variant = parse_approach_spec(spec)
        if name not in reg:
            raise KeyError(
                f"Unknown approach {name!r}. Known: {sorted(reg.keys())}. "
                f"Add it to {APPROACHES_TOML.relative_to(REPO_ROOT)} first."
            )
        reg[name].require_variant(variant)
        key = (name, variant)
        if key in seen:
            raise ValueError(
                f"Duplicate approach spec {format_approach_spec(name, variant)!r} "
                f"in {specs!r}."
            )
        seen.add(key)
        out.append((name, variant, format_approach_spec(name, variant)))
    return out

# ---------- Legacy-dir parsing ----------

# Legacy dirs look like: agent_bench_docker_version_0_1_14_codex_gpt_5_4_reference
LEGACY_DIR_RE = re.compile(
    r"^agent_bench_docker_version_"
    r"(?P<version>\d+_\d+_\d+)_"
    r"(?P<agent>claude|codex)_"
    r"(?P<rest>.+)$"
)

# Models known to appear after the agent in legacy dir names.
KNOWN_MODELS = [
    "claude_haiku_4_5",
    "claude_sonnet_4_6",
    "gpt_5_4",
    "gpt_5_5",
]

def parse_legacy_dir_name(name: str) -> Optional[dict]:
    """Return {agent, model, source, source_inferred, version} from a legacy
    directory name, or None.

    `source_inferred=True` means the directory name omitted the source suffix
    and we defaulted to nl_to_nki — true for the two earliest 0.1.14 claude
    dirs, whose kernels (cumsum, layernorm, rms, sparsemax) are all in the
    nl_to_nki source set.
    """
    m = LEGACY_DIR_RE.match(name)
    if not m:
        return None
    rest = m.group("rest")
    for model in KNOWN_MODELS:
        if rest == model:
            return {
                "agent":           m.group("agent"),
                "model":           _normalize_model(model),
                "source":          "nl_to_nki",
                "source_inferred": True,
                "version":         m.group("version").replace("_", "."),
            }
        if rest.startswith(model + "_"):
            source = rest[len(model) + 1:]
            return {
                "agent":           m.group("agent"),
                "model":           _normalize_model(model),
                "source":          source,
                "source_inferred": False,
                "version":         m.group("version").replace("_", "."),
            }
    return None

def _normalize_model(model: str) -> str:
    """Convert directory-style model name to canonical form (with hyphens)."""
    # gpt_5_4 -> gpt-5.4, claude_sonnet_4_6 -> claude-sonnet-4-6
    if model.startswith("gpt_"):
        # gpt_5_4 -> gpt-5.4
        parts = model.split("_")
        return f"{parts[0]}-{parts[1]}.{parts[2]}"
    return model.replace("_", "-")

def derive_legacy_agent_features(version: str, agent: str) -> str:
    """Map (version, agent) → agent_features tag for the migration.

    Cutoff: 0.1.17 and later use the v2_opt38 cohort (per git log b065cac, 83af92d
    and bc29ce4 codex harness update). 0.1.14–0.1.16 use the v1_opt37 cohort.
    """
    parts = [int(p) for p in version.split(".")]
    is_v2 = (parts[0], parts[1], parts[2]) >= (0, 1, 17)
    if agent == "codex":
        return "codex_v2_opt38" if is_v2 else "codex_v1_opt37"
    if agent == "claude":
        return "claude_v2_opt38" if is_v2 else "claude_v1_opt37"
    raise ValueError(f"Unknown agent: {agent}")

# ---------- metadata.json schema ----------

@dataclass
class Measurement:
    method: str                                  # tag from measurement_methods.toml
    measured_at: str                             # ISO timestamp
    baseline_ms: Optional[float] = None
    optimized_ms: Optional[float] = None
    speedup: Optional[float] = None
    per_case_baseline_ms: Optional[list[float]] = None
    per_case_optimized_ms: Optional[list[float]] = None
    per_case_speedups: Optional[list[float]] = None
    profile_result_file: Optional[str] = None    # filename within run dir
    profiler_version: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class RunMetadata:
    run_id: str
    source: str
    kernel: str
    approach: str
    approach_variant: str
    model: str
    agent_features: str
    replicate: int
    timestamp: str
    package_version: Optional[str] = None
    docker_image: Optional[str] = None
    reasoning_effort: Optional[str] = None
    cheat: bool = False
    cheat_reason: Optional[str] = None
    measurements: list[Measurement] = field(default_factory=list)
    original_path: Optional[str] = None  # legacy path, preserved for audit

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_path(cls, run_dir: Path) -> "RunMetadata":
        data = json.loads((run_dir / "metadata.json").read_text())
        ms = [Measurement(**m) for m in data.pop("measurements", [])]
        return cls(measurements=ms, **data)

    def get_measurement(self, method: str) -> Optional[Measurement]:
        for m in self.measurements:
            if m.method == method:
                return m
        return None

    def upsert_measurement(self, m: Measurement) -> None:
        for i, existing in enumerate(self.measurements):
            if existing.method == m.method:
                self.measurements[i] = m
                return
        self.measurements.append(m)


# ---------- Run-id generation ----------

def make_run_id(
    timestamp: str,
    source: str,
    kernel: str,
    approach: str,
    agent: str,
    model: str,
    *,
    approach_variant: str,
) -> str:
    """Stable, human-skimmable run id. The approach segment is always
    `<approach>@<variant>` — variant is required and never elided.
    """
    if not approach_variant:
        raise ValueError(
            f"make_run_id: approach_variant is required for approach {approach!r}"
        )
    ts_compact = timestamp.replace("-", "").replace(":", "").replace("T", "_").rstrip("Z")[:15]
    model_short = model.replace("-", "").replace(".", "")
    approach_seg = format_approach_spec(approach, approach_variant)
    return f"{ts_compact}__{source}__{kernel}__{approach_seg}__{agent}_{model_short}"


# Run-id parser. The approach segment is always `<name>@<variant>`.
RUN_ID_RE = re.compile(
    r"^(?P<ts>\d{8}_\d{6})__"
    r"(?P<source>[^_]+(?:_[^_]+)*?)__"
    r"(?P<kernel>[^_]+(?:_[^_]+)*?)__"
    r"(?P<approach>[^_@]+(?:_[^_@]+)*?)@(?P<variant>[^_]+)__"
    r"(?P<agent>[^_]+)_(?P<model_short>.+?)(?:__rep(?P<replicate>\d+))?$"
)


def parse_run_id(run_id: str) -> Optional[dict]:
    """Inverse of make_run_id. Returns None if the format doesn't match.

    Note: source and kernel can contain underscores ("nl_to_nki"), so the regex
    is best-effort. Authoritative source is always the metadata.json fields;
    parse_run_id is for tooling that wants to slice on directory names without
    opening every file.
    """
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None
    return {
        "timestamp":    m.group("ts"),
        "source":       m.group("source"),
        "kernel":       m.group("kernel"),
        "approach":     m.group("approach"),
        "approach_variant": m.group("variant"),
        "agent":        m.group("agent"),
        "model_short":  m.group("model_short"),
        "replicate":    int(m.group("replicate")) if m.group("replicate") else None,
    }
