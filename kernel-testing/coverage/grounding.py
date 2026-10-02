"""Automated, source-grounded behavior-class partition for a carrier.

Carriers (the benchmark kernels an optimization was applied to) grow as new
optimization runs land, so the behavior-class partition cannot be hand-mapped
per kernel. This module derives it automatically from each carrier's *source*:

  - the oracle's `_make_inputs(<knobs>, seed)` signature gives the shape knobs
    the generator controls (M, N, K, ...);
  - the kernel's tiling arithmetic (`knob // TILE`, `knob % TILE`) gives, per
    knob, the tile divisor(s) that partition that dimension's behavior.

The grounding is the *divisibility partition*: for each tiled knob and divisor D
the three behaviorally-distinct regimes are

    exact     knob % D == 0          whole tiles, inside the kernel's contract
    partial   knob > D and knob%D!=0  trailing remainder dropped      [critical]
    below     0 < knob < D           zero full tiles -> no work done  [critical]

(the adamw N//N_TILE=2048 carrier: N=2048 exact/correct, N=2049 partial/wrong,
N=1024 below/all-zeros -- see findings/equivalence-necessary-not-sufficient.md).

This is grounded in code STRUCTURE, not in observed outcomes, so covering the
partition is a non-circular adequacy claim. What it cannot map -- oracles that
don't use the `_make_inputs(<knobs>)` convention, knob/dim name mismatches,
derived dims -- is reported as `ungrounded` with a reason, never silently binned.

Pure stdlib (ast/re) over source text -- no torch, no device. Offline-testable.
"""
from __future__ import annotations

import ast
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import oracle_recipe as orc  # noqa: E402


# ---------------------------------------------------------------- oracle knobs

def oracle_knobs(oracle_src: str) -> list[str] | None:
    """Positional `_make_inputs` params (minus seed). None if the oracle has no
    such function or uses the single structured `shapes` arg (compiler_gen)."""
    try:
        tree = ast.parse(oracle_src)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "_make_inputs":
            names = [a.arg for a in n.args.args if a.arg != "seed"]
            if names == ["shapes"]:
                return None
            return names
    return None


# ---------------------------------------------------------------- tiling facts

def _eval_int(node: ast.AST, consts: dict[str, int]) -> int | None:
    """Evaluate an int-valued constant expression: literals, named constants, and
    arithmetic over them (`TILE_N * NUM_CORES`, `TILE_M * 4`). None if not
    statically an int."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, int) and not isinstance(node.value, bool) else None
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.BinOp):
        a, b = _eval_int(node.left, consts), _eval_int(node.right, consts)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, (ast.FloorDiv, ast.Div)) and b:
            return a // b
    return None


def _int_consts(tree: ast.AST) -> dict[str, int]:
    """name -> int for `NAME = <expr>` assignments anywhere in the module, where
    <expr> evaluates to an int over already-known constants. Fixpointed so
    computed constants resolve (`BLOCK_M = TILE_M * TILES_IN_BLOCK_M`)."""
    assigns: list[tuple[str, ast.AST]] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name):
                    assigns.append((tgt.id, n.value))
    consts: dict[str, int] = {}
    for _ in range(len(assigns) + 1):
        changed = False
        for name, val in assigns:
            if name in consts:
                continue
            r = _eval_int(val, consts)
            if r is not None:
                consts[name] = r
                changed = True
        if not changed:
            break
    return consts


# names treated as ceil-division helpers (`div_ceil(n, d) = (n + d - 1) // d`)
_CEIL_HELPER_RE = ("ceil",)


def _ceil_helper_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and any(k in n.name.lower() for k in _CEIL_HELPER_RE):
            out.add(n.name)
    return out


def _knob_in(node: ast.AST, knobset: set[str]) -> str | None:
    """The knob Name appearing anywhere in an expression (numerator), or None."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in knobset:
            return sub.id
    return None


def tiling_facts(kernel_src: str, knobs: list[str]) -> dict[str, dict[int, str]]:
    """knob -> {divisor: style}, style in {"drop", "pad"}.

    Detects every way the kernel tiles a knob dimension:
      bare  `knob // C`, `knob % C`                  -> "drop" (remainder dropped)
      ceil  `(knob + C - 1) // C`, `(knob + ..) // C`-> "pad"  (remainder handled)
      ceil  `math.ceil(knob / C)`                    -> "pad"
      ceil  `<ceil-helper>(knob, C)`                 -> "pad"
    C may be a named/computed constant or a product. drop wins over pad on merge
    (the hazard exists in at least one form). See findings/tiling-styles-and-
    criticality.md."""
    out: dict[str, dict[int, str]] = {}

    def add(knob: str, div: int | None, style: str):
        if knob and div and div > 1:
            cur = out.setdefault(knob, {})
            if cur.get(div) != "drop":           # drop is sticky
                cur[div] = style if cur.get(div) is None else ("drop" if "drop" in (cur[div], style) else "pad")

    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return out
    consts = _int_consts(tree)
    knobset = set(knobs)
    ceil_helpers = _ceil_helper_names(tree)

    for n in ast.walk(tree):
        # floor-division / modulo
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.FloorDiv, ast.Mod)):
            div = _eval_int(n.right, consts)
            if isinstance(n.left, ast.Name) and n.left.id in knobset:
                add(n.left.id, div, "drop")                 # bare knob // C
            else:
                knob = _knob_in(n.left, knobset)             # (knob + C - 1) // C
                if knob is not None:
                    add(knob, div, "pad")
        # math.ceil(knob / C)  ->  Call(ceil, [BinOp(knob / C)])
        elif isinstance(n, ast.Call):
            fname = n.func.attr if isinstance(n.func, ast.Attribute) else (
                n.func.id if isinstance(n.func, ast.Name) else "")
            if fname == "ceil" and n.args:
                arg = n.args[0]
                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div):
                    knob = _knob_in(arg.left, knobset)
                    add(knob, _eval_int(arg.right, consts), "pad")
            elif fname in ceil_helpers and len(n.args) >= 2:   # div_ceil(knob, C)
                knob = _knob_in(n.args[0], knobset)
                add(knob, _eval_int(n.args[1], consts), "pad")
    return out


def tiling_divisors(kernel_src: str, knobs: list[str]) -> dict[str, set[int]]:
    """knob -> {tile divisors} (style-agnostic view of tiling_facts)."""
    return {k: set(v) for k, v in tiling_facts(kernel_src, knobs).items()}


# ---------------------------------------------------------------- structural join (Convention B)

def _root_name(node: ast.AST) -> str | None:
    """Leftmost Name of a `.attr` / call chain: `x.reshape(..).contiguous()` -> 'x'."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            return None


# ops that reorder axes — an alias through these does NOT preserve logical axis order
_AXIS_REORDER = {"T", "mT", "transpose", "permute", "swapaxes", "moveaxis"}


def _reorders_axes(node: ast.AST) -> bool:
    """True if the expression contains a transpose-like op anywhere in its chain."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _AXIS_REORDER:
            return True
    return False


def _param_shape_axes(fn: ast.FunctionDef, params: set[str]) -> dict[str, list]:
    """param -> [dim names] from `<names> = <param-or-alias>.shape`. A direct
    `param.shape` unpack (logical axis order) is authoritative; aliases are
    followed one level for reshape/cast (`x_arg = x.reshape(...)`) but NOT through
    transpose-like ops (`lhsT = a.T`), which would invert the axis order."""
    alias: dict[str, str] = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if _reorders_axes(n.value):
                continue
            root = _root_name(n.value)
            if root in params or root in alias:
                alias[n.targets[0].id] = alias.get(root, root)

    direct: dict[str, list] = {}
    aliased: dict[str, list] = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple) \
                and isinstance(n.value, ast.Attribute) and n.value.attr == "shape":
            base = n.value.value
            if not isinstance(base, ast.Name):
                continue
            names = [e.id if isinstance(e, ast.Name) else None for e in n.targets[0].elts]
            if base.id in params:
                direct.setdefault(base.id, names)          # first direct unpack wins
            elif base.id in alias:
                aliased.setdefault(alias[base.id], names)
    return {**aliased, **direct}                            # direct overrides aliased


def kernel_param_axes(kernel_src: str, n_params: int) -> list | None:
    """Per positional param of the kernel function the oracle calls, the list of
    dimension names. Picks the LAST defined function with `.shape` unpacks and
    matching arity (the wrapper — its unpack is in logical order)."""
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    arity_match, any_unpack = [], []
    for fn in funcs:
        params = [a.arg for a in fn.args.args]
        axesmap = _param_shape_axes(fn, set(params))
        if not axesmap:
            continue
        any_unpack.append((params, axesmap))
        required = len(params) - len(fn.args.defaults)
        if required <= n_params <= len(params):     # callable with n positional args
            arity_match.append((params, axesmap))
    chosen = (arity_match or any_unpack)
    if not chosen:
        return None
    params, axesmap = chosen[-1]                     # last defined = wrapper
    return [axesmap.get(p) for p in params[:n_params]]


def _merge_style(facts: dict, src: str, dims: list[str]) -> None:
    for dim, divs in tiling_facts(src, dims).items():
        f = facts.setdefault(dim, {})
        for D, style in divs.items():
            f[D] = "drop" if "drop" in (f.get(D), style) else "pad"


def _structural_join(recipe, base_src: str, opt_src: str) -> dict[str, dict[int, str]] | None:
    """Map oracle knobs -> {divisor: style} by joining the recipe's per-tensor
    axes to the kernel's per-param axes by (arg position, axis position), then
    looking up each kernel dim's tiling. Correct under transposes."""
    n = len(recipe.helper_args)
    axes = kernel_param_axes(base_src, n) or kernel_param_axes(opt_src, n)
    if not axes:
        return None
    dimnames = sorted({d for a in axes if a for d in a if isinstance(d, str) and d != "_"})
    facts: dict[str, dict[int, str]] = {}
    _merge_style(facts, base_src, dimnames)
    _merge_style(facts, opt_src, dimnames)
    tname = recipe.tensor_by_name
    merged: dict[str, dict[int, str]] = {}
    for i, argname in enumerate(recipe.helper_args):
        if i >= len(axes) or axes[i] is None:
            continue
        t = tname.get(argname)
        if t is None:                       # scalar arg (e.g. eps)
            continue
        for axis_j, kdim in enumerate(axes[i]):
            if axis_j >= len(t.axes):
                continue
            knob = t.axes[axis_j]
            if isinstance(knob, str) and isinstance(kdim, str) and kdim in facts:
                m = merged.setdefault(knob, {})
                for D, style in facts[kdim].items():
                    m[D] = "drop" if "drop" in (m.get(D), style) else "pad"
    return merged


# ---------------------------------------------------------------- partition

@dataclass
class BehaviorClass:
    name: str
    knob: str
    divisor: int
    predicate: str          # python expr over the knob
    critical: bool


def _divisibility_classes(knob: str, divisor: int, style: str) -> list[BehaviorClass]:
    """Three regimes per (knob, divisor). The non-divisible regimes are critical
    only for `drop` tiling (bare floordiv silently drops the remainder); `pad`
    (ceil-div) handles it, so they are real classes but not bug-likely."""
    D = divisor
    crit = style == "drop"
    return [
        BehaviorClass(f"{knob}_x{D}_exact", knob, D,
                      f"{knob} % {D} == 0", critical=False),
        BehaviorClass(f"{knob}_x{D}_partial", knob, D,
                      f"{knob} > {D} and {knob} % {D} != 0", critical=crit),
        BehaviorClass(f"{knob}_x{D}_below", knob, D,
                      f"0 < {knob} < {D}", critical=crit),
    ]


@dataclass
class CarrierGrounding:
    source: str
    kernel: str
    knobs: list[str] | None = None
    tile_divisors: dict[str, list[int]] = field(default_factory=dict)
    classes: list[BehaviorClass] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return bool(self.classes)

    def summary(self) -> str:
        head = f"{self.source}/{self.kernel}"
        if not self.grounded:
            return f"{head}: UNGROUNDED ({'; '.join(self.ungrounded) or 'no tiled knobs'})"
        td = ", ".join(f"{k}//{sorted(v)}" for k, v in self.tile_divisors.items())
        crit = sum(c.critical for c in self.classes)
        line = f"{head}: knobs={self.knobs} tiled[{td}] -> {len(self.classes)} classes ({crit} critical)"
        if self.ungrounded:
            line += f"  | partial: {'; '.join(self.ungrounded)}"
        return line


def ground_sources(source: str, kernel: str, oracle_src: str,
                   baseline_src: str, optimized_src: str) -> CarrierGrounding:
    """Ground a carrier from raw source strings (the offline-testable core)."""
    knobs = oracle_knobs(oracle_src)
    g = CarrierGrounding(source=source, kernel=kernel, knobs=knobs)
    if knobs is not None:
        # Convention A: oracle exposes _make_inputs(<knobs>); join by name.
        merged = {}
        for facts in (tiling_facts(baseline_src, knobs), tiling_facts(optimized_src, knobs)):
            for knob, divs in facts.items():
                m = merged.setdefault(knob, {})
                for D, style in divs.items():
                    m[D] = "drop" if "drop" in (m.get(D), style) else "pad"
    else:
        # Convention B: inline-shape oracle; structural (arg, axis) join via recipe.
        recipe = orc.parse_recipe(oracle_src)
        if recipe is None:
            g.ungrounded.append("oracle has neither _make_inputs(<knobs>) nor a "
                                "parseable run_test recipe")
            return g
        g.knobs = recipe.knobs
        merged = _structural_join(recipe, baseline_src, optimized_src)
        if merged is None:
            g.ungrounded.append("could not map recipe knobs to kernel dims "
                                "(no param .shape unpack)")
            return g
    if not merged:
        g.ungrounded.append("no tiling found over oracle knobs "
                            "(derived dims or non-divisibility tiling)")
        return g
    g.tile_divisors = {k: sorted(v) for k, v in sorted(merged.items())}
    for knob in sorted(merged):
        for D in sorted(merged[knob]):
            g.classes.extend(_divisibility_classes(knob, D, merged[knob][D]))
    return g


def ground_carrier(carrier) -> CarrierGrounding:
    """Ground a run_index.RunPair by reading its source files."""
    return ground_sources(
        carrier.source, carrier.kernel,
        carrier.oracle_path.read_text(),
        carrier.baseline_path.read_text(),
        carrier.optimized_path.read_text(),
    )


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import run_index as ri

    rep = ri.index_runs()
    seen: dict[tuple, object] = {}
    for opt in rep.by_opt:
        for c in rep.by_opt[opt]:
            seen.setdefault((c.source, c.kernel), c)
    grounded = ungrounded = 0
    for key in sorted(seen):
        g = ground_carrier(seen[key])
        print(g.summary())
        grounded += g.grounded
        ungrounded += not g.grounded
    print(f"\n{grounded} grounded, {ungrounded} ungrounded "
          f"of {len(seen)} distinct carriers")
