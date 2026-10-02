"""Parse a Convention-B ("inline-shape") oracle's run_test into a structured recipe.

Convention-A oracles factor inputs into `_make_inputs(<knobs>, seed)`; grounding
and execution read that signature directly. Convention-B oracles (reference/*,
nl_to_nki/*, …) instead bake shapes into a literal list inside `run_test` and
construct inputs inline:

    shapes = [(64, 64, 64), (128, 128, 128), ...]      # or inline in the `for`
    for m, n, k in shapes:
        a = torch.randn((m, k), dtype=torch.float16)
        b = torch.randn((k, n), dtype=torch.float16)
        expected = _ref_matmul(a, b)
        actual = helper_func(a, b)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)

This module extracts, from the first differential loop, a `Recipe`:
  - knobs: loop variables that appear as a tensor dimension (filters structural
    loop vars like cumsum's `ndim`/`seq_len` that aren't bare tensor axes);
  - tensors: each constructed input as (name, per-axis knob/int, dtype, ctor, fill);
  - helper_args / ref_name / ref_args: call order, so execution can rebuild the calls;
  - scalars (e.g. eps), seed_shapes, and the assert_close atol/rtol.

The recipe lets grounding map oracle knobs to kernel tiled dims by tensor-arg +
axis position (a structural join, correct under transposes), and lets the
before/after oracle build inputs for a *generated* shape. Pure stdlib AST over
source text — no torch, no device.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

_TORCH_CTORS = {"randn", "rand", "ones", "zeros", "full", "empty", "randint"}


@dataclass
class TensorArg:
    name: str
    axes: list                 # per axis: knob name (str) or int literal
    dtype: str | None          # e.g. "float16", "float32"
    ctor: str                  # "randn" | "rand" | "full" | ...
    fill: float | None = None  # for torch.full((..), fill)


@dataclass
class Recipe:
    knobs: list[str]
    seed_shapes: list[tuple]
    tensors: list[TensorArg]               # in construction order
    helper_args: list[str]                 # names passed to the helper, in order
    ref_name: str | None = None
    ref_args: list[str] = field(default_factory=list)
    scalars: dict = field(default_factory=dict)   # name -> python value
    atol: float | None = None
    rtol: float | None = None

    @property
    def tensor_by_name(self) -> dict:
        return {t.name: t for t in self.tensors}


# ---------------------------------------------------------------- helpers

def _axis_elt(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _shape_axes(node):
    """A shape argument: a tuple/list literal -> list of axis elts, or None."""
    if isinstance(node, (ast.Tuple, ast.List)):
        elts = [_axis_elt(e) for e in node.elts]
        return None if any(e is None for e in elts) else elts
    return None


def _ctor_of(call: ast.Call):
    """('randn', shape_node, fill_node|None) if call is a recognised tensor ctor."""
    f = call.func
    name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
    if name == "tensor":          # torch.tensor(np.random...(shape)) — shape not literal here
        return None
    if name in _TORCH_CTORS and call.args:
        fill = call.args[1] if (name == "full" and len(call.args) > 1) else None
        return name, call.args[0], fill
    return None


def _dtype_of(call: ast.Call):
    for kw in call.keywords:
        if kw.arg == "dtype":
            v = kw.value
            if isinstance(v, ast.Attribute):     # torch.float16
                return v.attr
    return None


def _const(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def _call_arg_names(call: ast.Call):
    """Positional arg identifiers of a call (None for non-Name args)."""
    return [a.id if isinstance(a, ast.Name) else None for a in call.args]


# ---------------------------------------------------------------- main

def parse_recipe(oracle_src: str) -> Recipe | None:
    try:
        tree = ast.parse(oracle_src)
    except SyntaxError:
        return None
    run_test = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "run_test"), None)
    if run_test is None:
        return None
    helper_param = run_test.args.args[0].arg if run_test.args.args else None

    # module/function-level simple constant assignments (e.g. shapes=[...], eps=1e-5)
    list_vars: dict[str, list] = {}
    scalars: dict[str, object] = {}
    for n in run_test.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if isinstance(n.value, (ast.List, ast.Tuple)):
                tuples = []
                for e in n.value.elts:
                    if isinstance(e, ast.Tuple):
                        vals = [_const(x) for x in e.elts]
                        if all(v is not None for v in vals):
                            tuples.append(tuple(vals))
                if tuples:
                    list_vars[name] = tuples
            else:
                c = _const(n.value)
                if c is not None:
                    scalars[name] = c

    # first differential for-loop: targets are dims, iter is a tuple-list (literal or var)
    loop = None
    for n in run_test.body:
        if isinstance(n, ast.For):
            seeds = None
            if isinstance(n.iter, (ast.List, ast.Tuple)):
                seeds = _shape_list(n.iter)
            elif isinstance(n.iter, ast.Name):
                seeds = list_vars.get(n.iter.id)
            if seeds:
                loop = (n, seeds)
                break
    if loop is None:
        return None
    for_node, seed_shapes = loop
    loop_vars = ([t.id for t in for_node.target.elts]
                 if isinstance(for_node.target, ast.Tuple) else [for_node.target.id])

    # tensor constructions inside the loop body
    tensors: list[TensorArg] = []
    helper_args: list[str] = []
    ref_name = None
    ref_args: list[str] = []
    atol = rtol = None
    for stmt in ast.walk(for_node):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call):
            tgt = stmt.targets[0].id
            ctor = _ctor_of(stmt.value)
            if ctor:
                name, shape_node, fill_node = ctor
                axes = _shape_axes(shape_node)
                if axes is not None:
                    tensors.append(TensorArg(
                        name=tgt, axes=axes, dtype=_dtype_of(stmt.value),
                        ctor=name, fill=_const(fill_node) if fill_node else None))
                continue
            # expected = _ref_xxx(...)   /   actual = helper(...)
            f = stmt.value.func
            fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if fname == helper_param:
                helper_args = _call_arg_names(stmt.value)
            elif fname and fname.startswith("_ref"):
                ref_name, ref_args = fname, _call_arg_names(stmt.value)
        elif isinstance(stmt, ast.Call):
            f = stmt.func
            if isinstance(f, ast.Attribute) and f.attr == "assert_close":
                for kw in stmt.keywords:
                    if kw.arg == "atol":
                        atol = _const(kw.value)
                    elif kw.arg == "rtol":
                        rtol = _const(kw.value)

    if not tensors:
        return None
    tensor_names = {t.name for t in tensors}
    used_axis_knobs = {a for t in tensors for a in t.axes if isinstance(a, str)}
    knobs = [v for v in loop_vars if v in used_axis_knobs]
    # reorder seed_shapes columns to knob order (drop columns for non-knob loop vars)
    keep_idx = [i for i, v in enumerate(loop_vars) if v in knobs]
    seeds = [tuple(s[i] for i in keep_idx) for s in seed_shapes if len(s) == len(loop_vars)]

    return Recipe(
        knobs=knobs, seed_shapes=seeds, tensors=tensors,
        helper_args=helper_args, ref_name=ref_name, ref_args=ref_args,
        scalars={k: v for k, v in scalars.items() if k not in knobs},
        atol=atol, rtol=rtol,
    )


def assert_close_tol(oracle_src: str) -> tuple:
    """(atol, rtol) from the first `*.assert_close(..., atol=, rtol=)` in run_test,
    i.e. the tolerance the kernel's OWN oracle uses to judge correctness vs the
    reference. (None, None) if absent. Works for both oracle conventions."""
    try:
        tree = ast.parse(oracle_src)
    except SyntaxError:
        return (None, None)
    run_test = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "run_test"), None)
    if run_test is None:
        return (None, None)
    for n in ast.walk(run_test):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "assert_close":
            atol = rtol = None
            for kw in n.keywords:
                if kw.arg == "atol":
                    atol = _const(kw.value)
                elif kw.arg == "rtol":
                    rtol = _const(kw.value)
            if atol is not None or rtol is not None:
                return (atol, rtol)
    return (None, None)


def _shape_list(node):
    out = []
    for e in node.elts:
        if isinstance(e, ast.Tuple):
            vals = [_const(x) for x in e.elts]
            if all(v is not None for v in vals):
                out.append(tuple(vals))
    return out
