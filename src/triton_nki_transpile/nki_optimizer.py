"""
NKI kernel optimizer: plan → implement → test loop.

Supports two menu strategies for the planning phase:
  - AUTOCOMP: AutoComp's vague high-level optimization menu (~50 items).
  - PRE_EXPLORED: Our concretized menu from nki-optimizations/ with benchmark-verified
    speedups, concrete applicability signals, and "do NOT apply" compiler-handled items.
"""
import logging
import re
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from triton_nki_transpile.agent import Agent
from triton_nki_transpile.config import Config, get_config
from triton_nki_transpile.logger import trace_logger
from triton_nki_transpile.opt_profiler import NKIOptimizationProfiler
from triton_nki_transpile.prompts import (
    get_optimizer_plan_instructions,
    get_optimizer_implement_instructions,
)

logger = logging.getLogger(__name__)

# Path resolution for resource files
_PACKAGE_DIR = Path(__file__).parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent  # src/triton_nki_transpile -> src -> repo root
_AUTOCOMP_TOML_PATH = _PACKAGE_DIR / "resources" / "opt-menus" / "autocomp-menu.toml"
_NKI_OPT_DIR = _REPO_ROOT / "nki-optimizations"
_PRE_EXPLORED_TOML_PATH = _NKI_OPT_DIR / "optimization-quick-reference.toml"


class MenuMode(str, Enum):
    AUTOCOMP = "autocomp"
    PRE_EXPLORED = "pre-explored"


@dataclass
class OptimizationItem:
    """A single optimization option, loaded once at startup.

    Attributes:
        number:     Integer optimization number for PRE_EXPLORED items (e.g. 30),
                    or None for AUTOCOMP items which have no number.
        name:       Human-readable name (e.g. "Fuse Reduction + Transform").
                    Empty string for AUTOCOMP items.
        speedup:    Benchmark speedup string (e.g. "1.10×"), or None for AUTOCOMP.
        apply_when: Applicability criteria string, or None for AUTOCOMP.
        skip_when:  Counter-criteria string, or None for AUTOCOMP.
        raw:        Verbatim source text used for AUTOCOMP rendering.
                    Empty string for PRE_EXPLORED items (row is reconstructed from fields).
    """
    number: Optional[int]
    name: str
    speedup: Optional[str]
    apply_when: Optional[str]
    skip_when: Optional[str]
    raw: str


@dataclass
class PreExploredMenu:
    """Loaded representation of optimization-quick-reference.toml.

    The preamble and postamble are static markdown sections always shown to the LLM.
    Items are filtered between iterations; the table is reconstructed from their fields.
    """
    preamble: str
    items: List[OptimizationItem]
    postamble: str


def _load_pre_explored_menu() -> PreExploredMenu:
    """Load optimization-quick-reference.toml into a PreExploredMenu."""
    with open(_PRE_EXPLORED_TOML_PATH, "rb") as f:
        data = tomllib.load(f)
    items = [
        OptimizationItem(
            number=item["number"],
            name=item["name"],
            speedup=item["speedup"],
            apply_when=item["apply_when"],
            skip_when=item["skip_when"],
            raw="",
        )
        for item in data["items"]
    ]
    return PreExploredMenu(
        preamble=data["preamble"],
        items=items,
        postamble=data["postamble"],
    )


def _render_pre_explored_menu(menu: PreExploredMenu, items: List[OptimizationItem]) -> str:
    """Reconstruct the full markdown table with only the remaining optimization rows."""
    rows = [
        f"| **{item.number:02d}** | {item.name} | {item.speedup} | {item.apply_when} | {item.skip_when} |"
        for item in items
    ]
    return "\n".join([menu.preamble, *rows, menu.postamble])


def _load_autocomp_items() -> List[OptimizationItem]:
    """Load autocomp-menu.toml, one OptimizationItem per entry."""
    with open(_AUTOCOMP_TOML_PATH, "rb") as f:
        data = tomllib.load(f)
    return [
        OptimizationItem(number=None, name="", speedup=None, apply_when=None, skip_when=None, raw=text)
        for text in data["items"]
    ]


def _render_autocomp_menu(items: List[OptimizationItem]) -> str:
    numbered = "\n".join(f"{i+1}. {item.raw}" for i, item in enumerate(items))
    return f"<optimizations>:\n{numbered}\n"


def _extract_selected_optimization(plan: str) -> Optional[str]:
    """Extract the optimization name from the structured SELECTED_OPTIMIZATION line."""
    match = re.search(r"SELECTED_OPTIMIZATION:\s*(.+)", plan)
    if match:
        return match.group(1).strip()
    return None


def _extract_selected_optimizations(plan: str) -> List[str]:
    """Extract all optimization names from a multi-opt SELECTED_OPTIMIZATIONS block.

    Parses a block of the form:
        SELECTED_OPTIMIZATIONS:
        - #NN · Name
        - #NN · Name
    Returns a list of stripped item strings, e.g. ["#03 · Full Partition Utilization", ...].
    """
    match = re.search(r"SELECTED_OPTIMIZATIONS:\s*\n((?:\s*-\s*.+\n?)+)", plan)
    if match:
        return [item.strip() for item in re.findall(r"-\s*(.+)", match.group(1))]
    return []


def _extract_opt_number(selected: str) -> Optional[int]:
    """Extract the optimization number from an LLM response string.

    Used exclusively to parse the LLM's SELECTED_OPTIMIZATION line — never
    called on menu item strings (items have a pre-parsed .number field).

    Handles formats like: "#30", "30", "**30**", "Optimization 30", etc.
    """
    match = re.search(r"\b(\d+)\b", selected)
    return int(match.group(1)) if match else None


def _find_selected_item(
    selected: str,
    items: List[OptimizationItem],
) -> Optional[OptimizationItem]:
    """Match an LLM SELECTED_OPTIMIZATION string to one item in the remaining menu.

    Strategy (in order of specificity):
    1. Parse a number from `selected`; find the item with that .number (PRE_EXPLORED).
    2. Fallback: case-insensitive substring match against item.name or item.raw (AUTOCOMP).
    """
    opt_number = _extract_opt_number(selected)
    if opt_number is not None:
        for item in items:
            if item.number == opt_number:
                return item

    # Fallback: substring match on name (PRE_EXPLORED) or raw text (AUTOCOMP)
    selected_lower = selected.lower()
    for item in items:
        text = (item.name if item.name else item.raw).lower()
        if selected_lower in text or text in selected_lower:
            return item

    return None


def _load_opt_readme(opt_number: int) -> Optional[str]:
    """Load the README for the given optimization number (e.g. 30)."""
    matches = list(_NKI_OPT_DIR.glob(f"{opt_number:02d}-*"))
    if not matches:
        return None
    readme = matches[0] / "README.md"
    return readme.read_text() if readme.exists() else None


def _extract_code(response: str) -> str:
    """Extract Python code from a markdown code block in the LLM response."""
    if "```python" in response:
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "```" in response:
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response.strip()


class NKIOptimizer:
    """
    Single-threaded plan → implement → test optimizer for NKI kernels.

    The planning phase shows the LLM either:
    - AutoComp's vague high-level menu (MenuMode.AUTOCOMP)
    - Our pre-explored concretized menu with speedups and applicability criteria (MenuMode.PRE_EXPLORED)

    The implement phase is identical for both — the LLM writes the optimized kernel.
    The test phase uses neuron-profile for accurate hardware latency measurement.
    """

    def __init__(
        self,
        menu_mode: MenuMode = MenuMode.AUTOCOMP,
        config: Optional[Config] = None,
        multi_opt: bool = False,
    ):
        self.menu_mode = menu_mode
        self.multi_opt = multi_opt
        self.config = config or get_config()
        self.profiler = NKIOptimizationProfiler()

        if menu_mode == MenuMode.PRE_EXPLORED:
            self._pre_menu = _load_pre_explored_menu()
            self.menu_items: List[OptimizationItem] = list(self._pre_menu.items)
        else:
            self._pre_menu = None
            self.menu_items = _load_autocomp_items()

        plan_instructions = get_optimizer_plan_instructions(menu_mode.value, multi_opt=multi_opt)
        implement_instructions = get_optimizer_implement_instructions()

        # Two separate agents with separate conversation history
        self.plan_agent = Agent(instructions=plan_instructions, config=self.config)
        self.impl_agent = Agent(instructions=implement_instructions, config=self.config)

    def optimize(
        self,
        kernel_code: str,
        test_code: str,
        wrapper_name: str,
        max_iters: int = 3,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run the plan → implement → test loop for up to max_iters iterations.

        Returns:
            (best_kernel_code, history)
            history is a list of dicts with keys:
              iteration, plan, code, profile_result, accepted
        """
        history: List[Dict[str, Any]] = []

        # Profile baseline
        logger.info("Profiling baseline kernel...")
        trace_logger.log_step("opt_baseline_start", "Profiling baseline kernel")
        baseline_result = self.profiler.profile(kernel_code, test_code, wrapper_name)
        trace_logger.log_artifact("opt_baseline_kernel.py", kernel_code)
        trace_logger.log_event("opt_baseline_profile", baseline_result)

        if not baseline_result["success"]:
            logger.error(f"Baseline profiling failed: {baseline_result.get('error')}")
            return kernel_code, history

        baseline_latency = baseline_result["latency_ms"]
        if baseline_latency is None:
            logger.error("Baseline profiling succeeded but returned no latency measurement")
            return kernel_code, history
        logger.info(f"Baseline latency: {baseline_latency} ms")

        current_code = kernel_code
        current_latency = baseline_latency
        remaining_items: List[OptimizationItem] = list(self.menu_items)
        used_optimizations: List[str] = []

        for i in range(max_iters):
            iter_num = i + 1
            logger.info(f"=== Optimization iteration {iter_num}/{max_iters} ===")
            trace_logger.log_step(f"opt_iter_{iter_num}_start", f"Starting iteration {iter_num}")

            if not remaining_items:
                logger.info(f"[{iter_num}] All menu optimizations have been tried. Stopping.")
                break

            # ── Phase 1: Plan ──────────────────────────────────────────────
            logger.info(f"[{iter_num}] Planning optimization ({len(remaining_items)} options remaining)...")
            plan = self._plan(current_code, current_latency, remaining_items)
            trace_logger.log_artifact(f"opt_iter_{iter_num}_plan.txt", plan)

            opt_readme: Optional[str] = None
            if self.multi_opt:
                # Multi-opt: extract all selected optimizations, match each, collect READMEs
                selected_list = _extract_selected_optimizations(plan)
                readme_parts: List[str] = []
                for selected in selected_list:
                    logger.info(f"[{iter_num}] Selected optimization: {selected}")
                    used_optimizations.append(selected)
                    matched_item = _find_selected_item(selected, remaining_items)
                    if matched_item is not None:
                        remaining_items = [item for item in remaining_items if item is not matched_item]
                        if matched_item.number is not None:
                            readme = _load_opt_readme(matched_item.number)
                            if readme:
                                logger.info(f"[{iter_num}] Loaded README for optimization #{matched_item.number}")
                                readme_parts.append(f"## Optimization #{matched_item.number:02d}\n\n{readme}")
                            else:
                                logger.warning(f"[{iter_num}] No README found for optimization #{matched_item.number}")
                    else:
                        logger.warning(f"[{iter_num}] Could not match '{selected}' to any remaining menu item")
                if readme_parts:
                    opt_readme = "\n\n---\n\n".join(readme_parts)
            else:
                # Single-opt: extract one selected optimization
                selected = _extract_selected_optimization(plan)
                if selected:
                    logger.info(f"[{iter_num}] Selected optimization: {selected}")
                    used_optimizations.append(selected)
                    matched_item = _find_selected_item(selected, remaining_items)
                    if matched_item is not None:
                        remaining_items = [item for item in remaining_items if item is not matched_item]
                        if matched_item.number is not None:
                            opt_readme = _load_opt_readme(matched_item.number)
                            if opt_readme:
                                logger.info(f"[{iter_num}] Loaded README for optimization #{matched_item.number}")
                            else:
                                logger.warning(f"[{iter_num}] No README found for optimization #{matched_item.number}")
                    else:
                        logger.warning(f"[{iter_num}] Could not match '{selected}' to any remaining menu item")
            logger.info(f"[{iter_num}] Plan:\n{plan[:300]}...")

            # ── Phase 2: Implement ─────────────────────────────────────────
            logger.info(f"[{iter_num}] Implementing optimization...")
            new_code = self._implement(current_code, plan, opt_readme)
            trace_logger.log_artifact(f"opt_iter_{iter_num}_kernel.py", new_code)

            # ── Phase 3: Test (correctness + profile) ─────────────────────
            logger.info(f"[{iter_num}] Profiling candidate...")
            result = self.profiler.profile(new_code, test_code, wrapper_name)
            trace_logger.log_event(f"opt_iter_{iter_num}_profile", result)

            accepted = False
            if result["success"] and result["latency_ms"] is not None:
                new_latency = result["latency_ms"]
                if new_latency < current_latency:
                    logger.info(
                        f"[{iter_num}] Accepted: {current_latency} ms → {new_latency} ms "
                        f"({(current_latency/new_latency):.2f}× speedup)"
                    )
                    trace_logger.log_step(
                        f"opt_iter_{iter_num}_accepted",
                        f"Accepted: {current_latency} ms → {new_latency} ms",
                    )
                    current_code = new_code
                    current_latency = new_latency
                    accepted = True
                else:
                    logger.info(
                        f"[{iter_num}] Rejected: {new_latency} ms >= {current_latency} ms (no improvement)"
                    )
                    trace_logger.log_step(
                        f"opt_iter_{iter_num}_rejected",
                        f"No improvement: {new_latency} ms >= {current_latency} ms",
                    )
            elif not result["success"]:
                logger.warning(f"[{iter_num}] Rejected: correctness test failed — {result.get('error')}")
                trace_logger.log_step(
                    f"opt_iter_{iter_num}_failed",
                    f"Correctness failed: {result.get('error')}",
                )
            else:
                logger.warning(f"[{iter_num}] Rejected: no latency measurement available")

            history.append({
                "iteration": iter_num,
                "plan": plan,
                "code": new_code,
                "profile_result": result,
                "accepted": accepted,
            })

        final_speedup = (baseline_latency / current_latency) if current_latency else None
        logger.info(
            f"Optimization complete. "
            f"Baseline: {baseline_latency} ms, Final: {current_latency} ms"
            + (f", Speedup: {final_speedup:.2f}×" if final_speedup else "")
        )
        trace_logger.log_step(
            "opt_complete",
            f"baseline={baseline_latency}ms final={current_latency}ms speedup={final_speedup}",
        )
        trace_logger.log_artifact("opt_final_kernel.py", current_code)

        # Attach baseline info so callers can build comparison tables
        history.insert(0, {
            "iteration": 0,
            "plan": None,
            "code": kernel_code,
            "profile_result": baseline_result,
            "accepted": True,
        })

        return current_code, history

    # ── Private helpers ──────────────────────────────────────────────────────

    def _plan(
        self,
        kernel_code: str,
        current_latency_ms: Optional[float],
        remaining_items: List[OptimizationItem],
    ) -> str:
        """Ask the LLM to pick one or more optimizations and describe how to apply them.

        Single-opt mode: response begins with SELECTED_OPTIMIZATION: <name>.
        Multi-opt mode: response begins with a SELECTED_OPTIMIZATIONS: list block
        followed by one ### #NN · Name section per optimization.
        """
        latency_info = (
            f"Current measured latency: {current_latency_ms:.3f} ms\n"
            if current_latency_ms is not None
            else ""
        )
        if self.menu_mode == MenuMode.PRE_EXPLORED:
            menu_content = _render_pre_explored_menu(self._pre_menu, remaining_items)
        else:
            menu_content = _render_autocomp_menu(remaining_items)

        if self.multi_opt:
            user_prompt = f"""\
{latency_info}
Kernel code to optimize:
```python
{kernel_code}
```

Optimization menu:
{menu_content}

Analyze the kernel above against the optimization menu. For each menu item, check whether \
its "Apply when" condition is met by this specific code. Select ALL optimizations whose \
condition matches (and whose "Skip when" condition does NOT apply).

Start your response with exactly this block (one line per selected optimization):
SELECTED_OPTIMIZATIONS:
- #NN · Name of optimization
- #NN · Name of optimization

Then write one section per selected optimization:
### #NN · Name
**Signal**: what in this code triggers it (specific line/variable/pattern)
**Root cause**: why this is slow
**Fix**: exactly what to change (reference specific lines, loops, or variables)
**Expected benefit**: estimated improvement
"""
        else:
            user_prompt = f"""\
{latency_info}
Kernel code to optimize:
```python
{kernel_code}
```

Optimization menu:
{menu_content}

Pick the single most applicable optimization from the menu above and describe specifically \
how to apply it to this kernel to reduce execution time. \
Be concrete: reference specific lines, loops, or variables in the code. \
Do NOT simply restate the menu item — explain exactly what change to make and why it will help.

Start your response with exactly this line (fill in the name of the chosen optimization):
SELECTED_OPTIMIZATION: <name of chosen optimization>
"""
        with self.plan_agent.session():
            return self.plan_agent.run_sync(user_prompt)

    def _implement(self, kernel_code: str, plan: str, opt_readme: Optional[str] = None) -> str:
        """Ask the LLM to implement the plan, returning complete updated NKI kernel code."""
        readme_section = (
            f"\nDetailed optimization reference (before/after examples and profile data):\n"
            f"<optimization_readme>\n{opt_readme}\n</optimization_readme>\n"
            if opt_readme
            else ""
        )
        user_prompt = f"""\
Original NKI kernel code:
```python
{kernel_code}
```

Optimization plan:
{plan}
{readme_section}
Implement the plan above. Output the COMPLETE, updated NKI kernel code (kernel + wrapper). \
Do not leave any TODOs or placeholders.
"""
        with self.impl_agent.session():
            response = self.impl_agent.run_sync(user_prompt)
        return _extract_code(response)
