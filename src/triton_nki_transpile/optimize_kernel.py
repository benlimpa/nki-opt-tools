"""
CLI entrypoint for NKI kernel optimization experiment.

nki-optimizer: Plan → Implement → Test loop comparing two menu strategies:
  --menu autocomp       AutoComp's vague high-level optimization menu
  --menu pre-explored   Our concretized menu with benchmark-verified speedups and
                        applicability signals from nki-optimizations/
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from triton_nki_transpile import __version__
from triton_nki_transpile.config import get_config
from triton_nki_transpile.logger import trace_logger
from triton_nki_transpile.nki_optimizer import MenuMode, NKIOptimizer
from triton_nki_transpile.utils import extract_wrapper_name

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--kernel-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to NKI kernel (must contain '# Wrapper Function:' comment)",
)
@click.option(
    "--oracle-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to oracle test defining run_test(wrapper)",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Path to write the optimized NKI kernel",
)
@click.option(
    "--menu",
    type=click.Choice(["autocomp", "pre-explored"]),
    default="autocomp",
    show_default=True,
    help=(
        "Optimization menu strategy: "
        "'autocomp' uses AutoComp's vague high-level descriptions; "
        "'pre-explored' uses our concretized menu with speedups and applicability signals"
    ),
)
@click.option("--max-iters", type=int, default=3, show_default=True, help="Number of optimization iterations")
@click.option("--log-dir", type=click.Path(), help="Directory to save trace logs and artifacts")
@click.option("--model", type=str, help="LLM model name override")
@click.option(
    "--llm-provider",
    type=click.Choice(["openai", "bedrock"]),
    help="LLM provider override",
)
@click.option("--openai-api-base", type=str, help="Base URL for OpenAI-compatible API")
@click.option("--bedrock-region", type=str, help="AWS region for Bedrock")
@click.option(
    "--multi-opt",
    is_flag=True,
    default=False,
    help=(
        "Select multiple optimizations per iteration using structured SKILL.md-style planning. "
        "The LLM analyzes all menu items for applicability and implements them together in one pass."
    ),
)
def main(
    kernel_path: str,
    oracle_path: str,
    output_path: str,
    menu: str,
    max_iters: int,
    log_dir: Optional[str],
    model: Optional[str],
    llm_provider: Optional[str],
    openai_api_base: Optional[str],
    bedrock_region: Optional[str],
    multi_opt: bool,
):
    """
    Optimize an NKI kernel using a plan → implement → test loop.

    Two menu strategies are compared via separate runs of this command:
    \b
      autocomp      AutoComp-style vague high-level optimization descriptions
      pre-explored  Our concretized menu with benchmark-verified speedups and
                    concrete applicability signals from nki-optimizations/
    """
    if log_dir:
        trace_logger.set_log_dir(log_dir)
        trace_logger.log_version(__version__)
        logger.info(f"Logging to {log_dir}")

    config = get_config()
    if model:
        config.model_name = model
    if llm_provider:
        config.llm_provider = llm_provider
    if openai_api_base:
        config.openai_api_base = openai_api_base
    if bedrock_region:
        config.bedrock_region = bedrock_region

    kernel_code = Path(kernel_path).read_text()
    test_code = Path(oracle_path).read_text()

    wrapper_name = extract_wrapper_name(kernel_code)
    if not wrapper_name:
        logger.error(
            "Could not extract wrapper function name from kernel. "
            "Ensure the kernel has a '# Wrapper Function:' comment above the wrapper."
        )
        sys.exit(1)

    logger.info(f"Kernel: {kernel_path}")
    logger.info(f"Oracle: {oracle_path}")
    logger.info(f"Wrapper: {wrapper_name}")
    logger.info(f"Menu: {menu}")
    logger.info(f"Max iters: {max_iters}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Multi-opt: {multi_opt}")

    menu_mode = MenuMode(menu)
    optimizer = NKIOptimizer(menu_mode=menu_mode, config=config, multi_opt=multi_opt)

    best_code, history = optimizer.optimize(
        kernel_code=kernel_code,
        test_code=test_code,
        wrapper_name=wrapper_name,
        max_iters=max_iters,
    )

    # Write optimized kernel
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(best_code)
    logger.info(f"Wrote optimized kernel to {output_path}")

    # Write history JSON alongside output
    history_path = out.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2, default=str))
    logger.info(f"Wrote optimization history to {history_path}")

    # Print summary table
    _print_summary(history, menu)


def _print_summary(history: list, menu: str):
    """Print a concise summary table of the optimization run."""
    if not history:
        return

    baseline = history[0]
    baseline_latency = (
        baseline["profile_result"].get("latency_ms") if baseline["profile_result"]["success"] else None
    )

    click.echo(f"\n{'='*60}")
    click.echo(f"Optimization Summary  [menu={menu}]")
    click.echo(f"{'='*60}")
    click.echo(f"{'Iter':<6} {'Latency (ms)':<15} {'Speedup':<10} {'Accepted'}")
    click.echo(f"{'-'*6} {'-'*15} {'-'*10} {'-'*8}")

    for entry in history:
        it = entry["iteration"]
        res = entry["profile_result"]
        lat = res.get("latency_ms") if res["success"] else None
        lat_str = f"{lat:.3f}" if lat is not None else "—"
        speedup_str = (
            f"{baseline_latency/lat:.2f}×"
            if (lat and baseline_latency)
            else "—"
        )
        accepted_str = "yes" if entry["accepted"] else "no"
        label = "baseline" if it == 0 else str(it)
        click.echo(f"{label:<6} {lat_str:<15} {speedup_str:<10} {accepted_str}")

    click.echo(f"{'='*60}\n")


if __name__ == "__main__":
    main()
