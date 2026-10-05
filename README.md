# NKI Opt Tools

NKI Opt Tools is research software for selecting and testing LLM-generated kernel optimizations on AWS Trainium. Per-optimization performance models guide which NKI source rewrites to try. The behavioral testing framework derives input-space partitions from kernel structure and generates differential correctness tests around tile boundaries and other behavioral regimes.

The toolkit includes the optimization pipeline, performance models, benchmark kernels, and correctness-testing tools.

## Contents

| Directory | Purpose |
| --- | --- |
| `src/triton_nki_transpile/` | Optimizer, LLM interfaces, profiling, and benchmark-run utilities; the Python import name is retained for compatibility |
| `nki-opt-models-v2/` | Parametric predictors, calibration generators, fitting code, and sweep kernels |
| `nki-optimizations/` | Optimization applicability catalog and before/after code examples |
| `benchmark-opt/` | Agent workspace builder, experiment runner, benchmark kernels, and oracles |
| `kernel-testing/coverage/` | Partition derivation, input generation, differential oracle, and campaign code |
| `kernel-testing/repair/` | Agent-based repair code |
| `kernel-testing/lncfix-analysis/` | Analysis scripts requiring locally generated runs |

Generate calibration measurements and fitted coefficients locally before using the fitted predictors. The repository provides calibration plans, sweep kernels, and fitting scripts; experimental data is not included.

## Requirements

- Python 3.11 or newer for the offline code (`tomllib` is used).
- For kernel execution: an AWS Trainium instance with the appropriate Neuron driver, runtime, compiler, NKI, Neuron-enabled PyTorch, and `torch_xla` installed together.
- The container definition uses Python 3.12, Neuron SDK 2.28.0, and PyTorch NeuronX 2.9.0. Kernel sources mix `nki` Beta 2 and `neuronxcc.nki` APIs; arbitrary SDK upgrades are not assumed compatible.
- LLM access through the configured OpenAI or Amazon Bedrock provider; the Docker runner additionally uses the Codex CLI.

The SDK-specific environment cannot be reconstructed solely from `pyproject.toml`. Install into an existing compatible Neuron environment and preserve its vendor-provided Torch packages. A GPU/Triton server is not part of this artifact.

## Offline checks

These tests need no Trainium hardware, model credentials, or experiment data:

```bash
python kernel-testing/coverage/test_spectra.py
python kernel-testing/coverage/test_generate.py
python kernel-testing/coverage/test_grounding.py
python kernel-testing/coverage/test_oracle_recipe.py
python kernel-testing/coverage/test_run_index.py
python kernel-testing/coverage/test_campaign.py
```

The separate `test_kernel_oracle.py` uses Torch and NumPy for its CPU-side mock tests.

## Install on a prepared Neuron host

From the repository root, activate your existing Neuron environment. The scripts expect `/opt/aws_neuronx_venv_pytorch_2_9`; inspect and adjust that assumption if your environment differs.

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
uv pip install --no-deps -e .
```

The `--no-deps` option avoids replacing the SDK's Torch installation. Ensure the remaining packages listed in `pyproject.toml` are installed in that environment. Editable installation is required because scripts resolve the catalog, models, and benchmark files relative to the checkout.

Some benchmark adapters import AWS NKI Library. Obtain the pinned dependency separately:

```bash
git clone https://github.com/aws-neuron/nki-library.git nki-library
git -C nki-library checkout d2ad3a579498d1888aa8740af97fd325dd5f2740
export PYTHONPATH="$PWD/nki-library/src:$PWD/src:${PYTHONPATH:-}"
```

Set `LLM_PROVIDER`, `MODEL_NAME`, and the provider credentials in your environment. `OPENAI_API_KEY` is used for OpenAI; Bedrock uses the AWS credential chain and `AWS_DEFAULT_REGION`. Never commit credentials. Inspect `nki-optimizer --help`, `nki-profile --help`, and `bench-status --help` before device execution.

## Calibrate performance models

Inspect and validate the before/after pair and oracle on your target platform first. For example:

```bash
python nki-opt-models-v2/tools/run_calibration_sweep.py \
  --plan nki-opt-models-v2/opt01-sbuf-accumulator/calibration_plan.yaml --dry-run

# Requires Trainium; performs profiling and creates local measurements.
python nki-opt-models-v2/tools/run_calibration_sweep.py \
  --plan nki-opt-models-v2/opt01-sbuf-accumulator/calibration_plan.yaml

(cd nki-opt-models-v2/opt01-sbuf-accumulator && python fit.py)
```

Repeat for the models you need. Some structural models do not require a fit file. Calibration data and generated coefficients are ignored by Git. See `nki-opt-models-v2/WORKFLOW_V2.md` for the schema and `KNOB_GUIDE.md` for structural inputs.

## Inspect an agent workspace

The paper's performance-model-guided workflow is implemented by the benchmark workspace builder. The standalone `nki-optimizer` CLI supports the AutoComp and catalog menus; it is not the full model-guided experiment runner.

```bash
python benchmark-opt/setup_agent_workspace.py \
  --kernel-path benchmark-opt/NKIBenchB2/kernels/adamw.py \
  --oracle-path benchmark-opt/NKIBenchB2/oracles/adamw.py \
  --approach pre-explored --approach-variant v2 \
  --output-dir /tmp/nki-opt-example
```

For `model-guided-v2`, generate calibration fits first and provide a fresh baseline profile via `--baseline-profile-json`. Review generated instructions before starting an agent. Approach variants are configured in `benchmark-opt/approaches.toml`.

## Correctness testing

`kernel-testing/coverage/grounding.py` derives exact, partial-tile, and below-tile input regions from source structure. `campaign.py` plans covering shapes. `kernel_oracle.py` compares baseline, candidate, and reference behavior on device. Coverage means coverage of the modeled partitions; it is not a proof over all possible inputs.

Device campaigns and the repair loop require local before/after runs with their oracles.

## License

Licensed under the [Apache License 2.0](LICENSE). Third-party notices and license text are preserved in `third-party/`.
