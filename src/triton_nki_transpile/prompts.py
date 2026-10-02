from pathlib import Path

def get_resource_path(filename: str) -> Path:
    return Path(__file__).parent / "resources" / filename

def load_prompt(filename: str) -> str:
    path = get_resource_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()

NKI_IMPORTS = """
import nki as nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla
"""

IMPLEMENTATION_REQUIREMENTS = """
## Implementation Requirements

1. **Kernel vs Wrapper:**
   - The kernel itself (decorated with `@nki.jit`) MUST NOT be a placeholder or contain `TODO` comments. It must actually implement the requested computation adhering to NKI's architectural and programming model constraints.
   - The wrapper function (named correctly and preceded by the `# Wrapper Function:` comment) is responsible ONLY for:
     - Obtaining the Neuron device via `device = torch_xla.device()` and moving input tensors to it with `.to(device)`.
     - Performing any input reshaping or dtype casting needed before calling the kernel.
     - Calling the NKI kernel — the kernel allocates its own output buffers via `nl.ndarray(buffer=nl.shared_hbm)` and returns them directly.
     - Returning the result as a `torch.Tensor`.
   - **CRITICAL — no CUDA code in wrappers:** Do NOT add `torch.cuda.is_available()` checks, `torch.device("cuda")`, or any other CUDA-specific code. NKI kernels run on AWS Trainium/Inferentia via XLA, not on CUDA GPUs. Device is always `torch_xla.device()`.
   - **Do NOT pre-allocate output tensors with `torch.empty(...)` or `torch.zeros(...)` in the wrapper.** NKI kernels allocate and return their own outputs internally using `nl.ndarray(buffer=nl.shared_hbm)`.
   - You may modify the provided wrapper function as needed (e.g., adding grid/block arguments, adjusting tensor shapes).

2. **Completeness:**
   - Always output the COMPLETE, updated NKI code, including all necessary imports.
   - Do NOT leave any `TODO` comments or placeholders in the code.

3. **Convention:**
   - The high-level wrapper function MUST be preceded by the comment `# Wrapper Function:`.
   - The high-level wrapper function MUST be the LAST function defined in the code block.

## Wrapper Pattern (Minimal Correct Example)

```python
# Wrapper Function:
def my_op(x, weight):
    device = torch_xla.device()
    x = x.to(device)              # move inputs to Neuron device
    weight = weight.to(device)    # (do NOT check torch.cuda.is_available())
    result = my_op_kernel(x, weight)  # kernel allocates its own output and returns it
    return result
```
"""

OUTPUT_FORMAT_INSTRUCTIONS = """
## Output Format
To ensure robust parsing, your response MUST contain the Python code within a single markdown code block:

```python
# All imports here
...
# Decorated kernel here
...
# Wrapper Function:
def wrapper(...):
    ...
```
Do not include any text outside the code block if possible, or keep it to a minimum.
"""

def get_wrapper_instructions(is_nki: bool = False) -> str:
    kernel_type = "NKI" if is_nki else "Triton"
    device = "torch_xla.device()" if is_nki else "'cuda'"

    nki_extra = """
**CRITICAL — NKI wrapper rules (apply ONLY for NKI kernels):**
- Do NOT add `torch.cuda.is_available()` checks or any CUDA-specific code. NKI kernels run on AWS Trainium/Inferentia via XLA, not CUDA GPUs.
- Do NOT pre-allocate output tensors with `torch.empty(...)` or `torch.zeros(...)`. NKI kernels allocate their own outputs internally via `nl.ndarray(buffer=nl.shared_hbm)` and return them directly.
- Always obtain the device via `import torch_xla; device = torch_xla.device()` and move inputs with `.to(device)`.

Minimal correct NKI wrapper:
```python
# Wrapper Function:
def my_op(x, weight):
    import torch_xla
    device = torch_xla.device()
    x = x.to(device)
    weight = weight.to(device)
    result = my_op_kernel(x, weight)  # kernel allocates and returns output
    return result
```
""" if is_nki else ""

    return f"""
You are an expert in {kernel_type} and Python.
Your task is to generate a Python wrapper function for a given {kernel_type} kernel.

The wrapper should:
1. Have a clean, high-level signature (e.g., `add(x, y)` for an addition kernel).
2. Handle all device-specific boilerplate:
   - Moving input tensors to the correct device ({device}).
   - Setting up the grid/blocks and launching the kernel.
3. Return the result as a `torch.Tensor` on the same device.
4. Be named something descriptive related to the kernel operation.
{nki_extra}
The output should ONLY be the Python code for the wrapper function and any necessary imports.
Do not include the kernel code itself.

**CONVENTION:**
- The wrapper function MUST be preceded by the comment `# Wrapper Function:`.
- The wrapper function MUST be the LAST function defined in your code.

{OUTPUT_FORMAT_INSTRUCTIONS}
"""



def get_optimizer_plan_instructions(menu_mode: str = "autocomp", multi_opt: bool = False) -> str:
    nki_info = load_prompt("nki-info.md")
    if multi_opt:
        mode_note = (
            "Focus on concrete, benchmark-validated optimizations with clear applicability signals."
            if menu_mode == "pre-explored"
            else "Be creative — the menu items are high-level guides; your plan should be tailored to this specific code."
        )
        return f"""You are an expert NKI (Neuron Kernel Interface) performance engineer.
Your task is to analyze an NKI kernel, identify all applicable optimizations from the provided menu, and write a structured plan for implementing them together.

## NKI Architecture & Programming Model

{nki_info}

## Rules
1. Pick ALL optimizations from the menu whose "Apply when" condition is satisfied by the actual code. Include every match — do not limit to just one.
2. For each selected optimization, check its "Skip when" condition and omit it if that condition is met.
3. Be specific: reference actual lines, loop variables, or function names from the code.
4. Explain WHY each optimization helps (e.g., eliminates a redundant HBM→SBUF transfer, reduces spill_save_bytes, overlaps DMA and compute).
5. {mode_note}
6. Do NOT propose changes that alter the kernel's semantic behavior (correctness must be preserved).
7. Keep the same kernel function name and wrapper function name.

## Output Format

Start your response with exactly this block (one line per selected optimization):
SELECTED_OPTIMIZATIONS:
- #NN · Name of optimization
- #NN · Name of optimization

Then write one section per selected optimization using this structure:
### #NN · Name
**Signal**: what in this code triggers it (specific line/variable/pattern)
**Root cause**: why this is slow
**Fix**: exactly what to change (reference specific lines, loops, or variables)
**Expected benefit**: estimated improvement (e.g., fewer HBM reads, reduced spill, better parallelism)
"""
    else:
        return f"""You are an expert NKI (Neuron Kernel Interface) performance engineer.
Your task is to analyze an NKI kernel and select the single most impactful optimization to apply.

## NKI Architecture & Programming Model

{nki_info}

## Rules
1. Pick exactly ONE optimization from the provided menu.
2. Be specific: reference actual lines, loop variables, or function names from the code.
3. Explain WHY the optimization helps (e.g., eliminates a redundant HBM→SBUF transfer, reduces spill_save_bytes, overlaps DMA and compute).
4. {"Focus on concrete, benchmark-validated optimizations with clear applicability signals." if menu_mode == "pre-explored" else "Be creative — the menu items are high-level guides; your plan should be tailored to this specific code."}
5. Do NOT propose changes that alter the kernel's semantic behavior (correctness must be preserved).
6. Keep the same kernel function name and wrapper function name.

Output a concise but specific plan (2–5 sentences) that an implementer can follow directly.
"""


def get_optimizer_implement_instructions() -> str:
    api_spec = load_prompt("distilled-nki-api-spec.md")
    nki_info = load_prompt("nki-info.md")
    language_guide = load_prompt("nki-language-guide.rst")
    return f"""You are an expert NKI (Neuron Kernel Interface) kernel engineer.
Your task is to implement a given optimization plan on an NKI kernel.

## NKI API Reference

{api_spec}

## NKI Architecture & Programming Model Info

{nki_info}

## NKI Language Guide

{language_guide}

## Necessary Imports

All generated NKI kernels MUST start with these imports:

{NKI_IMPORTS}

{IMPLEMENTATION_REQUIREMENTS}

{OUTPUT_FORMAT_INSTRUCTIONS}
"""


def get_optimizer_instructions() -> str:
    base_prompt = load_prompt("optimizer-prompt-updated.md")
    api_spec = load_prompt("distilled-nki-api-spec.md")
    nki_info = load_prompt("nki-info.md")
    language_guide = load_prompt("nki-language-guide.rst")

    extra_instructions = f"""

## NKI API Reference

{api_spec}

## NKI Architecture & Programming Model Info

{nki_info}

## NKI Language Guide

{language_guide}

## Necessary Imports

All generated NKI kernels MUST start with these imports:

{NKI_IMPORTS}

{IMPLEMENTATION_REQUIREMENTS}

{OUTPUT_FORMAT_INSTRUCTIONS}
"""
    return base_prompt + "\n" + extra_instructions
