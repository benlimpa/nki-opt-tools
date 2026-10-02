import libcst as cst
from typing import Optional

class WrapperFunctionExtractor(cst.CSTVisitor):
    def __init__(self):
        self.wrapper_name = None
        self.last_func_name = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        # We only care about top-level functions for the wrapper
        # LibCST visitor visits nested functions too, but we can check the depth or just keep the last one seen at top level.
        # However, visit_FunctionDef is called for all.
        # For simplicity, let's just keep track of the last one and check comments.
        self.last_func_name = node.name.value

        # Check leading lines for the "Wrapper Function:" comment
        for line in node.leading_lines:
            if line.comment and "wrapper function:" in line.comment.value.lower():
                self.wrapper_name = node.name.value
                break
        return True

def extract_wrapper_name(code: str, strict: bool = False) -> Optional[str]:
    """
    Extracts the name of the wrapper function from the code using LibCST.
    The convention is that the wrapper function is preceded by '# Wrapper Function:'
    or it is the last function defined in the code.
    """
    try:
        tree = cst.parse_module(code)
        visitor = WrapperFunctionExtractor()
        tree.visit(visitor)

        # Return the specifically marked wrapper, or the last function as a fallback
        if visitor.wrapper_name:
            return visitor.wrapper_name
        elif strict:
            return None
        return visitor.last_func_name
    except Exception:
        if strict:
            return None
        # Fallback to a very simple regex if LibCST fails (e.g., partial code)
        import re
        match = re.search(r"#\s*Wrapper\s+Function:.*?\n\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        matches = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
        if matches:
            return matches[-1]
        return None

def pair_kernel_and_wrapper(kernel_code: str, wrapper_code: str) -> str:
    """
    Pairs a kernel with its wrapper function according to the convention.
    """
    kernel_code = kernel_code.strip()
    wrapper_code = wrapper_code.strip()

    # Add the convention comment if not present in the wrapper code itself
    if "# Wrapper Function:" not in wrapper_code:
        wrapper_code = "# Wrapper Function:\n" + wrapper_code

    return kernel_code + "\n\n" + wrapper_code
