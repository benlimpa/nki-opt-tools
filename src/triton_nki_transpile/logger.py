import os
import json
import logging
import datetime
from pathlib import Path
from typing import Any, List, Optional

class TraceLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceLogger, cls).__new__(cls)
            cls._instance.log_dir = None
            cls._instance.logger = logging.getLogger("triton_nki_transpile_trace")
        return cls._instance

    def set_log_dir(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler to standard logger as well
        fh = logging.FileHandler(self.log_dir / "trace.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)

    def log_version(self, version: str):
        if self.log_dir:
            version_file = self.log_dir / "VERSION"
            version_file.write_text(version)
            self.logger.info(f"LOGGED VERSION: {version}")

    def log_step(self, step_name: str, details: str):
        self.logger.info(f"STEP: {step_name} - {details}")
        if self.log_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            step_file = self.log_dir / f"step_{timestamp}_{step_name}.txt"
            step_file.write_text(details)

    def log_event(self, event_type: str, data: Any, name_hint: str = ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{event_type}"
        if name_hint:
            filename += f"_{name_hint}"
        filename += ".json"

        self.logger.info(f"EVENT: {event_type} {name_hint}")

        if self.log_dir:
            event_file = self.log_dir / filename
            with open(event_file, "w") as f:
                if isinstance(data, (dict, list)):
                    json.dump(data, f, indent=2)
                else:
                    f.write(str(data))

    def log_prompt(self, prompt: str, system_instructions: Optional[str] = None):
        data = {
            "system_instructions": system_instructions,
            "prompt": prompt
        }
        self.log_event("prompt", data)

    def log_response(self, response: str):
        """Save LLM response as Markdown."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_response.md"

        self.logger.info(f"RESPONSE: saved {filename}")

        if self.log_dir:
            resp_file = self.log_dir / filename
            content = response if response.endswith("\n") else response + "\n"
            resp_file.write_text(content)

    def log_llm_call(
        self,
        phase_tag: str,
        model: str,
        tools_used: List[str],
        messages_in: list,
        response_items: list,
    ):
        """Save a complete LLM API call (full input history + full response) for exact replay."""
        data = {
            "phase": phase_tag,
            "model": model,
            "tools": tools_used,
            "messages": messages_in,
            "response": response_items,
        }
        self.log_event("llm_call", data, name_hint=phase_tag)

    def log_tool_invocation(self, tool_name: str, args: Any, result: Any):
        data = {
            "tool": tool_name,
            "args": args,
            "result": result
        }
        self.log_event("tool_invocation", data, name_hint=tool_name)

    def log_error(self, error_msg: str, traceback_str: Optional[str] = None):
        data = {
            "error": error_msg,
            "traceback": traceback_str
        }
        self.log_event("error", data)

    def log_artifact(self, filename: str, content: str):
        """Save an artifact under the trace `log_dir/artifacts` with a timestamp prefix.

        The timestamp uses the same format as `log_event` and is prepended to the
        basename of `filename` while preserving any subdirectories.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if self.log_dir:
            orig_path = Path(filename)
            # If no parent directory is specified, default to 'artifacts/'
            if str(orig_path.parent) == ".":
                parent = Path("artifacts")
            else:
                parent = orig_path.parent

            basename = orig_path.name
            new_name = f"{timestamp}_{basename}"
            artifact_file = self.log_dir / parent / new_name
            artifact_file.parent.mkdir(parents=True, exist_ok=True)
            artifact_file.write_text(content)
            self.logger.info(f"LOGGED ARTIFACT: {parent / new_name}")

trace_logger = TraceLogger()
