import logging
import os
import contextlib
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import boto3
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from triton_nki_transpile.config import Config, get_config
from triton_nki_transpile.logger import trace_logger

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class Agent:
    """
    Agent class to manage prompts and LLM calls.
    Supports OpenAI Responses API and Amazon Bedrock Converse API.
    """
    def __init__(
        self,
        instructions: str,
        config: Optional[Config] = None,
    ):
        self.config = config or get_config()
        self.provider = self.config.llm_provider
        self.model_name = self.config.model_name
        self.instructions = instructions
        self.messages: List[Dict[str, Any]] = []

        if self.provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OPENAI_API_KEY must be set for OpenAI provider.")
            self.api_key = self.config.openai_api_key
            self.base_url = self.config.openai_api_base
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "bedrock":
            self.bedrock_client = boto3.client("bedrock-runtime", region_name=self.config.bedrock_region)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def clear_history(self):
        """
        Clears the conversation history.
        """
        self.messages = []

    @contextlib.contextmanager
    def session(self):
        """
        Context manager to ensure conversation history is maintained during a session
        and cleared afterwards.
        """
        self.clear_history()
        try:
            yield self
        finally:
            self.clear_history()

    def add_message(self, role: str, content: str):
        """
        Adds a message to the conversation history, handling provider-specific formats.
        """
        if self.provider == "openai":
            # Following the user's example format: { "role": "role", "content": "..." }
            self.messages.append({
                "role": role,
                "content": content,
            })
        elif self.provider == "bedrock":
            self.messages.append({
                "role": role,
                "content": [{"text": content}]
            })

    def pop_message(self) -> Dict[str, Any]:
        """
        Removes and returns the last message from the history.
        """
        if self.messages:
            return self.messages.pop()
        return {}

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Returns the current conversation history.
        """
        return self.messages

    def run_sync(
        self,
        user_prompt: str,
    ) -> str:
        """
        Runs the agent synchronously, maintaining conversation history.
        """
        trace_logger.log_prompt(user_prompt, system_instructions=self.instructions)

        self.add_message("user", user_prompt)

        if self.provider == "openai":
            response = self.client.responses.create(
                model=self.model_name,
                instructions=self.instructions,
                input=self.messages,
            )
            # Append the response's output directly to history
            self.messages += response.output
            output_text = response.output_text
        elif self.provider == "bedrock":
            system = [{"text": self.instructions}]

            response = self.bedrock_client.converse(
                modelId=self.model_name,
                messages=self.messages,
                system=system,
            )
            output_text = response["output"]["message"]["content"][0]["text"]
            self.add_message("assistant", output_text)

        trace_logger.log_response(output_text)
        return output_text

    async def run(
        self,
        user_prompt: str,
    ) -> str:
        """
        Runs the agent asynchronously, maintaining conversation history.
        """
        if self.provider == "openai":
            trace_logger.log_prompt(user_prompt, system_instructions=self.instructions)

            self.add_message("user", user_prompt)

            response = await self.async_client.responses.create(
                model=self.model_name,
                instructions=self.instructions,
                input=self.messages,
            )
            # Append the response's output directly to history
            self.messages += response.output
            output_text = response.output_text

            trace_logger.log_response(output_text)
            return output_text
        else:
            # Fallback to sync for bedrock as aiobotocore is not yet integrated
            return self.run_sync(user_prompt)
