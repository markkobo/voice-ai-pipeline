"""
Async OpenAI-compatible streaming LLM client with cancellation support.

Supports:
- Async streaming with time-to-first-token (TTFT) measurement
- Task cancellation for barge-in
- OpenAI-compatible base URL configuration
- Function calling for structured output
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional, Dict, Any, List

import openai
from openai import AsyncOpenAI

from telemetry import metrics


# Function schema for structured emotional response
EMOTION_FUNCTION_SCHEMA = {
    "name": "emotional_response",
    "description": "Respond with an emotion tag and content. Emotion must be one of: 幽默, 寵溺, 撒嬌, 毒舌, 溫和, 調皮, 開心, 感動, 生氣, 認真",
    "parameters": {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": ["幽默", "寵溺", "撒嬌", "毒舌", "溫和", "調皮", "開心", "感動", "生氣", "認真"]
            },
            "content": {"type": "string"}
        },
        "required": ["emotion", "content"]
    }
}


class LLMStreamEvent(Enum):
    """LLM streaming event types."""
    START = "start"
    CONTENT_DELTA = "content_delta"
    CONTENT_DONE = "content_done"
    ERROR = "error"
    CANCELLED = "cancelled"
    FUNCTION_CALL = "function_call"


@dataclass
class LLMStreamResult:
    """Result from a single streaming turn."""
    event: LLMStreamEvent
    content: str = ""
    ttft_seconds: Optional[float] = None
    total_tokens: int = 0
    error: Optional[str] = None
    # For function call results
    function_call: Optional[Dict[str, Any]] = None


class OpenAIClient:
    """
    Async streaming LLM client wrapping AsyncOpenAI.

    Supports cancellation via asyncio.Task and measures TTFT for telemetry.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_system_prompt: str = "You are a helpful voice AI assistant.",
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name (default: gpt-4o-mini)
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI-compatible APIs
            default_system_prompt: Default system prompt
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.default_system_prompt = default_system_prompt

        # Build client kwargs
        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = AsyncOpenAI(**client_kwargs)

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[LLMStreamResult]:
        """
        Stream LLM response with cancellation support.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            conversation_history: Optional list of {"role": ..., "content": ...} dicts
            temperature: Sampling temperature
            cancellation_event: Optional asyncio.Event; if set, cancelled tokens are skipped

        Yields:
            LLMStreamResult events with content deltas and telemetry
        """
        messages: List[Dict[str, str]] = []

        system = system_prompt or self.default_system_prompt
        messages.append({"role": "system", "content": system})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": prompt})

        total_tokens = 0
        ttft_recorded = False
        ttft_time: Optional[float] = None
        start_time = time.perf_counter()

        # Emit start event
        yield LLMStreamResult(event=LLMStreamEvent.START)

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=temperature,
            )

            content_buffer = ""

            async for chunk in stream:
                # Check cancellation
                if cancellation_event is not None and cancellation_event.is_set():
                    yield LLMStreamResult(
                        event=LLMStreamEvent.CANCELLED,
                        content=content_buffer,
                        total_tokens=total_tokens,
                    )
                    return

                delta = chunk.choices[0].delta
                if delta.content:
                    if not ttft_recorded:
                        ttft_time = time.perf_counter() - start_time
                        ttft_recorded = True
                        metrics.llm_ttft.labels(
                            component="llm",
                            model=self.model,
                        ).observe(ttft_time)

                    content_buffer += delta.content
                    yield LLMStreamResult(
                        event=LLMStreamEvent.CONTENT_DELTA,
                        content=delta.content,
                        ttft_seconds=ttft_time,
                    )

                if chunk.choices[0].finish_reason:
                    total_tokens = getattr(chunk, "usage", None) and getattr(
                        chunk.usage, "total_tokens", 0
                    ) or 0
                    metrics.llm_tokens_total.labels(
                        component="llm",
                        model=self.model,
                        session_id="",
                    ).inc(total_tokens)

            yield LLMStreamResult(
                event=LLMStreamEvent.CONTENT_DONE,
                content=content_buffer,
                total_tokens=total_tokens,
            )

        except asyncio.CancelledError:
            yield LLMStreamResult(
                event=LLMStreamEvent.CANCELLED,
                content=content_buffer,
            )
            raise
        except Exception as e:
            metrics.llm_failures_total.labels(
                component="llm",
                model=self.model,
                error_type=type(e).__name__,
            ).inc()
            yield LLMStreamResult(
                event=LLMStreamEvent.ERROR,
                error=str(e),
            )

    async def stream_with_function(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        cancellation_event: Optional[asyncio.Event] = None,
        function_schema: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMStreamResult]:
        """
        Stream LLM response with function calling for structured output.

        Yields:
            LLMStreamResult events with function_call data when available
        """
        if function_schema is None:
            function_schema = EMOTION_FUNCTION_SCHEMA

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append(
                {"role": "system", "content": self.default_system_prompt}
            )

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": prompt})

        total_tokens = 0
        ttft_recorded = False
        ttft_time: Optional[float] = None
        start_time = time.perf_counter()

        yield LLMStreamResult(event=LLMStreamEvent.START)

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=temperature,
                tools=[{"type": "function", "function": function_schema}],
                tool_choice={"type": "function", "function": {"name": function_schema["name"]}},
            )

            function_name = ""
            arguments_buffer = ""

            async for chunk in stream:
                if cancellation_event is not None and cancellation_event.is_set():
                    yield LLMStreamResult(
                        event=LLMStreamEvent.CANCELLED,
                        content=arguments_buffer,
                        total_tokens=total_tokens,
                    )
                    return

                delta = chunk.choices[0].delta

                # Handle function call
                if delta.tool_calls:
                    tool_call = delta.tool_calls[0]
                    if tool_call.function:
                        if tool_call.function.name:
                            function_name = tool_call.function.name
                        if tool_call.function.arguments:
                            arguments_buffer += tool_call.function.arguments

                            if not ttft_recorded:
                                ttft_time = time.perf_counter() - start_time
                                ttft_recorded = True

                            yield LLMStreamResult(
                                event=LLMStreamEvent.CONTENT_DELTA,
                                content=tool_call.function.arguments,
                                ttft_seconds=ttft_time,
                            )

                if chunk.choices[0].finish_reason:
                    total_tokens = getattr(chunk, "usage", None) and getattr(
                        chunk.usage, "total_tokens", 0
                    ) or 0
                    metrics.llm_tokens_total.labels(
                        component="llm",
                        model=self.model,
                        session_id="",
                    ).inc(total_tokens)

                    # Parse function arguments
                    if arguments_buffer:
                        try:
                            args = json.loads(arguments_buffer)
                            yield LLMStreamResult(
                                event=LLMStreamEvent.FUNCTION_CALL,
                                function_call={
                                    "name": function_name,
                                    "arguments": args,
                                },
                                total_tokens=total_tokens,
                            )
                        except json.JSONDecodeError:
                            yield LLMStreamResult(
                                event=LLMStreamEvent.ERROR,
                                error=f"Failed to parse function arguments: {arguments_buffer}",
                            )

            yield LLMStreamResult(
                event=LLMStreamEvent.CONTENT_DONE,
                total_tokens=total_tokens,
            )

        except asyncio.CancelledError:
            yield LLMStreamResult(
                event=LLMStreamEvent.CANCELLED,
                content=arguments_buffer,
            )
            raise
        except Exception as e:
            metrics.llm_failures_total.labels(
                component="llm",
                model=self.model,
                error_type=type(e).__name__,
            ).inc()
            yield LLMStreamResult(
                event=LLMStreamEvent.ERROR,
                error=str(e),
            )

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Non-streaming completion for testing or short prompts.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            temperature: Sampling temperature

        Returns:
            Dict with text and token count
        """
        messages = []
        system = system_prompt or self.default_system_prompt
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start_time = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - start_time

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return {
            "text": text,
            "total_tokens": tokens,
            "latency_seconds": elapsed,
        }


class MockLLMClient:
    """
    Mock LLM client for testing - simulates streaming with configurable latency.
    """

    def __init__(
        self,
        response_text: str = "這是模擬的 LLM 回應。",
        token_delay_ms: float = 50.0,
        ttft_ms: float = 200.0,
    ):
        """
        Initialize mock LLM client.

        Args:
            response_text: Text to return
            token_delay_ms: Delay between each "token"
            ttft_ms: Time to first token in ms
        """
        self.response_text = response_text
        self.token_delay_ms = token_delay_ms
        self.ttft_ms = ttft_ms

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[LLMStreamResult]:
        """Simulate streaming LLM response."""
        yield LLMStreamResult(event=LLMStreamEvent.START)

        # Simulate TTFT delay
        await asyncio.sleep(self.ttft_ms / 1000.0)
        ttft_time = self.ttft_ms / 1000.0

        metrics.llm_ttft.labels(component="llm", model="mock").observe(ttft_time)

        # Stream token by token
        for char in self.response_text:
            if cancellation_event is not None and cancellation_event.is_set():
                yield LLMStreamResult(
                    event=LLMStreamEvent.CANCELLED,
                    content=self.response_text[: self.response_text.index(char)]
                    if char in self.response_text
                    else "",
                )
                return

            await asyncio.sleep(self.token_delay_ms / 1000.0)
            yield LLMStreamResult(
                event=LLMStreamEvent.CONTENT_DELTA,
                content=char,
                ttft_seconds=ttft_time,
            )

        yield LLMStreamResult(
            event=LLMStreamEvent.CONTENT_DONE,
            content=self.response_text,
            total_tokens=len(self.response_text),
        )
