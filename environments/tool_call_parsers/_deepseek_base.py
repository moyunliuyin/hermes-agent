"""Shared scaffolding for DeepSeek V3 / V3.1 / V4 tool-call parsers.

Every DeepSeek release reuses the same outer boundary token; only the
inner per-call regex differs. Subclasses provide a `PATTERN` with named
groups `function_name` and `function_arguments` and (optionally) override
`_on_parse_error()` to log on failure.
"""

import re
import uuid
from typing import Iterable

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser


class _DeepSeekToolCallParserBase(ToolCallParser):
    """Common parse() implementation for DeepSeek tool-call formats.

    Subclasses MUST set:
        PATTERN: re.Pattern with named groups 'function_name' and
                 'function_arguments'
    Subclasses MAY override:
        START_TOKEN          — default is the standard DeepSeek begin token.
        _on_parse_error(exc) — default swallows silently (preserves V3.1
                               behavior); V3 overrides to log at error level.
    """

    START_TOKEN: str = "<｜tool▁calls▁begin｜>"
    PATTERN: re.Pattern  # subclass-defined

    def parse(self, text: str) -> ParseResult:
        if self.START_TOKEN not in text:
            return text, None
        try:
            tool_calls = list(self._iter_tool_calls(text))
        except Exception as exc:
            self._on_parse_error(exc)
            return text, None
        if not tool_calls:
            return text, None
        content = text[: text.find(self.START_TOKEN)].strip()
        return content if content else None, tool_calls

    def _iter_tool_calls(self, text: str) -> Iterable[ChatCompletionMessageToolCall]:
        for match in self.PATTERN.finditer(text):
            yield ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=Function(
                    name=match.group("function_name").strip(),
                    arguments=match.group("function_arguments").strip(),
                ),
            )

    def _on_parse_error(self, exc: Exception) -> None:
        # Default: silent (V3.1 historical behavior). V3 logs at error.
        return
