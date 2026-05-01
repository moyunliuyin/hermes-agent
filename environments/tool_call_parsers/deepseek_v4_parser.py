"""
DeepSeek V4 tool call parser.

DeepSeek V4 (released 2026-04) currently emits the same simplified
single-line format as V3.1:
    <｜tool▁call▁begin｜>function_name<｜tool▁sep｜>arguments<｜tool▁call▁end｜>

This parser is registered separately so any V4-specific format
divergence can land here without affecting V3.1 callers.
"""

from environments.tool_call_parsers import register_parser
from environments.tool_call_parsers.deepseek_v3_1_parser import (
    DeepSeekV31ToolCallParser,
)


@register_parser("deepseek_v4")
@register_parser("deepseek-v4")
class DeepSeekV4ToolCallParser(DeepSeekV31ToolCallParser):
    """V4 currently shares V3.1's wire format; subclassing keeps the
    registration distinct so any future V4-only format change can override
    only what differs (typically just `PATTERN`).
    """
    pass
