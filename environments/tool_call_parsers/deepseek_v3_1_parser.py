"""
DeepSeek V3.1 tool call parser.

Similar to V3 but with a slightly different format:
    <｜tool▁call▁begin｜>function_name<｜tool▁sep｜>arguments<｜tool▁call▁end｜>

Note: V3 has type+name before the separator, V3.1 has name before and args after.

Based on VLLM's DeepSeekV31ToolParser.extract_tool_calls()
"""

import re

from environments.tool_call_parsers import register_parser
from environments.tool_call_parsers._deepseek_base import _DeepSeekToolCallParserBase


@register_parser("deepseek_v3_1")
@register_parser("deepseek_v31")
class DeepSeekV31ToolCallParser(_DeepSeekToolCallParserBase):
    """
    Parser for DeepSeek V3.1 tool calls.

    Slightly different regex than V3: function_name comes before the separator,
    arguments come after (no type field, no json code block wrapper).
    """

    PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>",
        re.DOTALL,
    )
