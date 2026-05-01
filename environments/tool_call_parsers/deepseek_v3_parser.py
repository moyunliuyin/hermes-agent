"""
DeepSeek V3 tool call parser.

Format uses special unicode tokens:
    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>type<｜tool▁sep｜>function_name
    ```json
    {"arg": "value"}
    ```
    <｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

Fixes Issue #989: Support for multiple simultaneous tool calls.
"""

import logging
import re

from environments.tool_call_parsers import register_parser
from environments.tool_call_parsers._deepseek_base import _DeepSeekToolCallParserBase

logger = logging.getLogger(__name__)


@register_parser("deepseek_v3")
class DeepSeekV3ToolCallParser(_DeepSeekToolCallParserBase):
    """
    Parser for DeepSeek V3 tool calls.

    Uses special unicode tokens with fullwidth angle brackets and block elements.
    Extracts type, function name, and JSON arguments from the structured format.
    Ensures all tool calls are captured when the model executes multiple actions.
    """

    # Using \s* instead of literal \n for robustness against variations
    # in model formatting (Issue #989).
    PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<function_name>.*?)\s*```json\s*(?P<function_arguments>.*?)\s*```\s*<｜tool▁call▁end｜>",
        re.DOTALL,
    )

    def _on_parse_error(self, exc: Exception) -> None:
        logger.error(f"Error parsing DeepSeek V3 tool calls: {exc}")
