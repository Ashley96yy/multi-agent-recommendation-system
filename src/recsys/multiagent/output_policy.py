"""Opt-in, deterministic *format-only* normalization for model outputs.

No prompt change, new generation, semantic repair or source claim verification.
The historical Stage 3.1 experiment used strict mode and remains unchanged.
"""

import json
import re
from typing import Literal

FormatPolicy = Literal["strict", "safe_json_normalization"]

_EXACT_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\s*\Z",
    flags=re.IGNORECASE,
)


def normalize_json_response(raw: str, policy: FormatPolicy) -> tuple[str, list[str]]:
    """Only remove a complete JSON fence and fill an unambiguous missing flag.

    Missing no_suitable_match=False is inferred only for a nonempty list
    of recommendations and never overwrites an explicit value. Rejected JSON,
    missing citations, product IDs and natural-language reasons are untouched.
    """
    if policy not in ("strict", "safe_json_normalization"):
        raise ValueError("unknown output format policy")
    if not isinstance(raw, str):
        raise TypeError("model output must be text")
    if policy == "strict":
        return raw, []

    text = raw
    actions: list[str] = []
    match = _EXACT_FENCE.fullmatch(text)
    if match is not None:
        text = match.group("body")
        actions.append("strip_whole_json_fence")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return text, actions
    if (
        isinstance(document, dict)
        and "no_suitable_match" not in document
        and isinstance(document.get("recommendations"), list)
        and bool(document["recommendations"])
    ):
        document["no_suitable_match"] = False
        text = json.dumps(document, ensure_ascii=False)
        actions.append("set_missing_no_suitable_match_false")
    return text, actions
