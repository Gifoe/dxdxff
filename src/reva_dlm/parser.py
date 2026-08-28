"""Canonical numeric answer parsing for the GSM8K trajectory audit.

The parser deliberately has one implementation for ground truth, intermediate
states, and final states.  It prefers the last explicit ``Answer:`` or
``####`` marker.  When neither marker exists, it follows the Prophet fallback
and uses the last numeric token in the text.

Numeric comparison is performed with :class:`decimal.Decimal`; converting to
``float`` here would make correctness depend on binary floating-point
rounding.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Final


PARSER_NAME: Final[str] = "gsm8k_decimal"
PARSER_VERSION: Final[str] = "1.1.0"

# A valid number is either comma-free or uses groups of exactly three digits.
# The prefix accepts 12, -12, +12, $12, -$12, and $-12.  Word/dot/comma
# boundaries prevent silently accepting only a suffix of malformed input such
# as ``12,34`` or ``1.2.3``.
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.,])"
    r"(?P<prefix>(?:[+-]\s*\$?|\$\s*[+-]?))?\s*"
    r"(?P<number>(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))"
    r"(?!\w|\.\d|,\d)"
)
_MARKER_RE: Final[re.Pattern[str]] = re.compile(r"(?i)(?:####|answer\s*:)")


def _match_to_decimal(match: re.Match[str]) -> Decimal | None:
    prefix = match.group("prefix") or ""
    sign = "-" if "-" in prefix else ""
    normalized = sign + match.group("number").replace(",", "")
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def parse_gsm8k_answer(text: object) -> Decimal | None:
    """Extract a finite GSM8K numeric answer as a ``Decimal``.

    Extraction rules are deterministic:

    1. If an explicit ``Answer:`` or ``####`` marker occurs, inspect the text
       after the *last* marker and take its first valid number.
    2. If that marker has only trailing whitespace (an empty answer line),
       ignore it and use the numeric fallback on the preceding rationale. This
       is the parser-v1.1 project protocol revision; it is not represented as
       an exact emulation of every Prophet whitespace edge case.
    3. Otherwise, take the last valid number in the whole text (the Prophet
       collection script's fallback behaviour).
    4. Return ``None`` when no valid number exists.  Two missing answers never
       compare as correct via :func:`answers_equal`.

    A non-empty but non-numeric marker value such as ``Answer: unknown`` is a
    parse failure, not a license to reuse a rationale number.
    """

    if text is None:
        return None
    if isinstance(text, Decimal):
        return text if text.is_finite() else None

    source = str(text)
    markers = list(_MARKER_RE.finditer(source))
    if markers:
        suffix = source[markers[-1].end() :]
        match = _NUMBER_RE.search(suffix)
        if match is not None:
            return _match_to_decimal(match)
        if suffix.strip():
            return None
        source = source[: markers[-1].start()]

    matches = list(_NUMBER_RE.finditer(source))
    return _match_to_decimal(matches[-1]) if matches else None


def canonical_decimal(value: Decimal | None) -> str | None:
    """Return a non-exponential, trailing-zero-free representation."""

    if value is None or not value.is_finite():
        return None
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def extract_gsm8k_answer(text: object) -> str | None:
    """Extract an answer and return its canonical decimal string."""

    return canonical_decimal(parse_gsm8k_answer(text))


def answers_equal(prediction: object, ground_truth: object) -> bool:
    """Compare prediction and ground truth using the canonical parser."""

    predicted = parse_gsm8k_answer(prediction)
    expected = parse_gsm8k_answer(ground_truth)
    return predicted is not None and expected is not None and predicted == expected


# Small compatibility aliases keep analysis scripts terse while retaining a
# single parser implementation.
parse_answer = parse_gsm8k_answer
is_correct = answers_equal


__all__ = [
    "PARSER_NAME",
    "PARSER_VERSION",
    "answers_equal",
    "canonical_decimal",
    "extract_gsm8k_answer",
    "is_correct",
    "parse_answer",
    "parse_gsm8k_answer",
]
