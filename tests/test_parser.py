from decimal import Decimal

import pytest

from reva_dlm.parser import (
    answers_equal,
    canonical_decimal,
    extract_gsm8k_answer,
    parse_gsm8k_answer,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("calculation\n#### 1,234.50", Decimal("1234.50")),
        ("Answer: -$1,234.50 dollars.", Decimal("-1234.50")),
        ("Answer: $+0.75, after checking.", Decimal("0.75")),
        ("#### +.5 trailing text", Decimal("0.5")),
        ("The values were 3, 7, and -12.25 widgets.", Decimal("-12.25")),
        ("Answer: -1,000.00 trailing text containing 999", Decimal("-1000.00")),
        ("No numeric result is present.", None),
        (None, None),
    ],
)
def test_canonical_decimal_parser(text, expected):
    assert parse_gsm8k_answer(text) == expected


def test_last_explicit_marker_controls_extraction():
    assert parse_gsm8k_answer("#### 12\nCorrection. Answer: 13 widgets") == Decimal("13")
    assert parse_gsm8k_answer("The rationale used 42. Answer: unknown") is None
    assert parse_gsm8k_answer("The rationale concludes 68.\nAnswer:") == Decimal("68")
    assert parse_gsm8k_answer("The rationale concludes 34.\nAnswer:   ") == Decimal("34")


def test_malformed_comma_grouping_is_not_partially_accepted():
    assert parse_gsm8k_answer("Answer: 12,34") is None


def test_answer_equality_never_treats_two_parse_failures_as_correct():
    assert answers_equal("Answer: 1,200.00", "work\n#### 1200")
    assert answers_equal("Answer: -0.0", "#### 0")
    assert not answers_equal("unknown", "also unknown")
    assert not answers_equal("Answer: 2", "#### 3")


def test_canonical_string_format():
    assert canonical_decimal(Decimal("1234.5000")) == "1234.5"
    assert extract_gsm8k_answer("Answer: -0.000") == "0"
    assert extract_gsm8k_answer("Answer: +001.2500 trailing") == "1.25"
