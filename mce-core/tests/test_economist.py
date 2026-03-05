"""
MCE — Token Economist Tests
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure mce-core is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.token_economist import TokenEconomist, Action
from schemas.mce_config import TokenLimitsConfig


@pytest.fixture
def economist() -> TokenEconomist:
    return TokenEconomist(TokenLimitsConfig(safe_limit=100, squeeze_trigger=200, absolute_max=500))


class TestTokenCounting:
    def test_empty_string(self, economist: TokenEconomist):
        assert economist.count_tokens("") == 0

    def test_simple_string(self, economist: TokenEconomist):
        count = economist.count_tokens("hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_count_any_dict(self, economist: TokenEconomist):
        payload = {"key": "value", "nested": {"a": 1}}
        count = economist.count_any(payload)
        assert count > 0

    def test_count_any_list(self, economist: TokenEconomist):
        payload = [1, 2, 3, "hello"]
        count = economist.count_any(payload)
        assert count > 0


class TestBudgetEvaluation:
    def test_under_safe_limit(self, economist: TokenEconomist):
        report = economist.evaluate("hi")
        assert not report.is_over_budget
        assert report.recommended_action == Action.PASS_THROUGH

    def test_over_safe_limit(self, economist: TokenEconomist):
        # Generate a payload that exceeds 100 tokens
        big_payload = " ".join(["word"] * 200)
        report = economist.evaluate(big_payload)
        assert report.is_over_budget
        assert report.recommended_action == Action.SQUEEZE

    def test_dict_over_budget(self, economist: TokenEconomist):
        big_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        report = economist.evaluate(big_dict)
        assert report.is_over_budget
        assert report.recommended_action == Action.SQUEEZE

    def test_report_fields(self, economist: TokenEconomist):
        report = economist.evaluate("test")
        assert report.safe_limit == 100
        assert report.squeeze_trigger == 200
        assert isinstance(report.token_count, int)


class TestSerialization:
    def test_serialize_string(self, economist: TokenEconomist):
        assert economist.serialize("hello") == "hello"

    def test_serialize_dict(self, economist: TokenEconomist):
        result = economist.serialize({"a": 1})
        parsed = json.loads(result)
        assert parsed == {"a": 1}
