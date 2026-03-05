"""
MCE — Layer 1 Pruner Tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.squeeze.layer1_pruner import Layer1Pruner


@pytest.fixture
def pruner() -> Layer1Pruner:
    return Layer1Pruner(max_array_length=5, max_string_length=200)


class TestNullRemoval:
    def test_strips_null_values(self, pruner: Layer1Pruner):
        data = {"a": 1, "b": None, "c": "hello", "d": None}
        result = pruner.prune(data)
        assert "b" not in result
        assert "d" not in result
        assert result["a"] == 1

    def test_nested_null_removal(self, pruner: Layer1Pruner):
        data = {"outer": {"a": 1, "b": None}}
        result = pruner.prune(data)
        assert "b" not in result["outer"]


class TestArrayTruncation:
    def test_truncates_long_array(self, pruner: Layer1Pruner):
        data = list(range(20))
        result = pruner.prune(data)
        assert len(result) == 5
        assert any("truncated" in n for n in pruner.notices)

    def test_preserves_short_array(self, pruner: Layer1Pruner):
        data = [1, 2, 3]
        result = pruner.prune(data)
        assert len(result) == 3
        assert not pruner.notices


class TestHTMLConversion:
    def test_html_to_markdown(self, pruner: Layer1Pruner):
        html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        result = pruner.prune(html)
        assert "<html>" not in result
        assert "Title" in result

    def test_non_html_passthrough(self, pruner: Layer1Pruner):
        text = "Just a plain text string"
        result = pruner.prune(text)
        assert result == text


class TestBase64Stripping:
    def test_strips_base64_blobs(self, pruner: Layer1Pruner):
        blob = "A" * 150  # Long enough to match pattern
        text = f"before data:image/png;base64,{blob} after"
        result = pruner.prune(text)
        assert "base64 blob removed" in result or any("base64" in n for n in pruner.notices)

    def test_preserves_short_strings(self, pruner: Layer1Pruner):
        text = "normal short text"
        result = pruner.prune(text)
        assert result == text


class TestStringTruncation:
    def test_truncates_long_string(self, pruner: Layer1Pruner):
        # Use words with spaces so it won't match base64 regex
        long_text = "hello world " * 50  # ~600 chars
        result = pruner.prune(long_text)
        assert len(result) <= 200
        assert any("truncated" in n for n in pruner.notices)


class TestWhitespace:
    def test_normalizes_whitespace(self, pruner: Layer1Pruner):
        text = "line1\n\n\n\n\nline2"
        result = pruner.prune(text)
        assert "\n\n\n" not in result


class TestNotices:
    def test_notices_cleared_between_calls(self, pruner: Layer1Pruner):
        pruner.prune(list(range(20)))
        assert len(pruner.notices) > 0
        pruner.prune("short")
        assert len(pruner.notices) == 0
