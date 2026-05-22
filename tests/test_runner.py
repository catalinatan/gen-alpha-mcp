import pytest
from pathlib import Path
from gab.runner import render_prompt, _response_text, resolve_input_path


def test_render_prompt_fills_fields():
    template = "Explain {expression} to a {target_audience}."
    case = {"expression": "no cap", "target_audience": "boomer"}
    assert render_prompt(template, case) == "Explain no cap to a boomer."


def test_render_prompt_missing_field_raises():
    template = "Explain {expression} in {context}."
    with pytest.raises(KeyError, match="context"):
        render_prompt(template, {"expression": "slay"})


def test_render_prompt_extra_fields_ignored():
    template = "Explain {expression}."
    result = render_prompt(template, {"expression": "rizz", "extra": "ignored"})
    assert result == "Explain rizz."


def test_render_prompt_preserves_unknown_braces():
    # braces that aren't identifiers (e.g. format specs) are left alone
    template = "Score: {score:.2f}"
    result = render_prompt(template, {})
    assert "{score:.2f}" in result


def _make_response(*blocks):
    return type("R", (), {"content": list(blocks)})()


def test_response_text_dict_blocks():
    response = _make_response(
        {"type": "text", "text": "hello "},
        {"type": "tool_use", "id": "x"},
    )
    assert _response_text(response) == "hello"


def test_response_text_object_blocks():
    block = type("B", (), {"type": "text", "text": " world"})()
    assert _response_text(_make_response(block)) == "world"


def test_response_text_mixed_blocks():
    obj_block = type("B", (), {"type": "text", "text": "hello "})()
    response = _make_response(obj_block, {"type": "text", "text": "world"})
    assert _response_text(response) == "hello world"


def test_response_text_empty():
    assert _response_text(_make_response()) == ""


def test_resolve_input_path_existing(tmp_path):
    f = tmp_path / "golden.json"
    f.write_text("[]")
    assert resolve_input_path(str(f)) == f


def test_resolve_input_path_returns_path_object():
    result = resolve_input_path("nonexistent.json")
    assert isinstance(result, Path)
