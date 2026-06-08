from pathlib import Path

import pytest

from gab.runner import (
    HAIKU,
    SONNET,
    _response_text,
    estimate_cost_usd,
    pick_model,
    retrieve_few_shot_cases,
    render_prompt,
    resolve_input_path,
)


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


def _case(**overrides):
    base = {
        "expression": "no cap",
        "target_audience": "parent",
        "criteria": ["a", "b", "c", "d"],
    }
    base.update(overrides)
    return base


def test_pick_model_simple_case_uses_haiku():
    assert pick_model(_case()) == HAIKU


def test_pick_model_multiword_expression_uses_sonnet():
    assert pick_model(_case(expression="what the sigma")) == SONNET


def test_pick_model_many_criteria_uses_sonnet():
    assert pick_model(_case(criteria=["a", "b", "c", "d", "e"])) == SONNET


def test_pick_model_long_audience_uses_sonnet():
    assert pick_model(_case(target_audience="x" * 60)) == SONNET


def test_pick_model_handles_missing_fields():
    assert pick_model({}) == HAIKU


def test_estimate_cost_haiku_cheaper_than_sonnet():
    haiku = estimate_cost_usd(HAIKU, 1000, 500)
    sonnet = estimate_cost_usd(SONNET, 1000, 500)
    assert haiku < sonnet
    assert haiku == pytest.approx((1000 * 1.0 + 500 * 5.0) / 1_000_000)
    assert sonnet == pytest.approx((1000 * 3.0 + 500 * 15.0) / 1_000_000)


def test_estimate_cost_unknown_model_returns_zero():
    assert estimate_cost_usd("claude-mystery-9", 1000, 500) == 0.0


def test_retrieve_few_shot_cases_delegates_to_vector_store(monkeypatch):
    calls = []

    def fake_relevant_cases(**kwargs):
        calls.append(kwargs)
        return [{"id": "ga_002"}]

    monkeypatch.setattr("gab.runner.relevant_cases", fake_relevant_cases)
    result = retrieve_few_shot_cases(
        {"id": "ga_001", "expression": "no cap"},
        query="Explain no cap",
        top_k=2,
    )

    assert result == [{"id": "ga_002"}]
    assert calls[0]["query"] == "Explain no cap"
    assert calls[0]["top_k"] == 2
    assert calls[0]["exclude_case_id"] == "ga_001"


def test_retrieve_few_shot_cases_disabled_when_top_k_zero(monkeypatch):
    def fail_if_called(**_kwargs):
        raise AssertionError("retrieval should not run")

    monkeypatch.setattr("gab.runner.relevant_cases", fail_if_called)
    assert retrieve_few_shot_cases({}, query="unused", top_k=0) == []
