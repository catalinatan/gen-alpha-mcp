from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gab.judge import JudgeResult, judge


def _tool_use_response(score, reasoning, criteria_met, criteria_failed):
    block = SimpleNamespace(
        type="tool_use",
        name="submit_judgment",
        input={
            "score": score,
            "reasoning": reasoning,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
        },
    )
    return SimpleNamespace(content=[block])


def test_judge_extracts_from_tool_use():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _tool_use_response(
        score=4,
        reasoning="solid explanation",
        criteria_met=["clear language"],
        criteria_failed=[],
    )
    with patch("gab.judge.get_client", return_value=mock_client):
        result = judge(
            expression="no cap",
            context="That's no cap fr fr",
            target_audience="parents",
            criteria=["uses plain English", "avoids jargon"],
            output="It means 'no lie' or 'seriously'.",
        )
    assert isinstance(result, JudgeResult)
    assert result.score == 4
    assert result.reasoning == "solid explanation"
    assert result.criteria_met == ["clear language"]
    assert result.criteria_failed == []


def test_judge_handles_score_only_tool_use():
    mock_client = MagicMock()
    block = SimpleNamespace(type="tool_use", name="submit_judgment", input={"score": 3})
    mock_client.messages.create.return_value = SimpleNamespace(content=[block])

    with patch("gab.judge.get_client", return_value=mock_client):
        result = judge(
            expression="no cap",
            context="That's no cap fr fr",
            target_audience="parents",
            criteria=["uses plain English", "avoids jargon"],
            output="It means 'no lie' or 'seriously'.",
        )

    assert result.score == 3
    assert result.reasoning == "Judge returned a score without reasoning."
    assert result.criteria_met == []
    assert result.criteria_failed == []


def test_judge_raises_if_no_tool_use_block():
    mock_client = MagicMock()
    text_block = SimpleNamespace(type="text", text="Here's my evaluation: score 4")
    mock_client.messages.create.return_value = SimpleNamespace(content=[text_block])
    with patch("gab.judge.get_client", return_value=mock_client):
        with pytest.raises(ValueError, match="submit_judgment"):
            judge(
                expression="slay",
                context="you slay today",
                target_audience="adults",
                criteria=["accurate"],
                output="it means to do well",
            )


@pytest.mark.parametrize("score", [0, 6])
def test_score_out_of_bounds(score):
    with pytest.raises(Exception):
        JudgeResult(score=score, reasoning="x", criteria_met=[], criteria_failed=[])


def test_score_boundary_valid():
    low = JudgeResult(score=1, reasoning="x", criteria_met=[], criteria_failed=[])
    high = JudgeResult(score=5, reasoning="x", criteria_met=[], criteria_failed=[])
    assert low.score == 1
    assert high.score == 5
