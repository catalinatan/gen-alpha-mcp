import pytest
from gab.judge import _parse_judge_result, JudgeResult

_VALID = '{"score": 4, "reasoning": "solid", "criteria_met": ["clear"], "criteria_failed": []}'


def test_parse_clean_json():
    result = _parse_judge_result(_VALID)
    assert isinstance(result, JudgeResult)
    assert result.score == 4
    assert result.reasoning == "solid"
    assert result.criteria_met == ["clear"]
    assert result.criteria_failed == []


def test_parse_json_embedded_in_prose():
    text = f"Here is my evaluation:\n{_VALID}\nThat's all."
    result = _parse_judge_result(text)
    assert result.score == 4


def test_parse_no_json_raises():
    with pytest.raises(ValueError):
        _parse_judge_result("no json here at all")


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError):
        _parse_judge_result("{not: valid json}")


@pytest.mark.parametrize("score", [0, 6])
def test_score_out_of_bounds(score):
    with pytest.raises(Exception):
        JudgeResult(score=score, reasoning="x", criteria_met=[], criteria_failed=[])


def test_score_boundary_valid():
    low = JudgeResult(score=1, reasoning="x", criteria_met=[], criteria_failed=[])
    high = JudgeResult(score=5, reasoning="x", criteria_met=[], criteria_failed=[])
    assert low.score == 1
    assert high.score == 5
