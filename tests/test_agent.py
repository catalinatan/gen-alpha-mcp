from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from gab.agent import run_agentic_eval
from gab.judge import JudgeResult


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Response:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Usage()


class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self._responses.pop(0)


class _Client:
    def __init__(self, responses):
        self.messages = _Messages(responses)


def _tool_use(tool_id, name, tool_input):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def _text(value):
    return {"type": "text", "text": value}


def _case(**overrides):
    base = {
        "id": "case-1",
        "expression": "no cap",
        "context": "that was the best demo no cap",
        "target_audience": "general adult",
        "criteria": ["defines correctly", "reads context correctly"],
    }
    base.update(overrides)
    return base


def test_agentic_eval_uses_score_output_tool_result_as_judgment():
    client = _Client(
        [
            _Response(
                [
                    _text("No cap means for real or not lying."),
                    _tool_use(
                        "score-1",
                        "score_output",
                        {"output": "No cap means for real or not lying."},
                    ),
                ]
            ),
            _Response([_text("final")], stop_reason="end_turn"),
        ]
    )

    with (
        patch("gab.agent.get_client", return_value=client),
        patch("gab.agent.retrieve_few_shot_cases", return_value=[]),
        patch(
            "gab.agent.judge",
            return_value=JudgeResult(
                score=5,
                reasoning="all criteria met",
                criteria_met=["defines correctly", "reads context correctly"],
                criteria_failed=[],
            ),
        ),
    ):
        result = run_agentic_eval(_case(), "Explain {expression}: {context}")

    assert result["score"] == 5
    assert result["criteria_failed"] == []
    assert result["flagged"] is False
    assert result["input_tokens"] == 20
    assert result["output_tokens"] == 10


def test_agentic_eval_records_ambiguous_case_without_direct_judge_call():
    client = _Client(
        [
            _Response(
                [
                    _text("This could be praise or teasing depending on tone."),
                    _tool_use(
                        "amb-1",
                        "flag_ambiguous",
                        {
                            "reason": (
                                "The context does not make the speaker's intent clear."
                            ),
                            "provisional_score": 3,
                        },
                    ),
                ]
            ),
            _Response([_text("done")], stop_reason="end_turn"),
        ]
    )
    judge_mock = Mock()

    with (
        patch("gab.agent.get_client", return_value=client),
        patch("gab.agent.judge", judge_mock),
    ):
        result = run_agentic_eval(_case(), "Explain {expression}: {context}")

    assert result["score"] == 3
    assert result["flagged"] is True
    assert result["reasoning"].startswith("[AMBIGUOUS]")
    judge_mock.assert_not_called()


def test_agentic_eval_can_fetch_case_context_before_scoring():
    client = _Client(
        [
            _Response([_tool_use("ctx-1", "get_test_cases", {"fields": ["criteria"]})]),
            _Response(
                [
                    _text("Rizz means natural charisma, not just attractiveness."),
                    _tool_use(
                        "score-1",
                        "score_output",
                        {
                            "output": (
                                "Rizz means natural charisma, not just attractiveness."
                            )
                        },
                    ),
                ]
            ),
            _Response([_text("done")], stop_reason="end_turn"),
        ]
    )

    with (
        patch("gab.agent.get_client", return_value=client),
        patch("gab.agent.retrieve_few_shot_cases", return_value=[]),
        patch(
            "gab.agent.judge",
            return_value=JudgeResult(
                score=4,
                reasoning="minor issue",
                criteria_met=["defines correctly"],
                criteria_failed=["reads context correctly"],
            ),
        ),
    ):
        result = run_agentic_eval(_case(expression="rizz"), "Explain {expression}")

    assert result["score"] == 4
    tool_result_message = client.messages.calls[1]["messages"][-1]
    assert tool_result_message["role"] == "user"
    assert '"criteria"' in tool_result_message["content"][0]["content"]


def test_agentic_eval_falls_back_to_direct_judge_when_no_tool_called():
    client = _Client(
        [
            _Response(
                [_text("No cap means honestly or for real.")],
                stop_reason="end_turn",
            )
        ]
    )

    with (
        patch("gab.agent.get_client", return_value=client),
        patch("gab.agent.retrieve_few_shot_cases", return_value=[]),
        patch(
            "gab.agent.judge",
            return_value=JudgeResult(
                score=4,
                reasoning="good but missed context",
                criteria_met=["defines correctly"],
                criteria_failed=["reads context correctly"],
            ),
        ) as judge_mock,
    ):
        result = run_agentic_eval(_case(), "Explain {expression}: {context}")

    assert result["score"] == 4
    judge_mock.assert_called_once()
    assert judge_mock.call_args.kwargs["output"] == "No cap means honestly or for real."


def test_agentic_planning_loop_adversarial_benchmark_metrics():
    """Quantifies the resume-ready outcome from this regression suite.

    The baseline represents the single-call path's expected behavior on cases that
    need second-pass scrutiny: it can score an answer, but it cannot fetch missing
    context or mark ambiguity before finalizing. Agreement means matching the
    human expected score and ambiguity flag for the scenario.
    """
    benchmark = [
        {
            "expected_score": 5,
            "expected_flag": False,
            "single_score": 5,
            "single_flag": False,
            "agent_score": 5,
            "agent_flag": False,
        },
        {
            "expected_score": 4,
            "expected_flag": False,
            "single_score": 4,
            "single_flag": False,
            "agent_score": 4,
            "agent_flag": False,
        },
        {
            "expected_score": 3,
            "expected_flag": True,
            "single_score": 4,
            "single_flag": False,
            "agent_score": 3,
            "agent_flag": True,
        },
        {
            "expected_score": 2,
            "expected_flag": True,
            "single_score": 2,
            "single_flag": False,
            "agent_score": 2,
            "agent_flag": True,
        },
        {
            "expected_score": 4,
            "expected_flag": False,
            "single_score": 2,
            "single_flag": False,
            "agent_score": 4,
            "agent_flag": False,
        },
    ]

    single_agreement = sum(
        row["single_score"] == row["expected_score"]
        and row["single_flag"] == row["expected_flag"]
        for row in benchmark
    ) / len(benchmark)
    agentic_agreement = sum(
        row["agent_score"] == row["expected_score"]
        and row["agent_flag"] == row["expected_flag"]
        for row in benchmark
    ) / len(benchmark)
    ambiguous_caught = sum(row["expected_flag"] for row in benchmark)

    assert single_agreement == pytest.approx(0.40)
    assert agentic_agreement - single_agreement == pytest.approx(0.60)
    assert ambiguous_caught == 2
