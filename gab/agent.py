import json
from typing import Any

from gab.judge import JudgeResult, judge
from gab.runner import (
    _usage_tokens,
    estimate_cost_usd,
    get_client,
    pick_model,
    render_prompt,
    resolve_input_path,
)
from gab.store import ResultStore, validate_version

MAX_STEPS = 5

_GET_TEST_CASES_TOOL: dict[str, Any] = {
    "name": "get_test_cases",
    "description": "Retrieve fields from the current test case for evaluation context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Case fields to return (expression, context, target_audience, criteria). Defaults to all if omitted.",
            }
        },
        "required": [],
    },
}

_SCORE_OUTPUT_TOOL: dict[str, Any] = {
    "name": "score_output",
    "description": (
        "Score a candidate output against the evaluation criteria. "
        "Returns a structured judgment with score (1-5), reasoning, and per-criterion results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "output": {
                "type": "string",
                "description": "The candidate output text to score.",
            }
        },
        "required": ["output"],
    },
}

_FLAG_AMBIGUOUS_TOOL: dict[str, Any] = {
    "name": "flag_ambiguous",
    "description": "Flag this case as ambiguous, recording a reason and a provisional score.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why this case is ambiguous or hard to score definitively.",
            },
            "provisional_score": {
                "type": "integer",
                "description": "Best-effort score 1-5 given the ambiguity.",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": ["reason", "provisional_score"],
    },
}

_TOOLS = [_GET_TEST_CASES_TOOL, _SCORE_OUTPUT_TOOL, _FLAG_AMBIGUOUS_TOOL]


def _dispatch_tool(
    tool_name: str, tool_input: dict, case: dict, candidate_output: str
) -> str:
    if tool_name == "get_test_cases":
        requested = tool_input.get("fields") or list(case.keys())
        return json.dumps({f: case.get(f) for f in requested if f in case})

    if tool_name == "score_output":
        output = tool_input.get("output") or candidate_output
        result = judge(
            expression=case["expression"],
            context=case["context"],
            target_audience=case["target_audience"],
            criteria=case["criteria"],
            output=output,
        )
        return json.dumps(
            {
                "score": result.score,
                "reasoning": result.reasoning,
                "criteria_met": result.criteria_met,
                "criteria_failed": result.criteria_failed,
            }
        )

    if tool_name == "flag_ambiguous":
        return json.dumps(
            {
                "flagged": True,
                "reason": tool_input.get("reason", ""),
                "provisional_score": tool_input.get("provisional_score", 3),
            }
        )

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _extract_blocks(
    response: Any,
) -> tuple[list[dict], list[tuple[str, str, dict]]]:
    """Split response.content into (assistant_content_list, tool_use_blocks)."""
    assistant_content: list[dict] = []
    tool_use_blocks: list[tuple[str, str, dict]] = []

    for block in response.content:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "text":
            text = getattr(block, "text", None) or (
                block.get("text", "") if isinstance(block, dict) else ""
            )
            assistant_content.append({"type": "text", "text": text})
        elif block_type == "tool_use":
            tool_id = getattr(block, "id", None) or block.get("id", "")
            tool_name = getattr(block, "name", None) or block.get("name", "")
            tool_input = getattr(block, "input", None) or block.get("input", {})
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            tool_use_blocks.append((tool_id, tool_name, tool_input))

    return assistant_content, tool_use_blocks


def run_agentic_eval(case: dict, prompt_template: str) -> dict:
    """ReAct-style eval loop: agent reasons → picks a tool → observes result → decides next action.

    The agent receives the task prompt and can call:
      - score_output(output) — invokes the judge on a candidate answer
      - get_test_cases(fields?) — retrieves case fields for context
      - flag_ambiguous(reason, provisional_score) — marks hard-to-score cases
    When stop_reason == "end_turn" the loop exits and the last score_output
    result is used as the final judgment.
    """
    filled = render_prompt(prompt_template, case)
    messages: list[dict] = [{"role": "user", "content": filled}]

    model = pick_model(case)
    total_input_tokens = 0
    total_output_tokens = 0
    candidate_output = ""
    last_score_result: dict | None = None
    flagged: dict | None = None

    for _ in range(MAX_STEPS):
        response = get_client().messages.create(
            model=model,
            max_tokens=1000,
            tools=_TOOLS,
            messages=messages,
        )

        inp, out = _usage_tokens(response)
        total_input_tokens += inp
        total_output_tokens += out

        assistant_content, tool_use_blocks = _extract_blocks(response)

        for block in assistant_content:
            if block.get("type") == "text" and block.get("text", "").strip():
                candidate_output = block["text"].strip()

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            break

        if tool_use_blocks:
            tool_results = []
            for tool_id, tool_name, tool_input in tool_use_blocks:
                result_str = _dispatch_tool(tool_name, tool_input, case, candidate_output)
                result_data = json.loads(result_str)
                if tool_name == "score_output":
                    last_score_result = result_data
                elif tool_name == "flag_ambiguous":
                    flagged = result_data
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

    judgment = _build_judgment(case, candidate_output, last_score_result, flagged)

    return {
        "id": case["id"],
        "expression": case["expression"],
        "output": candidate_output,
        "score": judgment.score,
        "reasoning": judgment.reasoning,
        "criteria_met": judgment.criteria_met,
        "criteria_failed": judgment.criteria_failed,
        "model_used": model,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost_usd": estimate_cost_usd(model, total_input_tokens, total_output_tokens),
        "agentic": True,
        "flagged": flagged is not None,
    }


def _build_judgment(
    case: dict,
    candidate_output: str,
    last_score_result: dict | None,
    flagged: dict | None,
) -> JudgeResult:
    if last_score_result:
        return JudgeResult(
            score=last_score_result["score"],
            reasoning=last_score_result["reasoning"],
            criteria_met=last_score_result.get("criteria_met", []),
            criteria_failed=last_score_result.get("criteria_failed", []),
        )
    if flagged:
        return JudgeResult(
            score=flagged["provisional_score"],
            reasoning=f"[AMBIGUOUS] {flagged['reason']}",
            criteria_met=[],
            criteria_failed=[],
        )
    # Neither tool was called — fall back to a direct judge call
    return judge(
        expression=case["expression"],
        context=case["context"],
        target_audience=case["target_audience"],
        criteria=case["criteria"],
        output=candidate_output,
    )


def run_agentic_set(golden_set_path: str, prompt_path: str, version: str) -> list[dict]:
    """Run the agentic eval loop over every case in the golden set."""
    validate_version(version)

    import json as _json

    golden = _json.loads(
        resolve_input_path(golden_set_path).read_text(encoding="utf-8")
    )
    if not isinstance(golden, list):
        raise ValueError("Golden set must be a JSON array of test cases.")

    prompt_template = resolve_input_path(prompt_path).read_text(encoding="utf-8")

    results = []
    for case in golden:
        if not isinstance(case, dict):
            raise ValueError("Each golden set case must be a JSON object.")
        required_fields = {"id", "expression", "context", "target_audience", "criteria"}
        missing = sorted(required_fields - case.keys())
        if missing:
            raise KeyError(f"Golden set case is missing fields: {', '.join(missing)}")
        results.append(run_agentic_eval(case, prompt_template))

    store = ResultStore()
    for result in results:
        store.save(
            version=version,
            case_id=result["id"],
            score=result["score"],
            reasoning=result["reasoning"],
            model_used=result["model_used"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=result["cost_usd"],
        )
    return results
