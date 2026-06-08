from typing import Any

from pydantic import BaseModel, Field

client: Any | None = None


def get_client() -> Any:
    global client
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    return client


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str
    criteria_met: list[str] = Field(default_factory=list)
    criteria_failed: list[str] = Field(default_factory=list)


_SUBMIT_JUDGMENT_TOOL: dict[str, Any] = {
    "name": "submit_judgment",
    "description": (
        "Submit the structured evaluation judgment for the candidate output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": (
                    "Score from 1 (fails entirely) to 5 (fully meets all criteria)."
                ),
                "minimum": 1,
                "maximum": 5,
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the score.",
            },
            "criteria_met": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of criteria the output satisfied.",
            },
            "criteria_failed": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of criteria the output did not satisfy.",
            },
        },
        "required": ["score", "reasoning", "criteria_met", "criteria_failed"],
    },
}

JUDGE_PROMPT = """You are an impartial evaluator. Score the following AI output \
for a Gen Alpha slang explanation task.

Expression: {expression}
Example usage: {context}
Target audience: {target_audience}
Criteria to check:
{criteria}

Retrieved golden cases for calibration:
{few_shot_context}
Actual output: {output}

Rate from 1-5 where:
5 = fully meets all criteria
4 = meets most criteria, minor issues
3 = partially meets criteria
2 = significant issues
1 = fails criteria entirely

Be strict. A demo-quality answer that misses one criterion gets max 3.
Call submit_judgment with your evaluation."""


def format_few_shot_cases(cases: list[dict] | None) -> str:
    if not cases:
        return "(none)"

    blocks = []
    for idx, case in enumerate(cases, start=1):
        criteria = "\n".join(
            f"     - {criterion}" for criterion in case.get("criteria", [])
        )
        blocks.append(
            f"{idx}. Expression: {case.get('expression', '')}\n"
            f"   Example usage: {case.get('context', '')}\n"
            f"   Target audience: {case.get('target_audience', '')}\n"
            f"   Criteria:\n{criteria}"
        )
    return "\n\n".join(blocks)


def judge(
    expression: str,
    context: str,
    target_audience: str,
    criteria: list[str],
    output: str,
    few_shot_cases: list[dict] | None = None,
) -> JudgeResult:
    response = get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        tools=[_SUBMIT_JUDGMENT_TOOL],
        tool_choice={"type": "tool", "name": "submit_judgment"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    expression=expression,
                    context=context,
                    target_audience=target_audience,
                    criteria="\n".join(f"- {criterion}" for criterion in criteria),
                    few_shot_context=format_few_shot_cases(few_shot_cases),
                    output=output,
                ),
            }
        ],
    )
    for block in response.content:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "tool_use":
            tool_input = getattr(block, "input", None) or (
                block.get("input", {}) if isinstance(block, dict) else {}
            )
            tool_input = {
                "reasoning": "Judge returned a score without reasoning.",
                "criteria_met": [],
                "criteria_failed": [],
                **tool_input,
            }
            return JudgeResult.model_validate(tool_input)
    raise ValueError("Judge model did not call submit_judgment tool")
