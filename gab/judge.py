import json
import re
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


JUDGE_PROMPT = """You are an impartial evaluator. Score the following AI output \
for a Gen Alpha slang explanation task.

Expression: {expression}
Example usage: {context}
Target audience: {target_audience}
Criteria to check:
{criteria}
Actual output: {output}

Return a score from 1-5 where:
5 = fully meets all criteria
4 = meets most criteria, minor issues
3 = partially meets criteria
2 = significant issues
1 = fails criteria entirely

Be strict. A demo-quality answer that misses one criterion gets max 3.

Return only valid JSON with this shape:
{{
  "score": 5,
  "reasoning": "brief explanation of the score",
  "criteria_met": ["criterion text"],
  "criteria_failed": ["criterion text"]
}}"""


def _response_text(response) -> str:
    return "".join(
        block.get("text", "") if isinstance(block, dict) else block.text
        for block in response.content
        if (isinstance(block, dict) and block.get("type") == "text")
        or (getattr(block, "type", None) == "text" and hasattr(block, "text"))
    ).strip()


def _parse_judge_result(text: str) -> JudgeResult:
    def validate_json(raw: str) -> JudgeResult:
        if hasattr(JudgeResult, "model_validate_json"):
            return JudgeResult.model_validate_json(raw)
        return JudgeResult.parse_raw(raw)

    def validate_obj(obj: dict) -> JudgeResult:
        if hasattr(JudgeResult, "model_validate"):
            return JudgeResult.model_validate(obj)
        return JudgeResult.parse_obj(obj)

    try:
        return validate_json(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Judge response did not contain JSON: {text}") from None
        try:
            return validate_obj(json.loads(match.group(0)))
        except ValueError as exc:
            raise ValueError(
                f"Judge response was not valid JudgeResult JSON: {text}"
            ) from exc


def judge(
    expression: str,
    context: str,
    target_audience: str,
    criteria: list[str],
    output: str,
) -> JudgeResult:
    response = get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    expression=expression,
                    context=context,
                    target_audience=target_audience,
                    criteria="\n".join(f"- {criterion}" for criterion in criteria),
                    output=output,
                ),
            }
        ],
    )
    return _parse_judge_result(_response_text(response))
