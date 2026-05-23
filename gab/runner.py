import json
from pathlib import Path
from string import Formatter
from typing import Any

from gab.judge import judge
from gab.store import ResultStore, validate_version

client: Any | None = None

ROOT_DIR = Path(__file__).resolve().parent.parent

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"

# USD per million tokens. Update when Anthropic pricing changes.
MODEL_PRICING_USD_PER_MTOK = {
    SONNET: {"input": 3.0, "output": 15.0},
    HAIKU: {"input": 1.0, "output": 5.0},
}


def pick_model(case: dict) -> str:
    """Route simple cases to Haiku, complex ones to Sonnet.

    A case is "complex" if any of:
    - the expression is 3+ words (compound slang, more context needed),
    - it has 5+ criteria to satisfy,
    - the target audience description is 60+ characters (ambiguous framing).
    """
    expression_words = len(case.get("expression", "").split())
    criteria_count = len(case.get("criteria", []) or [])
    audience_len = len(case.get("target_audience", "") or "")
    is_complex = expression_words >= 3 or criteria_count >= 5 or audience_len >= 60
    return SONNET if is_complex else HAIKU


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_USD_PER_MTOK.get(model)
    if not pricing:
        return 0.0
    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def get_client() -> Any:
    global client
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    return client


def _literal_brace(field: str, spec: str, conv: str | None) -> str:
    piece = "{" + field
    if conv:
        piece += "!" + conv
    if spec:
        piece += ":" + spec
    return piece + "}"


def render_prompt(template: str, case: dict) -> str:
    tokens = list(Formatter().parse(template))
    required = {
        field
        for _, field, spec, _ in tokens
        if field and field.isidentifier() and not spec
    }
    missing = sorted(required - case.keys())
    if missing:
        raise KeyError(
            f"Golden set case {case.get('id', '<unknown>')} "
            f"is missing fields: {', '.join(missing)}"
        )

    out: list[str] = []
    formatter = Formatter()
    for literal, field, spec, conv in tokens:
        out.append(literal)
        if field is None:
            continue
        if field in case:
            value = case[field]
            if conv:
                value = formatter.convert_field(value, conv)
            out.append(format(value, spec or ""))
        else:
            out.append(_literal_brace(field, spec or "", conv))
    return "".join(out)


def _response_text(response) -> str:
    return "".join(
        block.get("text", "") if isinstance(block, dict) else block.text
        for block in response.content
        if (isinstance(block, dict) and block.get("type") == "text")
        or (getattr(block, "type", None) == "text" and hasattr(block, "text"))
    ).strip()


def resolve_input_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    repo_candidate = ROOT_DIR / path
    if repo_candidate.exists():
        return repo_candidate

    return candidate


def run_eval(golden_set_path: str, prompt_path: str, version: str) -> list[dict]:
    validate_version(version)

    golden = json.loads(resolve_input_path(golden_set_path).read_text(encoding="utf-8"))
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

        filled_prompt = render_prompt(prompt_template, case)

        model = pick_model(case)
        response = get_client().messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": filled_prompt}],
        )
        output = _response_text(response)
        input_tokens, output_tokens = _usage_tokens(response)
        cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)

        judgment = judge(
            expression=case["expression"],
            context=case["context"],
            target_audience=case["target_audience"],
            criteria=case["criteria"],
            output=output,
        )
        results.append(
            {
                "id": case["id"],
                "expression": case["expression"],
                "output": output,
                "score": judgment.score,
                "reasoning": judgment.reasoning,
                "criteria_met": judgment.criteria_met,
                "criteria_failed": judgment.criteria_failed,
                "model_used": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
            }
        )

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
