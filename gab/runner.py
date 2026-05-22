import json
from pathlib import Path
from string import Formatter
from typing import Any

from gab.judge import judge
from gab.store import validate_version, ResultStore

client: Any | None = None

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_client() -> Any:
    global client
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    return client


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_prompt(template: str, case: dict) -> str:
    fields = {
        name
        for _, name, _, _ in Formatter().parse(template)
        if name and name.isidentifier()
    }
    missing = sorted(field for field in fields if field not in case)
    if missing:
        raise KeyError(f"Golden set case {case.get('id', '<unknown>')} is missing fields: {', '.join(missing)}")

    return template.format_map(SafeFormatDict(case))


def _response_text(response) -> str:
    return "".join(
        block.get("text", "") if isinstance(block, dict) else block.text
        for block in response.content
        if (
            isinstance(block, dict) and block.get("type") == "text"
        ) or (
            getattr(block, "type", None) == "text" and hasattr(block, "text")
        )
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

        response = get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": filled_prompt}]
        )
        output = _response_text(response)

        judgment = judge(
            expression=case["expression"],
            context=case["context"],
            target_audience=case["target_audience"],
            criteria=case["criteria"],
            output=output
        )
        results.append({
            "id": case["id"],
            "expression": case["expression"],
            "output": output,
            "score": judgment.score,
            "reasoning": judgment.reasoning,
            "criteria_met": judgment.criteria_met,
            "criteria_failed": judgment.criteria_failed,
        })

    store = ResultStore()
    for result in results:
        store.save(version=version, case_id=result["id"], score=result["score"], reasoning=result["reasoning"])
    return results
