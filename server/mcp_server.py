import json
import re
from mcp.server.fastmcp import FastMCP
from pathlib import Path
from gab.store import ResultStore

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_SETS_DIR = ROOT_DIR / "golden_sets"
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

mcp = FastMCP("golden-set-server")
_store = ResultStore()


def dataset_path(dataset: str) -> Path:
    """Return a validated golden set path."""
    if not DATASET_NAME_RE.fullmatch(dataset):
        raise ValueError("Dataset names may only contain letters, numbers, '_' and '-'.")

    path = (GOLDEN_SETS_DIR / f"{dataset}.json").resolve()
    if not path.is_relative_to(GOLDEN_SETS_DIR.resolve()):
        raise ValueError("Invalid dataset path.")

    return path

@mcp.tool()
def get_test_cases(dataset: str) -> list[dict]:
    """Fetch test cases from a named golden set"""
    path = dataset_path(dataset)
    return json.loads(path.read_text())

@mcp.tool()
def save_result(version: str, case_id: str, score: int, reasoning: str) -> None:
    """Save an eval result to the database"""
    _store.save(version=version, case_id=case_id, score=score, reasoning=reasoning)

@mcp.tool()
def get_leaderboard() -> list[dict]:
    """Return average scores per prompt version"""
    return _store.leaderboard()


if __name__ == "__main__":
    mcp.run()
