import datetime
import json
import re
import sqlite_utils
from mcp.server.fastmcp import FastMCP
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_SETS_DIR = ROOT_DIR / "golden_sets"
DB_PATH = ROOT_DIR / "results.db"
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

mcp = FastMCP("golden-set-server")
db = sqlite_utils.Database(DB_PATH)


def ensure_results_table() -> None:
    """Create the results table before the first insert/query."""
    db["results"].create(
        {
            "version": str,
            "case_id": str,
            "score": int,
            "reasoning": str,
            "run_at": str,
        },
        if_not_exists=True,
    )


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
    ensure_results_table()
    db["results"].insert({
        "version": version,
        "case_id": case_id,
        "score": score,
        "reasoning": reasoning,
        "run_at": datetime.datetime.now(datetime.UTC).isoformat()
    })

@mcp.tool()
def get_leaderboard() -> list[dict]:
    """Return average scores per prompt version"""
    ensure_results_table()
    return list(db.query(
        "SELECT version, AVG(score) as avg_score, COUNT(*) as cases FROM results GROUP BY version ORDER BY avg_score DESC"
    ))


if __name__ == "__main__":
    mcp.run()
