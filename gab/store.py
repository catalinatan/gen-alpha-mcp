import datetime
import re
import sqlite_utils
from pathlib import Path

VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_version(version: str) -> None:
    if not VERSION_RE.fullmatch(version):
        raise ValueError("Version may only contain letters, numbers, '.', '_' and '-'.")

DB_PATH = Path(__file__).resolve().parent.parent / "results.db"

_SCHEMA = {
    "version": str,
    "case_id": str,
    "score": int,
    "reasoning": str,
    "run_at": str,
}


class ResultStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db = sqlite_utils.Database(db_path)
        self.db["results"].create(_SCHEMA, if_not_exists=True)

    def save(self, version: str, case_id: str, score: int, reasoning: str) -> None:
        self.db["results"].insert({
            "version": version,
            "case_id": case_id,
            "score": score,
            "reasoning": reasoning,
            "run_at": datetime.datetime.now(datetime.UTC).isoformat(),
        })

    def leaderboard(self) -> list[dict]:
        return list(self.db.query(
            "SELECT version, AVG(score) AS avg_score, COUNT(*) AS cases "
            "FROM results GROUP BY version ORDER BY avg_score DESC"
        ))

    def results(self, version: str | None = None) -> list[dict]:
        if version is None:
            return list(self.db["results"].rows)
        return list(self.db["results"].rows_where("version = ?", [version]))
