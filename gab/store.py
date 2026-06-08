import datetime
import re
from pathlib import Path

import sqlite_utils

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
    "model_used": str,
    "input_tokens": int,
    "output_tokens": int,
    "cost_usd": float,
    "flagged": bool,
    "ambiguity_reason": str,
}

# Columns added after the initial release; existing databases need them backfilled.
_MIGRATABLE_COLUMNS = {
    "model_used": str,
    "input_tokens": int,
    "output_tokens": int,
    "cost_usd": float,
    "flagged": bool,
    "ambiguity_reason": str,
}


class ResultStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db = sqlite_utils.Database(db_path)
        table = self.db["results"]
        table.create(_SCHEMA, if_not_exists=True)
        existing = set(table.columns_dict)
        for col, col_type in _MIGRATABLE_COLUMNS.items():
            if col not in existing:
                table.add_column(col, col_type)

    def save(
        self,
        version: str,
        case_id: str,
        score: int,
        reasoning: str,
        model_used: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        flagged: bool = False,
        ambiguity_reason: str = "",
    ) -> None:
        self.db["results"].insert(
            {
                "version": version,
                "case_id": case_id,
                "score": score,
                "reasoning": reasoning,
                "run_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "model_used": model_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "flagged": flagged,
                "ambiguity_reason": ambiguity_reason,
            }
        )

    def leaderboard(self) -> list[dict]:
        return list(
            self.db.query(
                "SELECT version, "
                "AVG(score) AS avg_score, "
                "COUNT(*) AS cases, "
                "SUM(CASE WHEN COALESCE(flagged, 0) THEN 1 ELSE 0 END) "
                "AS ambiguous_cases, "
                "SUM(COALESCE(cost_usd, 0)) AS total_cost_usd, "
                "AVG(COALESCE(cost_usd, 0)) AS avg_cost_usd, "
                "SUM(COALESCE(input_tokens, 0)) AS total_input_tokens, "
                "SUM(COALESCE(output_tokens, 0)) AS total_output_tokens, "
                "CASE WHEN SUM(COALESCE(cost_usd, 0)) > 0 "
                "THEN AVG(score) / SUM(cost_usd) ELSE NULL END "
                "AS score_per_dollar "
                "FROM results GROUP BY version ORDER BY avg_score DESC"
            )
        )

    def results(self, version: str | None = None) -> list[dict]:
        if version is None:
            return list(self.db["results"].rows)
        return list(self.db["results"].rows_where("version = ?", [version]))

    def ambiguous_results(self, version: str | None = None) -> list[dict]:
        where = "COALESCE(flagged, 0) = 1"
        params: list[str] = []
        if version is not None:
            where += " AND version = ?"
            params.append(version)
        return list(self.db["results"].rows_where(where, params))
