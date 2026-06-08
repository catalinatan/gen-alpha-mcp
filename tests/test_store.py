import pytest
import sqlite_utils

from gab.store import ResultStore, validate_version


def test_validate_version_valid():
    validate_version("v1")
    validate_version("v1.0")
    validate_version("my-prompt_v2.1")


@pytest.mark.parametrize("bad", ["has spaces", "has/slash", "bad!", ""])
def test_validate_version_invalid(bad):
    with pytest.raises(ValueError):
        validate_version(bad)


def test_save_and_retrieve(store):
    store.save("v1", "case-1", 4, "looks good")
    rows = store.results()
    assert len(rows) == 1
    assert rows[0]["version"] == "v1"
    assert rows[0]["case_id"] == "case-1"
    assert rows[0]["score"] == 4
    assert rows[0]["reasoning"] == "looks good"
    assert rows[0]["flagged"] == 0
    assert rows[0]["ambiguity_reason"] == ""
    assert "run_at" in rows[0]


def test_results_filter_by_version(store):
    store.save("v1", "case-1", 4, "good")
    store.save("v2", "case-1", 3, "ok")
    assert len(store.results("v1")) == 1
    assert len(store.results("v2")) == 1
    assert len(store.results()) == 2


def test_leaderboard_ordering(store):
    store.save("v1", "case-1", 2, "bad")
    store.save("v2", "case-1", 5, "great")
    board = store.leaderboard()
    assert board[0]["version"] == "v2"
    assert board[1]["version"] == "v1"


def test_leaderboard_aggregates(store):
    store.save("v1", "case-1", 4, "good")
    store.save("v1", "case-2", 2, "meh")
    board = store.leaderboard()
    assert len(board) == 1
    assert board[0]["avg_score"] == pytest.approx(3.0)
    assert board[0]["cases"] == 2
    assert board[0]["ambiguous_cases"] == 0


def test_leaderboard_empty(store):
    assert store.leaderboard() == []


def test_save_persists_cost_columns(store):
    store.save(
        "v1",
        "case-1",
        4,
        "good",
        model_used="claude-haiku-4-5",
        input_tokens=120,
        output_tokens=80,
        cost_usd=0.00052,
    )
    rows = store.results()
    assert rows[0]["model_used"] == "claude-haiku-4-5"
    assert rows[0]["input_tokens"] == 120
    assert rows[0]["output_tokens"] == 80
    assert rows[0]["cost_usd"] == pytest.approx(0.00052)


def test_save_defaults_cost_columns_to_zero(store):
    store.save("v1", "case-1", 4, "good")
    rows = store.results()
    assert rows[0]["model_used"] == ""
    assert rows[0]["input_tokens"] == 0
    assert rows[0]["output_tokens"] == 0
    assert rows[0]["cost_usd"] == 0.0


def test_save_persists_ambiguity_columns(store):
    store.save(
        "v1",
        "case-1",
        3,
        "[AMBIGUOUS] tone unclear",
        flagged=True,
        ambiguity_reason="tone unclear",
    )
    rows = store.results()
    assert rows[0]["flagged"] == 1
    assert rows[0]["ambiguity_reason"] == "tone unclear"


def test_leaderboard_counts_ambiguous_cases(store):
    store.save("v1", "c1", 4, "good")
    store.save("v1", "c2", 3, "unclear", flagged=True)
    board = store.leaderboard()
    assert board[0]["ambiguous_cases"] == 1


def test_ambiguous_results_filters_by_version(store):
    store.save("v1", "c1", 3, "unclear", flagged=True)
    store.save("v1", "c2", 4, "good")
    store.save("v2", "c3", 2, "unclear", flagged=True)
    assert [row["case_id"] for row in store.ambiguous_results("v1")] == ["c1"]
    assert {row["case_id"] for row in store.ambiguous_results()} == {"c1", "c3"}


def test_leaderboard_aggregates_cost(store):
    store.save("v1", "c1", 4, "good", cost_usd=0.01, input_tokens=100, output_tokens=50)
    store.save("v1", "c2", 2, "meh", cost_usd=0.03, input_tokens=200, output_tokens=150)
    board = store.leaderboard()
    row = board[0]
    assert row["total_cost_usd"] == pytest.approx(0.04)
    assert row["avg_cost_usd"] == pytest.approx(0.02)
    assert row["total_input_tokens"] == 300
    assert row["total_output_tokens"] == 200
    # avg_score (3.0) / total_cost (0.04) = 75
    assert row["score_per_dollar"] == pytest.approx(75.0)


def test_leaderboard_score_per_dollar_null_when_no_cost(store):
    store.save("v1", "c1", 4, "good")
    board = store.leaderboard()
    assert board[0]["score_per_dollar"] is None


def test_store_migrates_legacy_schema(tmp_path):
    """Existing databases without cost columns get them added on init."""
    db_path = tmp_path / "legacy.db"
    legacy = sqlite_utils.Database(db_path)
    legacy["results"].create(
        {
            "version": str,
            "case_id": str,
            "score": int,
            "reasoning": str,
            "run_at": str,
        }
    )
    legacy["results"].insert(
        {
            "version": "old",
            "case_id": "c1",
            "score": 5,
            "reasoning": "pre-migration row",
            "run_at": "2025-01-01T00:00:00",
        }
    )

    store = ResultStore(db_path=db_path)
    columns = set(store.db["results"].columns_dict)
    assert {
        "model_used",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "flagged",
        "ambiguity_reason",
    } <= columns

    # legacy row survives, new row coexists with cost data
    store.save("new", "c2", 4, "post", cost_usd=0.01)
    rows = store.results()
    assert len(rows) == 2
