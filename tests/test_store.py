import pytest

from gab.store import validate_version


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


def test_leaderboard_empty(store):
    assert store.leaderboard() == []
