from unittest.mock import patch
from typer.testing import CliRunner
from gab.cli import app
from gab.store import ResultStore

runner = CliRunner()


def _patched_store(store):
    return patch("gab.cli.ResultStore", return_value=store)


def test_leaderboard_empty(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    with _patched_store(store):
        result = runner.invoke(app, ["leaderboard"])
    assert result.exit_code == 0
    assert "No saved runs" in result.output


def test_leaderboard_shows_version(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.save("v1", "case-1", 4, "good")
    store.save("v1", "case-2", 2, "meh")
    with _patched_store(store):
        result = runner.invoke(app, ["leaderboard"])
    assert result.exit_code == 0
    assert "v1" in result.output
    assert "3.00" in result.output


def test_leaderboard_ordering_in_output(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.save("slow", "c1", 2, "bad")
    store.save("fast", "c1", 5, "great")
    with _patched_store(store):
        result = runner.invoke(app, ["leaderboard"])
    assert result.output.index("fast") < result.output.index("slow")


def test_run_no_golden_file_errors():
    result = runner.invoke(app, ["run", "v1", "--golden", "nonexistent.json"])
    assert result.exit_code != 0
