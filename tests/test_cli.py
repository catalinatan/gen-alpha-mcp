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


def _fake_results(*scores):
    return [{"score": s} for s in scores]


def test_run_fail_below_exits_nonzero_when_avg_under_threshold():
    with patch("gab.cli.run_eval", return_value=_fake_results(2, 3, 3)):
        result = runner.invoke(app, ["run", "v1", "--fail-below", "3.5"])
    assert result.exit_code == 1
    assert "Quality gate failed" in result.output


def test_run_fail_below_passes_when_avg_meets_threshold():
    with patch("gab.cli.run_eval", return_value=_fake_results(4, 4, 3)):
        result = runner.invoke(app, ["run", "v1", "--fail-below", "3.5"])
    assert result.exit_code == 0
    assert "Quality gate failed" not in result.output


def test_run_without_fail_below_does_not_gate():
    with patch("gab.cli.run_eval", return_value=_fake_results(1, 1, 1)):
        result = runner.invoke(app, ["run", "v1"])
    assert result.exit_code == 0


def test_leaderboard_show_cost_renders_cost_columns(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.save("v1", "c1", 4, "good", cost_usd=0.01, input_tokens=100, output_tokens=50)
    store.save("v1", "c2", 2, "meh", cost_usd=0.03, input_tokens=200, output_tokens=80)
    with _patched_store(store):
        result = runner.invoke(app, ["leaderboard", "--show-cost"])
    assert result.exit_code == 0
    assert "Total $" in result.output
    assert "Score/$" in result.output
    assert "$0.0400" in result.output


def test_leaderboard_without_show_cost_hides_cost_columns(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.save("v1", "c1", 4, "good", cost_usd=0.01)
    with _patched_store(store):
        result = runner.invoke(app, ["leaderboard"])
    assert result.exit_code == 0
    assert "Total $" not in result.output
    assert "Score/$" not in result.output
