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
    assert "Ambiguous" in result.output
    assert "0" in result.output


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


def test_run_reports_ambiguous_case_count():
    results = [
        {"score": 4, "flagged": False},
        {"score": 3, "flagged": True},
    ]
    with patch("gab.cli.run_eval", return_value=results):
        result = runner.invoke(app, ["run", "v1"])
    assert result.exit_code == 0
    assert "Flagged 1/2 ambiguous cases" in result.output


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


def test_run_passes_few_shot_k_to_standard_eval():
    with patch("gab.cli.run_eval", return_value=_fake_results(4)) as run_mock:
        result = runner.invoke(app, ["run", "v1", "--few-shot-k", "5"])
    assert result.exit_code == 0
    assert run_mock.call_args.kwargs["few_shot_k"] == 5


def test_run_passes_few_shot_k_to_agentic_eval():
    with patch("gab.cli.run_agentic_set", return_value=_fake_results(4)) as run_mock:
        result = runner.invoke(app, ["run", "v1", "--agentic", "--few-shot-k", "2"])
    assert result.exit_code == 0
    assert run_mock.call_args.kwargs["few_shot_k"] == 2


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


def test_ambiguous_command_lists_flagged_cases(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.save(
        "v1",
        "c1",
        3,
        "[AMBIGUOUS] tone unclear",
        flagged=True,
        ambiguity_reason="tone unclear",
    )
    store.save("v1", "c2", 4, "good")
    with _patched_store(store):
        result = runner.invoke(app, ["ambiguous", "v1"])
    assert result.exit_code == 0
    assert "c1" in result.output
    assert "tone unclear" in result.output
    assert "c2" not in result.output


def test_ambiguous_command_empty_state(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    with _patched_store(store):
        result = runner.invoke(app, ["ambiguous", "v1"])
    assert result.exit_code == 0
    assert "No ambiguous cases found for version v1" in result.output
