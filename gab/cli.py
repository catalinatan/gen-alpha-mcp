import typer
from rich.console import Console
from rich.table import Table

from gab.agent import run_agentic_set
from gab.runner import run_eval
from gab.store import ResultStore

app = typer.Typer()
console = Console()


@app.command()
def run(
    version: str = typer.Argument(..., help="Version name used for the saved run."),
    golden: str = typer.Option(
        "golden_sets/gen_alpha.json",
        "--golden",
        "-g",
        help="Path to the golden set JSON file.",
    ),
    prompt: str = typer.Option(
        "prompts/v1.txt",
        "--prompt",
        "-p",
        help="Path to the prompt template file.",
    ),
    fail_below: float | None = typer.Option(
        None,
        "--fail-below",
        help="Exit with code 1 if the average score is below this threshold.",
    ),
    agentic: bool = typer.Option(
        False,
        "--agentic",
        help="Use the ReAct-style agentic judge loop instead of a single judge call.",
    ),
) -> None:
    """Run eval against a golden set"""
    if agentic:
        results = run_agentic_set(golden, prompt, version)
    else:
        results = run_eval(golden, prompt, version)
    if not results:
        console.print(f"[yellow]Version {version}: no cases were evaluated[/yellow]")
        raise typer.Exit(code=1)

    avg = sum(result["score"] for result in results) / len(results)
    console.print(f"[green]Version {version}: avg score {avg:.2f}/5[/green]")
    console.print(f"[dim]Saved to results.db (version={version})[/dim]")

    if fail_below is not None and avg < fail_below:
        console.print(
            f"[red]Quality gate failed: "
            f"avg score {avg:.2f} < threshold {fail_below}[/red]"
        )
        raise typer.Exit(code=1)


@app.command()
def leaderboard(
    show_cost: bool = typer.Option(
        False,
        "--show-cost",
        help="Include total cost, average cost per case, and score-per-dollar.",
    ),
) -> None:
    """Show scores across all prompt versions"""
    rows = ResultStore().leaderboard()
    if not rows:
        console.print("[yellow]No saved runs found yet.[/yellow]")
        return

    table = Table(title="Prompt Version Scores")
    table.add_column("Version")
    table.add_column("Avg Score", justify="right")
    table.add_column("Cases", justify="right")
    if show_cost:
        table.add_column("Total $", justify="right")
        table.add_column("Avg $/case", justify="right")
        table.add_column("Score/$", justify="right")

    for row in rows:
        cells = [
            row["version"],
            f"{row['avg_score']:.2f}",
            str(row["cases"]),
        ]
        if show_cost:
            total = row.get("total_cost_usd") or 0.0
            avg = row.get("avg_cost_usd") or 0.0
            spd = row.get("score_per_dollar")
            cells.append(f"${total:.4f}")
            cells.append(f"${avg:.4f}")
            cells.append(f"{spd:.1f}" if spd else "—")
        table.add_row(*cells)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
