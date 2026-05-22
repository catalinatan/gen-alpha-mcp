import typer
from rich.console import Console
from rich.table import Table

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
) -> None:
    """Run eval against a golden set"""
    results = run_eval(golden, prompt, version)
    if not results:
        console.print(f"[yellow]Version {version}: no cases were evaluated[/yellow]")
        raise typer.Exit(code=1)

    avg = sum(result["score"] for result in results) / len(results)
    console.print(f"[green]Version {version}: avg score {avg:.2f}/5[/green]")
    console.print(f"[dim]Saved to results.db (version={version})[/dim]")


@app.command()
def leaderboard() -> None:
    """Show scores across all prompt versions"""
    rows = ResultStore().leaderboard()
    if not rows:
        console.print("[yellow]No saved runs found yet.[/yellow]")
        return

    table = Table(title="Prompt Version Scores")
    table.add_column("Version")
    table.add_column("Avg Score", justify="right")
    table.add_column("Cases", justify="right")

    for row in rows:
        table.add_row(
            row["version"],
            f"{row['avg_score']:.2f}",
            str(row["cases"]),
        )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
