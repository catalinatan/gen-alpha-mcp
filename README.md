# Gen Alpha MCP

![CI](https://github.com/catalinatan/gen-alpha-mcp/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![MCP](https://img.shields.io/badge/protocol-MCP-7c3aed.svg)

**API key caveat:** Anthropic usage is billed per token. Set `ANTHROPIC_API_KEY` in your environment before running evaluations. The CLI is rate-limit aware but will fail loudly if the key is missing or exhausted.

## What This Project Does

A prompt evaluation harness for grading how well LLMs explain Gen Alpha slang — built as a Model Context Protocol (MCP) server with an `LLM-as-judge` scoring pipeline, a versioned prompt leaderboard, and a Typer-based CLI.

Core capabilities:

- **Golden set evaluation** — runs a candidate prompt against a curated JSON test set of Gen Alpha expressions (`no cap`, `sigma`, `skibidi toilet`, etc.), each with its own audience and pass criteria
- **LLM-as-judge scoring** — Claude Sonnet acts as the judge, returning structured JSON with a 1–5 score, reasoning, and per-criterion pass/fail breakdown
- **Versioned leaderboard** — every run is persisted to SQLite keyed by prompt version, so prompt iterations can be compared directly via `gab leaderboard`
- **MCP server** — exposes `get_test_cases`, `save_result`, and `get_leaderboard` over the Model Context Protocol, allowing any MCP-aware client (Claude Desktop, IDE agents) to drive evaluations as tools
- **CI pipeline** — GitHub Actions runs `ruff check` + `ruff format --check` and the pytest suite on every push

## Key Engineering Highlights

| Area | Detail |
| --- | --- |
| **MCP server** | [server/mcp_server.py](server/mcp_server.py) registers three tools on a `FastMCP` instance; path traversal is blocked by a regex allow-list on dataset names and a `is_relative_to` check on the resolved path |
| **LLM-as-judge pipeline** | [gab/judge.py](gab/judge.py) prompts Claude with structured criteria and parses the response via Pydantic; falls back to regex-extracted JSON if the model wraps its answer in prose |
| **Safe prompt templating** | [gab/runner.py](gab/runner.py) uses a `SafeFormatDict` + `string.Formatter` walk to validate that every `{field}` referenced by the template exists in the golden case, raising a descriptive `KeyError` instead of silently rendering `{missing}` |
| **Versioned result store** | [gab/store.py](gab/store.py) uses `sqlite-utils` with a regex-validated version key, indexed by `version`, `case_id`, `score`, and ISO-8601 `run_at` timestamp |
| **Leaderboard SQL** | Single grouped query aggregates `AVG(score)` and case counts per prompt version, ordered descending — no Python-side reduction |
| **Typer + Rich CLI** | [gab/cli.py](gab/cli.py) exposes `gab run <version>` and `gab leaderboard`, with colourised output and a Rich table for the leaderboard view |
| **Test coverage** | Six pytest files cover the CLI (mocked Anthropic), judge parser (clean JSON / embedded JSON / invalid JSON), runner template safety, store CRUD + leaderboard ordering, and MCP server path validation |
| **CI on GitHub Actions** | `ruff check .` and `ruff format --check .` enforced on every push and pull request |

## Tech Stack

| Layer | Technology |
| --- | --- |
| Runtime | Python 3.11+ |
| LLM provider | Anthropic Claude (Sonnet 4.6) via `anthropic` SDK |
| Protocol | Model Context Protocol via `mcp` (`FastMCP`) |
| CLI | Typer + Rich |
| Storage | SQLite via `sqlite-utils` |
| Validation | Pydantic 2 |
| Testing | pytest |
| Linting / formatting | Ruff |
| CI | GitHub Actions |

## Project Structure

```
.
├── gab/                        # Eval harness package
│   ├── cli.py                  # Typer CLI: `gab run`, `gab leaderboard`
│   ├── runner.py               # Prompt rendering + Anthropic call loop
│   ├── judge.py                # LLM-as-judge scoring + Pydantic parsing
│   └── store.py                # SQLite result persistence + leaderboard query
├── server/
│   └── mcp_server.py           # FastMCP server exposing eval tools
├── golden_sets/
│   └── gen_alpha.json          # Curated test cases (expression, context, audience, criteria)
├── prompts/
│   ├── v1.txt                  # Baseline prompt
│   ├── v2.txt                  # Adds audience-targeted instructions
│   └── v3.txt                  # Few-shot examples + structured output rules
├── tests/                      # pytest suite (CLI, judge, runner, store, MCP server)
├── .github/workflows/ci.yml    # Ruff + pytest CI
├── cli.py                      # Top-level entry shim
└── pyproject.toml
```

## Local Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/catalinatan/gen-alpha-mcp.git
cd gen-alpha-mcp

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package with dev extras
pip install -e ".[dev]"

# 4. Set environment variables
export ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Usage

### Run an evaluation

```bash
gab run v1                              # uses prompts/v1.txt and golden_sets/gen_alpha.json
gab run v2 --prompt prompts/v2.txt      # score a different prompt version
gab run v3 -p prompts/v3.txt -g golden_sets/gen_alpha.json
```

Each run scores every case in the golden set via the judge model and persists results to `results.db` under the supplied version label.

### View the leaderboard

```bash
gab leaderboard
```

Renders a Rich table of average scores and case counts across every prompt version recorded in the database.

### Run the MCP server

```bash
python -m server.mcp_server
```

The server exposes three tools — `get_test_cases(dataset)`, `save_result(version, case_id, score, reasoning)`, and `get_leaderboard()` — over stdio for consumption by any MCP-compatible client.

## Running Tests

```bash
pytest
```

The suite covers CLI invocation (with mocked Anthropic calls), judge JSON parsing edge cases, prompt rendering safety, store CRUD + leaderboard aggregation, and MCP server path validation.

```bash
ruff check .
ruff format --check .
```

Linting and formatting checks mirror the CI configuration in [.github/workflows/ci.yml](.github/workflows/ci.yml).

## License

Released under the MIT License — see [LICENSE](LICENSE).
