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
- **Tiered model routing** — a heuristic router in `runner.py` sends simple cases to Haiku and complex cases (multi-word slang, 5+ criteria, or ambiguous audience) to Sonnet, instead of paying Sonnet rates on every call
- **Per-case cost tracking** — every result row stores `model_used`, `input_tokens`, `output_tokens`, and `cost_usd`, computed from `response.usage` and a per-model price table
- **Versioned leaderboard** — every run is persisted to SQLite keyed by prompt version, so prompt iterations can be compared directly via `gab leaderboard`, with `--show-cost` adding total spend, per-case spend, and score-per-dollar columns
- **MCP server** — exposes `get_test_cases`, `save_result`, and `get_leaderboard` over the Model Context Protocol, allowing any MCP-aware client (Claude Desktop, IDE agents) to drive evaluations as tools
- **CI pipeline** — GitHub Actions runs `ruff check` + `ruff format --check` and the pytest suite on every push

## Key Engineering Highlights

| Area | Detail |
| --- | --- |
| **MCP server** | [server/mcp_server.py](server/mcp_server.py) registers three tools on a `FastMCP` instance; path traversal is blocked by a regex allow-list on dataset names and a `is_relative_to` check on the resolved path |
| **LLM-as-judge pipeline** | [gab/judge.py](gab/judge.py) prompts Claude with structured criteria and parses the response via Pydantic; falls back to regex-extracted JSON if the model wraps its answer in prose |
| **Model router** | `pick_model(case)` in [gab/runner.py](gab/runner.py) inspects expression word count, criteria count, and audience description length to route between Haiku (cheap, simple) and Sonnet (expensive, complex). Pricing lives in a single `MODEL_PRICING_USD_PER_MTOK` table so it can be retuned without touching call sites |
| **Cost capture** | `response.usage.input_tokens` and `output_tokens` are pulled from the Anthropic SDK response, multiplied by the per-model price, and persisted alongside `model_used`. The leaderboard exposes `score_per_dollar` as `AVG(score) / SUM(cost_usd)` so prompt versions can be compared by economic efficiency, not just raw quality |
| **Safe prompt templating** | [gab/runner.py](gab/runner.py) walks `string.Formatter().parse()` to require every bare `{field}` placeholder and preserve any `{field:format-spec}` ones verbatim when the value isn't supplied, raising a descriptive `KeyError` only for missing required fields |
| **Backwards-compatible schema migration** | [gab/store.py](gab/store.py) declares cost columns in a `_MIGRATABLE_COLUMNS` table and `add_column`s any that are missing on startup, so existing `results.db` files from earlier versions keep their data and gain the new columns transparently |
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

# 4. Set up your API key
cp .env.example .env
# then edit .env and paste your real ANTHROPIC_API_KEY
```

`.env` is loaded automatically on import of the `gab` package (via [gab/__init__.py](gab/__init__.py)), so both the CLI (`gab run …`) and the MCP server (`python -m server.mcp_server`) pick up `ANTHROPIC_API_KEY` without any extra wiring. `.env` is gitignored — never commit your real key.

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
gab leaderboard                # avg score + case count per version
gab leaderboard --show-cost    # adds total $, avg $/case, and score-per-dollar columns
```

`--show-cost` compares prompt versions by economic efficiency: it sums `cost_usd` per version, derives `score_per_dollar = AVG(score) / SUM(cost_usd)`, and renders both numbers next to the raw average score so a more expensive prompt has to clearly earn its premium.

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
