# Test Deduplicator

Simple AI-powered CLI that flags potentially duplicate QA test cases from a CSV export by comparing semantic similarity of title and body text. This was a failed experiment to see if it was a viable approach on its own - it's not, but I'm uploading to Github so that anyone who wants to play around with it can

## What It Does

- Loads test cases from CSV.
- Generates embeddings for title and body text (via OpenAI-compatible embeddings endpoint, typically LM Studio local server).
- Builds a weighted similarity matrix.
- Prints test-case pairs above a configurable similarity threshold.

## Requirements

- Windows, macOS, or Linux
- Python 3.12+ (project is configured with `requires-python = ">=3.12"`)
- An OpenAI-compatible embeddings endpoint 
	- Default target is LM Studio local server at `http://localhost:1234/v1`
- (For integration tests) LM Studio CLI `lms` available on your PATH

## Environment Description

This project loads environment variables from a repository-level `.env` file.

Environment variables used by the app:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_BASE` | No | `http://localhost:1234/v1` | Base URL for embeddings endpoint |
| `OPENAI_API_KEY` | No | `dummy-key-for-local-lm-studio` | API key for endpoint auth |
| `TEST_DEDUPLICATOR_FILE` | No | none | Input CSV path |
| `TEST_DEDUPLICATOR_ID_COL` | Yes* | none | Unique test case ID column name |
| `TEST_DEDUPLICATOR_TITLE_COL` | Yes* | none | Test title/summary column name |
| `TEST_DEDUPLICATOR_BODY_COL` | Yes* | none | Steps/description column name |
| `TEST_DEDUPLICATOR_MODEL` | Yes* | none | Embedding model ID served by endpoint |
| `TEST_DEDUPLICATOR_THRESHOLD` | No | `0.85` | Duplicate similarity threshold |
| `TEST_DEDUPLICATOR_TITLE_WEIGHT` | No | `0.60` | Title embedding weight |
| `TEST_DEDUPLICATOR_BODY_WEIGHT` | No | `0.40` | Body embedding weight |

`*` Required unless passed via CLI flags.

Recommended `.env` example:

```env
# Endpoint
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_API_KEY=dummy-key-for-local-lm-studio

# CSV schema mapping
TEST_DEDUPLICATOR_ID_COL=ID
TEST_DEDUPLICATOR_TITLE_COL=Title
TEST_DEDUPLICATOR_BODY_COL=Steps

# Embedding model exposed by your local/server endpoint
TEST_DEDUPLICATOR_MODEL=text-embedding-nomic-embed-text-v1.5@f32

# Optional tuning
TEST_DEDUPLICATOR_THRESHOLD=0.85
TEST_DEDUPLICATOR_TITLE_WEIGHT=0.60
TEST_DEDUPLICATOR_BODY_WEIGHT=0.40
```

## Setup

From repository root:

```sh
uv sync
```

Activate the virtual environment:

```sh
# macOS/Linux (bash/zsh)
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
.venv\Scripts\activate.bat
```

## Running The CLI

### Option A: Mostly from `.env`

```sh
python -m test_deduplicator.main
```

### Option B: Explicit CLI flags

```sh
python -m test_deduplicator.main \
  --file sample_data/sample_cases.csv \
  --id-col ID \
  --title-col Title \
  --body-col Steps \
  --model text-embedding-nomic-embed-text-v1.5@f32 \
  --threshold 0.75
```

Output is printed to stdout and includes each high-similarity pair plus final duplicate-cluster count.

## Running Tests

From repository root:

```sh
python -m pytest -q
```

Current suite includes:

- Unit tests for argument/env resolution and data-processing helpers
- Integration test that runs full workflow against `sample_data/sample_cases.csv`

⚠️ First-run note: if LM Studio does not already have `text-embedding-nomic-embed-text-v1.5-GGUF@f32` on disk, the integration fixture will automatically download it. This is roughly a 500MB download and can take a few minutes depending on network speed.

Integration fixture behavior:

- Checks LM Studio server health at `http://localhost:1234/v1/models`
- Attempts to bootstrap LM Studio with `lms daemon up` and `lms server start` when needed
- Uses `lms ls --embedding --json` to detect whether the required model exists locally
- Downloads the model with `lms get <model> -y` when missing
- Loads the model with `lms load <model>` and waits until it appears on the local server

## Project Structure

```text
sample_data/
	sample_cases.csv
src/test_deduplicator/
	main.py
tests/
	conftest.py
	test_main.py
	test_main_unit.py
```

## Notes

- CSV must include the columns you map with `--id-col`, `--title-col`, and `--body-col`.
- Title/body weights are combined linearly; keeping them near a 1.0 total is usually easiest to reason about.
