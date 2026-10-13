# CLAUDE.md — fund-ai-exposure

## What this project is
A small research project: use a language model (Claude Haiku 5.5 via the Anthropic API) to classify
every equity holding in Norway's Government Pension Fund Global by its exposure to AI, and measure
how far the model can be trusted against a hand-labelled set of 40 companies.

The point is not the classifier. The point is the evaluation: where the model is right, where it
is wrong, and whether its stated confidence can be trusted. Honest method beats polish.

## Key documents
- `docs/codebook.md` — category definitions. It is also the system prompt for the model.
  Never change it without adding an entry to `docs/codebook_changelog.md`.

## Hard rules
- **Never read, print, log or commit `.env`.** Load the key in Python from `FUND_AI_ANTHROPIC_KEY`
  via python-dotenv and pass it explicitly to `anthropic.Anthropic(api_key=...)`.
  Never use the variable name `ANTHROPIC_API_KEY`.
- **Never call the API on `data/labels/sample.csv` before I say the labels are finished.**
  Smoke tests use companies that are not in the sample.
- **The test split is used once.** Tune prompt and codebook on `dev` only. After the final test
  run, do not change the prompt and re-run test.
- Ask before any command that could cost more than USD 2 in API usage. Print an estimate first.
- The model gets no web search and no tools other than the forced output tool.

## Tech choices
- Python 3.11+, virtual env in `.venv`, dependencies in `requirements.txt`.
- pandas, matplotlib, `anthropic` SDK, `python-dotenv`.
- Models: `claude-haiku-5-5` (default), `claude-sonnet-5-5` (comparison on test only).
- Structured output via a forced tool call named `record_classification`, validated against the
  allowed values. Retry up to 3 times with backoff.
- Outputs are appended to JSONL (one line per company, with model, prompt_version, token usage,
  timestamp) so runs can resume after a crash.
- Random seed 42 everywhere.

## Folder layout
```
data/raw/                 downloaded NBIM file (not committed)
data/holdings_equities.csv
data/labels/sample.csv    hand labels with dev/test split
src/                      scripts, one job each
results/                  outputs (jsonl), tables (csv), figures (png)
docs/                     codebook and changelog
```

## Style
- Small scripts with `main()` and argparse flags (`--model`, `--split`, `--limit`).
- English in code, comments, README and commit messages, even if I write to you in Norwegian.
- Simple, readable code that I can explain line by line.
- After each step: summarise what changed in 3–5 lines and say what I should check by hand,
  then commit with a clear message.
