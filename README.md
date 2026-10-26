# Research_AI

**From a rough research idea to a verified, end-to-end research plan, using open sources only, no API keys, and local LLMs.**

You write a few lines describing an idea. Research_AI tells you **what to investigate** (novelty check, gap map,
scored research directions, a reviewer-style critique) and **how to carry out the whole study** (hypotheses, method,
baselines, datasets, experiments, statistics, risks, target venue, paper skeleton, timeline). Every claim is anchored to
real papers, and a citation guard rejects anything the model invents.

> Design document and phase-by-phase plan: [PLAN.md](PLAN.md) (in Italian).
> A real output: [examples/report-attention-is-all-you-need.md](examples/report-attention-is-all-you-need.md).

## Highlights

- **Open sources only, no API keys.** arXiv, Semantic Scholar (anonymous), Crossref and Unpaywall (IEEE and other open access).
- **No invented citations.** The model can only cite IDs (`[P1]`, `[P2]`, ...) that exist in the run's corpus; unknown IDs are
  stripped, and every quoted evidence sentence is checked against the source text (✔ in the report).
- **Pillars + frontier.** The corpus mixes highly cited papers (how research is done in the field) with recent, fast-growing
  ones (what is still open).
- **Feasibility-aware.** The plan knows your real resources (GPU, boards, testbeds, hours per week) from `resources.yaml`.
- **Fully local.** LLMs run on [Ollama](https://ollama.com); a hard VRAM cap (default 50%) keeps a shared GPU usable.
- **Resumable and reproducible.** Each run is saved under `runs/<id>/` (queries, retrieved IDs, prompts, model versions,
  outputs); re-launching with the same `--run-id` resumes from the last completed stage.
- **Measurable.** Built-in evaluation with temporal cut-offs and on real past projects.

## How it works

```
idea ─► analyse idea ─► retrieve (arXiv/S2/Crossref) ─► rank (embeddings + LLM relevance) ─► cards
     ─► novelty + gaps ─► scored directions + reviewer critique ─► 3-part research plan ─► cite guard ─► report.md
```

| Stage | Module (`src/ra/`) | What it does |
|---|---|---|
| Harvest / ingest | `harvest.py`, `ingest.py`, `parse.py`, `db.py` | Builds an optional local base (SQLite) of papers per topic; PDF to text with PyMuPDF; pillar/frontier labelling by citation rate |
| Idea analysis + retrieval | `retrieve.py` | Turns the idea into a spec and queries, pools candidates, ranks them with `nomic-embed-text`, scores relevance 0-3 with an LLM; a "canonical lane" keeps very-cited papers as baseline candidates |
| Cards | `cards.py` | Structured cards per paper: from full text for the closest papers, from abstract for the rest |
| Analysis | `analysis.py` | Novelty verdict with the nearest works, gap map (G1-G6), scored directions, reviewer-style fatal-flaw check |
| Plan | `plan.py` | Research plan in three parts: question/hypotheses/method, baselines/datasets/experiments/statistics, venue/paper skeleton/timeline/risks |
| Guard | `guard.py` | Cite guard: valid IDs only, quotes verified against the sources |
| Report | `report.py`, `run.py` | Orchestration and the final Markdown report |
| LLM wrapper | `llm.py` | One entry point for Ollama with pydantic-validated JSON, retries and the GPU budget |
| Dashboard | `dash.py`, `dash.html` | Web UI with login: submit ideas, follow runs, read reports, rate them |
| Evaluation | `evaluate.py` | Cut-off and past-project backtests |

Directions are ranked with a weighted score (configurable in `config.yaml`):
`novelty 0.25 · impact 0.20 · feasibility 0.25 · risk 0.15 · profile fit 0.15`. A direction flagged by the critic with a
fatal flaw (⛔) is penalised and never chosen automatically.

## Requirements

- Linux, Python 3.10+
- [Ollama](https://ollama.com) with the models named in `config.yaml` (defaults):
  `gpt-oss:20b` (reasoning), `mistral-small:24b` (full-text cards), `qwen2.5:14b` (fast cards and relevance), `nomic-embed-text` (embeddings)
- A GPU is strongly recommended (developed on a shared NVIDIA L40 46 GB); smaller models can be set in `config.yaml`
- Internet access for arXiv, Semantic Scholar, Crossref and Unpaywall (no accounts)

## Install

```bash
git clone https://github.com/lucadagati/Research_AI && cd Research_AI
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama pull gpt-oss:20b && ollama pull mistral-small:24b && ollama pull qwen2.5:14b && ollama pull nomic-embed-text
```

Then edit two files:

1. `resources.yaml`: **your** hardware, testbeds, weekly hours and research profile (the plan must be feasible with these).
2. `harvest.py` (`MAIL`): set your e-mail, which Unpaywall requires as a polite-pool identifier (no sign-up).

## Usage

```bash
./ra idea "my idea in a few lines" [--venue "IEEE T-ASE"] [--weeks 12] [--cutoff 2021-01-01] [--direction 2] [--fast]
./ra idea idea.txt --run-id my-run      # an interrupted run resumes when relaunched with the same --run-id
./ra chat <run-id>                      # follow-up questions on a finished run
./ra dash [--port 8765] [--host 0.0.0.0]  # web dashboard (see below)
./ra ingest [--topics N] [--no-cards]   # build the local base for the topics in config.yaml
./ra eval cutoff|ld|retrieval [--n 6]   # evaluation, results in eval/results_*.json
```

| Option | Meaning |
|---|---|
| `--fast` | Cards from abstracts only (about 15 minutes); without it the 10 closest papers get full-text cards (about 1-1.5 h) |
| `--cutoff DATE` | Ignore papers published after this date (used by the temporal evaluation) |
| `--venue`, `--weeks` | Target venue and time budget that shape the plan |
| `--direction N` | Build the plan for direction N instead of the top-scored one |
| `--run-id` | Name the run; also the key for resuming |

### Output: `runs/<id>/`

`report.md` is the plan you read. Alongside it: `spec.json`, `corpus.json`, `novelty.json`, `gaps.json`,
`directions.json`, `plan.json` / `plan_raw.json`, `guard.json` (citation check), `meta.json` (models and weights used).
The report is written in the language set in `config.yaml` (`language: it` by default; the prompts are Italian-first).

### Dashboard

`./ra dash` serves a small web app (default `0.0.0.0:8765`) protected by HTTP Basic auth. Credentials come from a
git-ignored `.dash_auth` file (`user:password`) or the `RA_DASH_USER` / `RA_DASH_PASS` environment variables; the server refuses to start
without them. From the UI you can submit ideas (queued, one at a time), follow progress, read reports, re-plan on another
direction, resume a run, and vote on the quality of a report (votes go to `eval/ratings.jsonl`, a human evaluation set).
Put it behind a VPN or reverse proxy with TLS if you expose it.

## Configuration

| File | Purpose |
|---|---|
| `config.yaml` | Citation thresholds (pillars >= 100, niche >= 50), year windows, models, context length, GPU budget, retrieval sizes, scoring weights, ingest topics |
| `resources.yaml` | Author's real resources and profile used for feasibility and profile-fit scoring |

**GPU etiquette.** `llm.gpu_budget_frac` (default 0.5) caps the VRAM used by the tool's models: before loading a new one it
unloads its own largest ones and halves the context if needed (never below `min_ctx`), and models are dropped after
`keep_alive` of inactivity. The dashboard runs one job at a time.

## Evaluation

Quality is measured, not assumed (`eval/`):

- **Temporal cut-off** (`cases_cutoff.json`): famous papers are given as an idea as it could have been phrased before
  publication, with the search restricted to earlier papers. We check whether the real contribution reappears among the
  proposed directions and whether the real baselines are retrieved.
- **Past projects** (`cases_ld.json`): the initial ideas of the author's own finished papers are replayed against the
  published outcome.
- **Retrieval recall** (`results_retrieval.json`): share of the real baselines present in the candidate pool and in the final corpus.

Snapshot (local runs, October 2026, `--fast`):

| Test | Result |
|---|---|
| Past projects, hit@5 (real research direction among the top 5) | **4 / 4** |
| Citations that exist in the corpus | **100%** |
| Baseline recall of retrieval (n = 6 cut-off cases) | **0.60** |

The judge is an LLM, the sample is tiny, and the author's own papers are a biased test: treat these as sanity checks, not
benchmarks. The honest test is your own judgement on real ideas, which the dashboard's voting is meant to collect.

## Known limitations

- Semantic Scholar anonymous search often answers 429; it is tried and then skipped by a circuit breaker, so arXiv and Crossref carry most of the retrieval.
- Running two processes against arXiv at once triggers 429 as well.
- Novelty verdicts depend on retrieval: "new" with 0 close papers found means "not found", not "does not exist".
  The report states the confidence.
- LLM plans can still contain plausible but weak ideas. Automatic regeneration was tried and did not help (it started inventing
  datasets), so the citation check is reported as information instead.
- The prompts and the default profile are tuned for the author's fields (IoT/edge, serverless, WebAssembly, LoRaWAN,
  control barrier functions, LLM agents); change `resources.yaml` and the ingest topics for yours.

## Repository layout

```text
.
├── ra                 # launcher (./ra idea | ingest | eval | chat | dash)
├── config.yaml        # thresholds, models, GPU budget, weights, topics
├── resources.yaml     # your resources and profile
├── PLAN.md            # design document, phases F0-F12 (Italian)
├── requirements.txt
├── src/ra/            # pipeline, LLM wrapper, cite guard, dashboard
├── eval/              # test cases and results
└── examples/          # a real generated report
```

`data/` (local paper base, PDFs, SQLite) and `runs/` (one folder per idea) are generated locally and git-ignored.

## Contributing

Issues and pull requests are welcome, especially: more open sources, better relevance ranking, new evaluation cases,
and translations of the prompts.

## Authors

[Luca D'Agati](https://github.com/lucadagati) (University of Messina), with
[@fabiooraziomirto](https://github.com/fabiooraziomirto) as collaborator.

## License

[MIT](LICENSE)
