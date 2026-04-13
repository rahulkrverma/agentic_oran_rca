## Agentic Retrieval-Augmented Root Cause Analysis System for Cell Outage Recovery in O-RAN Networks

**Architecture (diagrams):** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layered stack, sequences, agent flows, offline evaluation.

**Project report (college submission):** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) — formal write-up, figure placeholders, validation, test results. To build **Word** (`.docx`), install [Pandoc](https://pandoc.org/installing.html), then from the repo root:

```powershell
& "$env:LOCALAPPDATA\Pandoc\pandoc.exe" -f markdown -t docx -o docs/PROJECT_REPORT.docx docs/PROJECT_REPORT.md
```

`docs/PROJECT_REPORT.md`, `docs/ARCHITECTURE.md`, and `docs/PROJECT_REPORT.docx` are listed in `.gitignore` (omit from version control if you prefer).

### Runbook (complete instructions)

#### Prerequisites

- Docker Desktop (recommended), or Python 3.10+ for local run
- If using Docker: enough RAM for Neo4j + Ollama (8GB+ recommended)

#### Start application (Docker)

From the `agentic_oran_rca/` folder:

```bash
docker compose up -d --build
```

Services and ports:

- API: `http://localhost:8000`
- Neo4j Browser UI: `http://localhost:7474` (user: `neo4j`, password: `admin1234`)
- Chroma: `http://localhost:8001`
- Ollama: `http://localhost:11434`

#### One-time initialization (Docker)

1. Pull Ollama models (LLM + embeddings):

```bash
docker exec -it agentic_oran_rca-ollama-1 ollama pull llama3
docker exec -it agentic_oran_rca-ollama-1 ollama pull nomic-embed-text
```

2. Generate dataset, populate Neo4j, index Chroma, and run evaluation:

```bash
docker compose exec api python -m agentic_oran_rca.main generate-data --rows 800 --seed 42 --start-time-utc 2026-01-01T00:00:00 --span-days 60 --log-level INFO
docker compose exec api python -m agentic_oran_rca.main build-graph
docker compose exec api python -m agentic_oran_rca.main index-vectors
docker compose exec api python -m agentic_oran_rca.main evaluate --test-size 0.25 --seed 42
```

3. (Optional) Ranking evaluation (Precision@K, Recall@K, F1@K): requires `index-vectors` completed and the embedding model available. Uses the same required environment variables as the API (including Neo4j), because the CLI loads full settings:

```bash
docker compose exec api python -m agentic_oran_rca.main evaluate-ranking --test-size 0.25 --seed 42 --k-values 1,3,5,7,10 --retrieve-pool 40
```

`--retrieve-pool` must be greater than or equal to the largest K in `--k-values` (comma-separated integers).

#### Run RCA (API)

```bash
curl -X POST http://localhost:8000/run_rca ^
  -H "Content-Type: application/json" ^
  -d "{\"cell\":\"Cell15\",\"alarm\":\"Cell Down\",\"kpi\":\"RSRP Drop\"}"
```

#### Run RCA with self-healing (optional)

Runs RCA, then simulates remediation, writes an auto-correction report, and records a notification:

```powershell
$body = @{ cell = "Cell15"; alarm = "Cell Down"; kpi = "RSRP Drop" } | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:8000/run_rca_with_healing" -Method POST -ContentType "application/json" -Body $body
```

Outputs:
- `results/healing_reports/healing_<cell>_<timestamp>.json` – auto-correction report
- `results/notifications.jsonl` – notification log (success or failure)

#### Start application (Local, without Docker)

1. Start Neo4j, Chroma, and Ollama yourself (or use Docker for those services only).
2. Set required environment variables (no defaults):

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `CHROMA_HOST`, `CHROMA_PORT`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`
- `LOG_LEVEL`

3. Install and run:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

python -m agentic_oran_rca.main generate-data --rows 800 --seed 42 --start-time-utc 2026-01-01T00:00:00 --span-days 60 --log-level INFO
python -m agentic_oran_rca.main build-graph
python -m agentic_oran_rca.main index-vectors
python -m agentic_oran_rca.main evaluate --test-size 0.25 --seed 42
python -m agentic_oran_rca.main evaluate-ranking --test-size 0.25 --seed 42 --k-values 1,3,5,7,10 --retrieve-pool 40
python -m agentic_oran_rca.main serve --host 0.0.0.0 --port 8000
```

#### See test / evaluation results

After running `evaluate`, outputs are written to:

- `results/evaluation_results.csv` (method comparison table)
- `results/alarm_only_report.txt` (classification report)
- `results/context_aware_report.txt` (classification report)
- `results/graphs/` (all figures in **PNG** and **PDF**):
  - `accuracy_comparison.(png|pdf)`
  - `confusion_matrix_alarm_only.(png|pdf)`
  - `confusion_matrix_context_aware.(png|pdf)`
  - `fault_distribution_hist.(png|pdf)`
  - `kpi_anomaly_distribution.(png|pdf)`
  - `network_topology.(png|pdf)`

After `evaluate-ranking`, Precision@K / Recall@K / F1@K artifacts are under `results/ranking_evaluation/` (full steps and API access in **Ranking evaluation reports** below).

#### Ranking evaluation reports (Precision@K, Recall@K, F1@K)

Use this when you want retrieval ranking metrics from Chroma (after incidents are embedded). It does **not** replace `evaluate`; it measures how often the true `expected_root_cause` appears within the top‑K **deduplicated** root causes returned for each test query.

**Prerequisites**

1. `generate-data` has produced `data/telecom_dataset.csv`.
2. `index-vectors` has populated the `oran_rca_incidents` collection in Chroma.
3. Ollama has the embedding model pulled (e.g. `nomic-embed-text` in Docker).
4. All required environment variables are set (same as running the API: Neo4j, Chroma, Ollama, `LOG_LEVEL`), because `evaluate-ranking` uses the same configuration loader.

**Generate reports (Docker)** — from the `agentic_oran_rca/` folder, with the stack running:

```bash
docker compose exec api python -m agentic_oran_rca.main evaluate-ranking --test-size 0.25 --seed 42 --k-values 1,3,5,7,10 --retrieve-pool 40
```

Optional: set the **row label** in the difficulty tables (default is `Chroma+<OLLAMA_EMBED_MODEL>`):

```bash
docker compose exec api python -m agentic_oran_rca.main evaluate-ranking --test-size 0.25 --seed 42 --k-values 1,3,5,7,10 --retrieve-pool 40 --report-name "MyRetrieval"
```

**Generate reports (local)** — venv activated, dependencies installed, env vars exported:

```bash
python -m agentic_oran_rca.main evaluate-ranking --test-size 0.25 --seed 42 --k-values 1,3,5,7,10 --retrieve-pool 40
```

**Output directory:** `results/ranking_evaluation/`

| File | Description |
|------|-------------|
| `ranking_metrics_summary_<timestamp>.json` | Mean P@K / R@K / F1@K, `mean_by_difficulty_at_k`, sample rows |
| `ranking_metrics_summary_latest.json` | Latest summary (overwritten each run) |
| `ranking_metrics_report_<timestamp>.txt` | Short text summary + per-difficulty means |
| `ranking_metrics_report_latest.txt` | Latest text report |
| `ranking_metrics_by_k_<timestamp>.csv` | One row per K (overall mean metrics) |
| `ranking_metrics_by_k_latest.csv` | Latest CSV |
| `ranking_metrics_difficulty_flat_<timestamp>.csv` | **Wide table**: one row, columns grouped like Precision@K×(Simple,Difficult,Mixed), then Recall@K, then F1@K (good for Excel / papers) |
| `ranking_metrics_difficulty_flat_latest.csv` | Latest wide CSV |
| `ranking_metrics_difficulty_table_<timestamp>.txt` | **Paper-style** fixed-width table (grouped headers per K) |
| `ranking_metrics_difficulty_table_latest.txt` | Latest |
| `ranking_metrics_difficulty_table_<timestamp>.md` | Same numbers in a Markdown pipe table |
| `ranking_metrics_difficulty_table_latest.md` | Latest |

**Difficulty tiers** (from `ALARM_TO_FAULT` in code): **Simple** = one candidate root cause for the alarm; **Mixed** = 2–3 candidates; **Difficult** = four or more. Metrics are **macro means** over held-out test rows in each tier.

**View or download reports via API** (start the API with `serve` or Docker, then):

- List files: `GET http://localhost:8000/evaluation/ranking/reports`
- Fetch one file by name: `GET http://localhost:8000/evaluation/ranking/reports/<filename>`  
  Example: `ranking_metrics_summary_latest.json` returns JSON; `.txt` and `.csv` return plain text. Only safe filenames are allowed (`[A-Za-z0-9._-]`).

```bash
curl -s http://localhost:8000/evaluation/ranking/reports
curl -s http://localhost:8000/evaluation/ranking/reports/ranking_metrics_summary_latest.json
```

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/evaluation/ranking/reports"
Invoke-RestMethod -Uri "http://localhost:8000/evaluation/ranking/reports/ranking_metrics_summary_latest.json"
```

#### Stop / reset

- Stop stack: `docker compose down`
- Stop and delete volumes (removes Neo4j/Chroma/Ollama data): `docker compose down -v`

### Quickstart (Docker)

1. Install Docker Desktop.
2. Start the stack:

```bash
docker compose up -d --build
```

3. Pull Ollama models (LLM + embeddings) inside the Ollama container:

```bash
docker exec -it agentic_oran_rca-ollama-1 ollama pull llama3
docker exec -it agentic_oran_rca-ollama-1 ollama pull nomic-embed-text
```

4. Generate dataset, build Neo4j graph, index Chroma, run evaluation (one-time):

```bash
docker compose exec api python -m agentic_oran_rca.main generate-data --rows 800 --seed 42 --start-time-utc 2026-01-01T00:00:00 --span-days 60 --log-level INFO
docker compose exec api python -m agentic_oran_rca.main build-graph
docker compose exec api python -m agentic_oran_rca.main index-vectors
docker compose exec api python -m agentic_oran_rca.main evaluate --test-size 0.25 --seed 42
```

5. (Optional) Ranking evaluation reports — see **Ranking evaluation reports** in the runbook above.

6. Run RCA via API:

```bash
curl -X POST http://localhost:8000/run_rca ^
  -H "Content-Type: application/json" ^
  -d "{\"cell\":\"Cell15\",\"alarm\":\"Cell Down\",\"kpi\":\"RSRP Drop\"}"
```

### Outputs

- **Dataset**: `data/telecom_dataset.csv`
- **Graphs**: `results/graphs/*.png` and `results/graphs/*.pdf`
- **Ranking evaluation** (after `evaluate-ranking`): `results/ranking_evaluation/` (JSON, TXT, CSV; see runbook)
- **Postman / API testing (step-by-step)**: `POSTMAN_API_GUIDE.md` (next to this README); legacy notes: `docs/POSTMAN_TESTING.md`
- **Neo4j UI**: `http://localhost:7474` (user: `neo4j`, password: `admin1234`)
- **Chroma**: `http://localhost:8001`

### Local (without Docker)

Set required environment variables (no defaults):

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `CHROMA_HOST`, `CHROMA_PORT`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`
- `LOG_LEVEL`

Then:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

python -m agentic_oran_rca.main generate-data --rows 800 --seed 42 --start-time-utc 2026-01-01T00:00:00 --span-days 60 --log-level INFO
python -m agentic_oran_rca.main build-graph
python -m agentic_oran_rca.main index-vectors
python -m agentic_oran_rca.main evaluate --test-size 0.25 --seed 42
python -m agentic_oran_rca.main evaluate-ranking --test-size 0.25 --seed 42 --k-values 1,3,5,7,10 --retrieve-pool 40
python -m agentic_oran_rca.main serve --host 0.0.0.0 --port 8000
```

