## Agentic Retrieval-Augmented Root Cause Analysis System for Cell Outage Recovery in O-RAN Networks

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

5. Run RCA via API:

```bash
curl -X POST http://localhost:8000/run_rca ^
  -H "Content-Type: application/json" ^
  -d "{\"cell\":\"Cell15\",\"alarm\":\"Cell Down\",\"kpi\":\"RSRP Drop\"}"
```

### Outputs

- **Dataset**: `data/telecom_dataset.csv`
- **Graphs**: `results/graphs/*.png` and `results/graphs/*.pdf`
- **Postman testing**: `docs/POSTMAN_TESTING.md`
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
python -m agentic_oran_rca.main serve --host 0.0.0.0 --port 8000
```

