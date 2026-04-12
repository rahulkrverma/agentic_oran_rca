# Postman: step-by-step guide to call and test all APIs

Base URL when the stack is running locally: **`http://localhost:8000`**

Prerequisites:

- Docker Compose is up (`docker compose up -d --build`) or the API is running via `python -m agentic_oran_rca.main serve ...`.
- One-time data prep is done (`generate-data`, `build-graph`, `index-vectors`) so Neo4j and Chroma match your dataset (see `README.md`).
- Ollama models are pulled (`llama3`, `nomic-embed-text`) if you use Docker.

---

## Step 1: Create a Postman collection

1. Open Postman.
2. Click **Collections** → **New** → name it **Agentic O-RAN RCA**.
3. Open the collection → **Variables** tab.
4. Add variable **`base_url`**, initial value **`http://localhost:8000`**, current value the same.
5. Save the collection.

You will use `{{base_url}}` in every request URL.

---

## Step 2 (optional): Import the OpenAPI schema

1. Start the API so OpenAPI is served.
2. In Postman: **Import** → **Link** → enter: `http://localhost:8000/openapi.json`
3. Postman generates requests; you can still follow the manual steps below for clarity.

Interactive docs in a browser: `http://localhost:8000/docs`

---

## Step 3: Configure defaults for JSON POST bodies

For all **POST** endpoints below:

1. Method is **POST**.
2. **Headers**: add **`Content-Type`** = **`application/json`**.
3. **Body** → select **raw** → type **JSON**.

---

## Step 4: Test `POST /run_rca`

Root cause analysis only (context + RCA + explanation).

1. **New request** in your collection → name it **Run RCA**.
2. **URL**: `{{base_url}}/run_rca`
3. **Method**: **POST**
4. **Headers**: `Content-Type: application/json`
5. **Body** (raw JSON), example:

```json
{
  "cell": "Cell15",
  "alarm": "Cell Down",
  "kpi": "RSRP Drop"
}
```

6. Click **Send**.
7. **Expected (200)**: JSON with `predicted_root_cause`, `confidence`, `explanation`.
8. **Alternate field names** (equivalent body):

```json
{
  "cell_id": "Cell15",
  "alarm_type": "Cell Down",
  "kpi_metric": "RSRP Drop"
}
```

9. If you get **500** with `detail` / `traceback`: check API logs, Neo4j/Chroma/Ollama, and that the `cell` exists in the graph.

---

## Step 5: Test `POST /run_rca_with_healing`

Same RCA flow, then simulated remediation, healing report file, and notification log entry.

1. **New request** → **Run RCA with healing**.
2. **URL**: `{{base_url}}/run_rca_with_healing`
3. **Method**: **POST**
4. **Headers**: `Content-Type: application/json`
5. **Body** (same shape as Step 4):

```json
{
  "cell": "Cell15",
  "alarm": "Cell Down",
  "kpi": "RSRP Drop"
}
```

6. Click **Send**.
7. **Expected (200)**: JSON with `predicted_root_cause`, `confidence`, `explanation`, and nested **`healing`** (`remediation_action`, `remediation_description`, `success`, `message`, `report_path`, `notification_sent`).
8. On the host machine, confirm files under `results/healing_reports/` and append to `results/notifications.jsonl` when the API container has `./results` mounted (Docker Compose default).

---

## Step 6: Test `POST /run_auto_correction`

Orchestrates context retrieval, RCA, explanation, then a reviewer pass that may correct the root cause and returns a consolidated explanation.

1. **New request** → **Run auto correction**.
2. **URL**: `{{base_url}}/run_auto_correction`
3. **Method**: **POST**
4. **Headers**: `Content-Type: application/json`
5. **Body**:

```json
{
  "cell": "Cell15",
  "alarm": "Cell Down",
  "kpi": "RSRP Drop"
}
```

6. Click **Send**.
7. **Expected (200)**: JSON including:
   - `context_retrieval` (cell, du, cu, neighbors, related_faults, similar_incidents)
   - `initial_predicted_root_cause`, `initial_confidence`, `initial_explanation`
   - `corrected_root_cause`, `correction_applied`, `correction_rationale`, `final_explanation`
8. This endpoint performs **more LLM calls** than `/run_rca`; expect higher latency.

---

## Step 7: Test `GET /evaluation/ranking/reports`

Lists ranking evaluation report files (after you have run `evaluate-ranking` and produced files under `results/ranking_evaluation/`).

1. **New request** → **List ranking reports**.
2. **URL**: `{{base_url}}/evaluation/ranking/reports`
3. **Method**: **GET**
4. No body.
5. Click **Send**.
6. **Expected (200)**: JSON with `directory` and `reports` (array of `name`, `size_bytes`, `modified_utc`). If no reports were generated yet, `reports` may be an empty array.

---

## Step 8: Test `GET /evaluation/ranking/reports/{filename}`

Downloads one report file by name. Only safe filenames are allowed: letters, digits, `.`, `_`, `-` (no path segments).

1. **New request** → **Get ranking report file**.
2. From Step 7, copy a **`name`** value, for example `ranking_metrics_summary_latest.json`.
3. **URL**: `{{base_url}}/evaluation/ranking/reports/ranking_metrics_summary_latest.json`  
   (Replace the filename with the one you need.)
4. **Method**: **GET**
5. Click **Send**.
6. **Expected (200)**:
   - For **`.json`**: parsed JSON body in Postman.
   - For **`.txt`** or **`.csv`**: raw text (Postman shows it in the response body).
7. **400** if the filename is invalid; **404** if the file does not exist.

Useful filenames (after running ranking evaluation):

- `ranking_metrics_summary_latest.json`
- `ranking_metrics_report_latest.txt`
- `ranking_metrics_by_k_latest.csv`
- `ranking_metrics_difficulty_table_latest.txt`
- `ranking_metrics_difficulty_flat_latest.csv`
- `ranking_metrics_difficulty_table_latest.md`

---

## Quick reference table

| Step | Method | Path | Body |
|------|--------|------|------|
| 4 | POST | `/run_rca` | JSON: `cell`, `alarm`, `kpi` (or `cell_id`, `alarm_type`, `kpi_metric`) |
| 5 | POST | `/run_rca_with_healing` | Same as Step 4 |
| 6 | POST | `/run_auto_correction` | Same as Step 4 |
| 7 | GET | `/evaluation/ranking/reports` | None |
| 8 | GET | `/evaluation/ranking/reports/{filename}` | None |

---

## Troubleshooting

- **Could not get response / connection refused**: API not listening on port 8000; confirm `docker compose ps` shows `api` **Up**, or local `serve` is running.
- **500** on POST routes: read `detail` in the JSON body; often Neo4j/Chroma/Ollama misconfiguration, missing graph data, or LLM JSON parse errors.
- **422** on POST: malformed JSON or missing required logical fields (`cell`/`cell_id`, etc.).
- **Empty ranking report list**: run ranking evaluation from the README (`evaluate-ranking`) so `results/ranking_evaluation/` contains files visible to the API container.

---

## Related documentation

- Project runbook: `README.md`
- Older Postman notes (if present): `docs/POSTMAN_TESTING.md`
