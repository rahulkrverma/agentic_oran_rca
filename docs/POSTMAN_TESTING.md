# Testing the RCA API with Postman

## Prerequisites

- API server running at `http://localhost:8000`
- Postman installed ([postman.com](https://www.postman.com/downloads/))

---

## 1. RCA Endpoint (`POST /run_rca`)

### Request

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `http://localhost:8000/run_rca` |
| **Headers** | `Content-Type: application/json` |
| **Body** | Raw → JSON |

### Body (JSON)

```json
{
  "cell": "Cell15",
  "alarm": "Cell Down",
  "kpi": "RSRP Drop"
}
```

Alternative field names: `cell_id`, `alarm_type`, `kpi_metric` are also accepted.

### Steps in Postman

1. Create a new request.
2. Set method to **POST**.
3. Enter URL: `http://localhost:8000/run_rca`.
4. Go to **Headers** tab → Add: `Content-Type` = `application/json`.
5. Go to **Body** tab → Select **raw** → Choose **JSON** (not Text) from dropdown.
6. Paste the JSON body above. Ensure no extra quotes wrap the entire body.
7. Click **Send**.

### Expected Response (200 OK)

```json
{
  "predicted_root_cause": "DU failure",
  "confidence": 0.84,
  "explanation": "Cell Cell15 experienced a service outage. The cell is served by DU2 which connects to CU1. The alarm indicates a power failure and the KPI shows signal loss. The most likely root cause is a DU2 power failure."
}
```

---

## 2. RCA with Self-Healing Endpoint (`POST /run_rca_with_healing`)

### Request

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `http://localhost:8000/run_rca_with_healing` |
| **Headers** | `Content-Type: application/json` |
| **Body** | Raw → JSON |

### Body (JSON)

```json
{
  "cell": "Cell15",
  "alarm": "Cell Down",
  "kpi": "RSRP Drop"
}
```

### Steps in Postman

1. Create a new request.
2. Set method to **POST**.
3. Enter URL: `http://localhost:8000/run_rca_with_healing`.
4. Go to **Headers** tab → Add: `Content-Type` = `application/json`.
5. Go to **Body** tab → Select **raw** → Choose **JSON**.
6. Paste the JSON body above.
7. Click **Send**.

### Expected Response (200 OK)

```json
{
  "predicted_root_cause": "DU failure",
  "confidence": 0.84,
  "explanation": "Cell Cell15 experienced a service outage...",
  "healing": {
    "remediation_action": "restart_du",
    "remediation_description": "Restart Distributed Unit to recover from software/hardware hang",
    "success": true,
    "message": "Remediation 'restart_du' completed successfully for Cell15.",
    "report_path": "/app/results/healing_reports/healing_Cell15_2026-03-16T12-30-45.json",
    "notification_sent": true
  }
}
```

---

## 3. Valid Input Values

Use values from your dataset for realistic tests:

| Field | Example Values |
|-------|----------------|
| **cell** | `Cell1`, `Cell15`, `Cell40` (from topology) |
| **alarm** | `Cell Down`, `High Latency`, `Packet Loss`, `Signal Degradation`, `Power Failure`, `Transport Link Failure` |
| **kpi** | `RSRP Drop`, `RSRQ Drop`, `SINR Drop`, `Throughput Drop`, `Latency Spike`, `PRB Utilization High`, `Packet Retransmissions High` |

---

## 4. Postman Collection (Optional)

Create a collection with two requests:

1. **RCA** – `POST` `http://localhost:8000/run_rca`
2. **RCA with Healing** – `POST` `http://localhost:8000/run_rca_with_healing`

Use the same body for both; save as a collection variable if needed.

---

## 5. Troubleshooting

| Issue | Check |
|-------|-------|
| Connection refused | API server running? `docker compose ps` or `python -m agentic_oran_rca.main serve --host 0.0.0.0 --port 8000` |
| 422 Unprocessable Entity | Body must be valid JSON with `cell`, `alarm`, `kpi` (all non-empty strings) |
| 500 Internal Server Error | Check API logs; ensure Neo4j, Chroma, Ollama are up and models are pulled |
