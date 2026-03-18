# Archive

Unused or redundant code moved during initial cleanup.

## Contents

- **root_wrappers/** – Root-level re-export modules that duplicated `agentic_oran_rca` package exports. All runtime code lives in `agentic_oran_rca/agentic_oran_rca/`.
  - `agents/` – context_agent, rca_agent, explanation_agent (1-line re-exports)
  - `api/` – server (1-line re-export)
  - `data/` – dataset_generator (1-line re-export)
  - `evaluation/` – graph_generator, metrics (1-line re-exports)
  - `graph/` – graph_builder, neo4j_client (1-line re-exports)
  - `rag/` – retriever, vector_store (1-line re-exports)

- **package_results/** – Redundant `results/` folder that was inside the package. Active results are written to project root `results/`.
