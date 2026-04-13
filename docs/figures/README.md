# Figures for the project report

Place exported architecture and result images here so `PROJECT_REPORT.md` can reference them with stable paths.

Suggested filenames (match **Section 8 — Figures** in `docs/PROJECT_REPORT.md`):

| File | Source |
|------|--------|
| `fig01-system-context.png` | Mermaid §1 in `docs/ARCHITECTURE.md` |
| `fig02-layered-architecture.png` | Mermaid §2 in `docs/ARCHITECTURE.md` |
| `fig03-rca-sequence.png` | Mermaid §3 in `docs/ARCHITECTURE.md` |
| `fig04-offline-evaluation.png` | Mermaid §7 in `docs/ARCHITECTURE.md` |
| `fig05-accuracy-comparison.png` | `results/graphs/accuracy_comparison.png` (copy or symlink) |
| `fig06-confusion-context-aware.png` | `results/graphs/confusion_matrix_context_aware.png` |
| `fig07-postman-rca.png` | Screenshot: Postman `POST /run_rca` 200 response |
| `fig08-neo4j-browser.png` | Screenshot: Neo4j Browser showing graph (optional) |

Export Mermaid: open [mermaid.live](https://mermaid.live), paste the code block from `ARCHITECTURE.md`, download PNG/SVG.
