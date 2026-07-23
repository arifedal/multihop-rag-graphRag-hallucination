# Stage 1 — NLP Pipeline and RAG vs GraphRAG

Notebook: `multihop_nlp_pipeline.ipynb`

## What this notebook does

1. Loads 2WikiMultihopQA and samples a balanced 500-question subset (250 `bridge_comparison`, 250 `comparison`, seed 42).
2. Extracts linguistic features with spaCy — NER, dependency parsing, subject–relation–object triples — and derives a 0–10 complexity score.
3. Runs **vector RAG**: MiniLM embeddings → cosine top-3 → Llama-3.1-8B.
4. Runs **GraphRAG**: per-question NetworkX knowledge graph → entity lookup → 2-hop traversal → top-25 triples + source passages → Llama-3.1-8B.
5. Scores both with LLM-as-a-Judge and breaks results down by question type and reasoning depth.

## Headline results

| Metric | RAG | GraphRAG |
|---|---|---|
| Overall accuracy | 32.6% | 30.0% |
| Refusal rate | 57.2% | 19.6% |
| 2-step accuracy | 46.2% | 36.9% |
| 4-step accuracy | 18.5% | 23.0% |

GraphRAG trades a small amount of overall accuracy for a **3× reduction in refusals** and better performance where multi-hop reasoning is actually required.

## Outputs

| File | Contents |
|---|---|
| `nlp_ozellikleri.csv` | Per-question linguistic features |
| `rag_degerlendirilmis.csv` | RAG answers + correctness labels |
| `graphrag_degerlendirilmis.csv` | GraphRAG answers + correctness labels |
| `rag_vs_graphrag_sonuclar.png` | Four-panel comparison figure |
| `adim_sayisi_vs_performans.png` | Accuracy and refusal rate vs reasoning depth |

The two evaluated CSVs are the input to Stage 2.

## Setup

```bash
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm
export GROQ_API_KEY="your-key-here"
```

Runtime is roughly 30 minutes per system on the 500-question subset, dominated by API latency and rate limiting.
