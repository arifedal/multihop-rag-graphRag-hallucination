# Stage 2 — Predicting Hallucination Risk with LoRA

Notebook: `LoRA_Halusinasyon_Siniflandirma.ipynb`

## The question

Can we tell, from a question alone, that a retrieval-augmented system is going to get it wrong — before it answers?

## Labelling

A question is labelled `1` (high hallucination risk) when **both** RAG and GraphRAG from Stage 1 answered it incorrectly, and `0` when at least one succeeded. This gives 234 positive / 266 negative examples out of 500 — close to balanced, so accuracy and macro F1 stay interpretable.

Split: 400 train / 100 test, stratified.

## Model

A hybrid classifier over two complementary signals:

- **Semantic** — DistilBERT with a LoRA adapter (`r=8`, `α=16`, targeting `q_lin` and `v_lin`, dropout 0.1), CLS token pooled.
- **Linguistic** — 8 hand-built features from Stage 1 (reasoning steps, complexity score, entity count, verb count, clause count, unique entities, person count, question type) passed through a small dense block.

The two are concatenated and classified by a 128-unit head. Ten epochs, AdamW, lr 2e-4.

Only **147,456 of 66,510,336 parameters (0.22%)** are updated — the point of LoRA here is that a full fine-tune would badly overfit 400 examples.

## Results

| Model | Macro F1 |
|---|---|
| Random Forest (features only) | 0.580 |
| Logistic Regression (features only) | 0.667 |
| TF-IDF (text only) | 0.667 |
| **LoRA + DistilBERT + features** | **0.706** |

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Correct answer (0) | 0.71 | 0.77 | 0.74 | 53 |
| Hallucination risk (1) | 0.71 | 0.64 | 0.67 | 47 |

Overall accuracy 71%.

## What the ablation shows

Text alone reaches 0.667. Features alone reach 0.667. Together they reach 0.706 — the signals are complementary, not redundant. That is the main justification for the hybrid architecture.

## Which features matter

From Random Forest importances:

| Feature | Importance |
|---|---|
| Complexity score | 0.578 |
| Person-entity count | 0.108 |
| Verb count | 0.071 |
| Entity count | 0.057 |
| Unique entities | 0.053 |
| Clause count | 0.047 |
| Question type | 0.044 |
| Reasoning-step count | 0.043 |

The nominal number of hops is the *weakest* predictor. How syntactically tangled a question is predicts failure far better than how many hops it formally requires — which suggests the difficulty is at least partly in parsing the question, not only in chaining the retrieval.

## Limitations

The test set is 100 examples, so small differences are not significant. Labels come from one generator model and one retrieval pipeline, so "hallucination risk" is pipeline-specific rather than intrinsic to the question.

## Prerequisites

Run Stage 1 first — this notebook reads `rag_degerlendirilmis.csv`, `graphrag_degerlendirilmis.csv` and `nlp_ozellikleri.csv`.
