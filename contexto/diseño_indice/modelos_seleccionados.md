# Modelos Seleccionados — Proyecto Kapak (Indicador 4)

> Basado en `contexto/esquemas_finales.md`  
> Roh López, F. (2024). *Detección de comentarios acusatorios en procesos de contratación pública del Ecuador.* USFQ.

---

## Contexto

Los tres modelos provienen del catálogo experimental de 1.658 configuraciones documentadas en `Models_Results.xlsx`. Todos comparten `text-embedding-3-large` como representación de texto, lo que permite calcular el embedding **una sola vez** por comentario y reutilizarlo en los tres clasificadores (~66 % de ahorro en API).

**Criterios de selección comunes:**
- AUC-ROC ≥ 0,90
- Representación: `text-embedding-3-large` (3.072 dimensiones, OpenAI)
- Cada modelo ocupa un perfil distinto en el espacio Recall–Precision

---

## M1 — Modelo Equilibrado

| Atributo | Valor |
|---|---|
| **Algoritmo** | Logistic Regression |
| **Representación** | `text-embedding-3-large` (3.072 dims) |
| **Dataset** | Balanced 4 → `balanced_total_sinonimos` (3.886 muestras/clase) |
| **Aumentación** | Sinónimos (WordNet, `nlpaug`) |
| **Hiperparámetros** | `C=100`, `penalty='l2'`, `solver='saga'` |

| AUC-ROC | Recall | F1 Score | Precision | TP | FN | FP | TN | Rank global |
|---|---|---|---|---|---|---|---|---|
| 0,940 | 0,759 | **0,677** | 0,611 | 22 | 7 | 14 | 958 | **#1 / 1.658** |

**Criterio de selección:** máximo F1 Score del experimento completo.

**Embedding en servidor:**
```
/backup/Sandbox/froh/FinalProject_cl/embeddings/
    train_embeddings_balanced_total_sinonimos_text-embedding-3-large.csv
```

---

## M2 — Modelo Sensible

| Atributo | Valor |
|---|---|
| **Algoritmo** | Naive Bayes (GaussianNB) |
| **Representación** | `text-embedding-3-large` (3.072 dims) |
| **Dataset** | Consolidated Data Augmentation → `balanced_AUG_GPT4o_mini_total_augmented` (3.886–21.766 muestras) |
| **Aumentación** | Todas las técnicas consolidadas (WordNet + BERT + GPT-2 + GPT-4o mini v1 y v2) |
| **Hiperparámetros** | `var_smoothing=0.35111917342151305` |

| AUC-ROC | Recall | F1 Score | Precision | TP | FN | FP | TN | Rank global |
|---|---|---|---|---|---|---|---|---|
| 0,940 | **1,000** | 0,227 | 0,128 | 29 | **0** | 198 | 774 | #1023 / 1.658 |

**Criterio de selección:** máximo Recall — cero falsos negativos. El ranking por F1 no aplica para este modelo; si el ranking fuera por Recall, sería #1.

**Embedding en servidor:**
```
/backup/Sandbox/froh/FinalProject_cl/embeddings/
    train_embeddings_balanced_AUG_GPT4o_mini_total_augmented_text-embedding-3-large.csv
```

---

## M3 — Modelo Conservador

| Atributo | Valor |
|---|---|
| **Algoritmo** | Random Forest Classifier |
| **Representación** | `text-embedding-3-large` (3.072 dims) |
| **Dataset** | Balanced 4 → `balanced_total_sentence_prompt_GPT4o_mini` (3.886 muestras/clase) |
| **Aumentación** | GPT-4o mini con prompt v1 |
| **Hiperparámetros** | `max_depth=None`, `min_samples_leaf=1`, `n_estimators=200` |

| AUC-ROC | Recall | F1 Score | Precision | TP | FN | FP | TN | Rank global |
|---|---|---|---|---|---|---|---|---|
| 0,940 | 0,552 | 0,640 | **0,762** | 16 | 13 | **5** | 967 | #12 / 1.658 |

**Criterio de selección:** máxima Precision entre configuraciones con F1 ≥ 0,45 y Recall ≥ 0,45. El filtro garantiza que el modelo aún detecte una fracción razonable de casos reales.

**Embedding en servidor:**
```
/backup/Sandbox/froh/FinalProject_cl/embeddings/
    train_embeddings_balanced_total_sentence_prompt_GPT4o_mini_text-embedding-3-large.csv
```

---

## Tabla comparativa

| | M1 — Equilibrado (LR) | M2 — Sensible (NB) | M3 — Conservador (RFC) |
|---|---|---|---|
| **Dataset** | Balanced 4 | Consolidated DA | Balanced 4 |
| **Aumentación** | Sinónimos | Todas | GPT-4o mini v1 |
| **AUC-ROC** | 0,940 | 0,940 | 0,940 |
| **Recall** | 0,759 | **1,000** | 0,552 |
| **Precision** | 0,611 | 0,128 | **0,762** |
| **F1 Score** | **0,677** | 0,227 | 0,640 |
| **FN** | 7 | **0** | 13 |
| **FP** | 14 | 198 | **5** |
| **Rank por F1** | #1 | #1023 | #12 |
| **Métrica de selección** | max(F1) | max(Recall) | max(Precision) |

---

## Test set compartido

Los tres modelos usan el **mismo conjunto de prueba fijo** (20 % estratificado, `random_state=72`):

```
/backup/Sandbox/froh/FinalProject_cl/embeddings/
    test_embeddings_text-embedding-3-large.csv
```

---

*Documento generado el 11 de marzo de 2026*
