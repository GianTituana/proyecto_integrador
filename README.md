# Índice de Riesgo de Corrupción — Entregable 2

**Trabajo de Fin de Carrera — Universidad San Francisco de Quito (USFQ)**

> Diseño e implementación de un índice de riesgo de corrupción basado en preguntas acusatorias del Sistema Oficial de Contratación Pública del Ecuador (SOCE) para el proceso de Giro Específico de Negocio (GEN) mediante técnicas de Machine Learning y Procesamiento de Lenguaje Natural.

**Autor:** Gian Martín Tituana Intriago
**Tutor:** Daniel Riofrío, Ph.D.
**Quito, mayo de 2026**

---

## Contexto

El SOCE centraliza la información de los procesos contractuales del Ecuador, incluyendo preguntas y respuestas publicadas durante las licitaciones. Los procesos GEN pueden contener preguntas acusatorias —intervenciones textuales con señalamientos explícitos o implícitos de irregularidades— que actualmente se analizan con heurísticas limitadas (full-text search sobre 8 palabras clave) dentro del ecosistema **Kapak**.

Este proyecto replica y adapta el protocolo experimental de Roh López (2024), construyendo un pipeline completo de PLN para clasificar preguntas acusatorias y transformar la salida probabilística de un ensemble de modelos en un índice continuo de riesgo de corrupción.

---

## Estructura del repositorio

```
entregable_2-3/
├── datasets/                                  # Conjuntos de datos (Git LFS)
│   ├── train_dataset.csv                      # Dataset de entrenamiento
│   ├── test_dataset.csv                       # Dataset de evaluación (n=1,001)
│   ├── GEN2008_2023.csv                       # Dataset histórico GEN 2008-2023
│   ├── calculo_postgre_SIE_2024_*.csv         # Preguntas SIE 2024 (sin aclaraciones)
│   └── resumen_densidad_mensual.csv           # Resumen de densidad mensual por proceso
│
├── pipelines/                                 # Notebooks de experimentación (Git LFS)
│   │
│   ├── — Clasificación y embeddings —
│   ├── kapak_indicador4_replica.ipynb         # Réplica del protocolo Roh (2024)
│   ├── kapak_indicador4_pipeline.ipynb        # Pipeline de inferencia: embeddings + ensemble
│   ├── kapak_gen_indicador4.ipynb             # Pipeline completo sobre GEN 2008-2023
│   │
│   ├── — Índice de riesgo (IRC) —
│   ├── kapak_indicador4_indice.ipynb          # Implementación operativa del IRC4
│   ├── kapak_indicador4_irc_media.ipynb       # IRC continuo: media de probabilidades
│   ├── kapak_indicador4_irc_media_binaria.ipynb  # IRC por media de predicciones binarizadas
│   │
│   ├── — LLM Judge —
│   ├── kapak_llm_judge.ipynb                  # Evaluación con o4-mini/low (muestra 2024, n=4 079)
│   ├── kapak_gen_llm_judge.ipynb              # Evaluación con o4-mini/medium sobre GEN 2008-2023
│   ├── kapak_llm_judge_ablacion.ipynb         # Ablación: o4-mini/low vs gpt-5-mini/medium
│   │
│   ├── — Comparación y validación —
│   ├── kapak_comparacion_gt_bd_vs_llm.ipynb   # BD vs LLM Judge como ground truth (2024)
│   ├── kapak_muestreo_representativo.ipynb    # Muestreo estratificado para validación IRC4
│   ├── kapak_distribucion_procesos.ipynb      # Distribución de densidades del ensemble por proceso
│   │
│   └── baseline_bd/
│       ├── kapak_baseline_bd.ipynb            # Comparación baseline BD vs. ensemble ML
│       └── output/                            # Métricas y gráficos generados
│
├── llm_judge_project/                         # Desarrollo y calibración del prompt LLM Judge
│   ├── prompts/                               # Historial de versiones del prompt (v1–v17)
│   ├── notebooks/
│   │   └── prompt_alignment.ipynb             # Alineación y evaluación de prompts
│   └── data/
│       ├── splits/                            # Particiones train/test para el LLM Judge
│       └── results/                           # Predicciones, métricas y logs por iteración
│
├── sql/                               # Consultas de extracción de datos
│   ├── consultas.sql                  # Consultas sobre procesos GEN/SIE 2022
│   └── gen_soce_indicador.sql         # Cálculo del indicador en la BD
│
├── .env.example                       # Plantilla de variables de entorno
├── .gitattributes                     # Configuración Git LFS
└── .gitignore
```

> Las carpetas `contexto/` (documentación) y `postgres/` (exportaciones CSV de la BD) están excluidas del repositorio por `.gitignore`.

---

## Modelos del ensemble

El indicador se construye sobre tres modelos con perfiles complementarios, entrenados con embeddings `text-embedding-3-large` (OpenAI, 3,072 dimensiones):

| Modelo | Algoritmo | Perfil | F1 | Recall | Precision |
|--------|-----------|--------|----|--------|-----------|
| M1 | Logistic Regression | Equilibrado | 0.677 | 0.759 | 0.611 |
| M2 | Naive Bayes | Sensible (Recall=1.0) | 0.227 | 1.000 | 0.128 |
| M3 | Random Forest | Conservador | 0.640 | 0.552 | 0.762 |

**Sistema de votación:** 3/3 votos → alerta ROJA · 2/3 → AMARILLA · 1/3 → VERDE

---

## Setup

### 1. Variables de entorno

```bash
cp .env.example .env
# Completar con credenciales reales en .env
```

Variables requeridas:

```
OPENAI_API_KEY=...   # Para generar embeddings con text-embedding-3-large
DB_HOST=...          # Host PostgreSQL (Kapak)
DB_PORT=5432
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
```

### 2. Dependencias

```bash
pip install pandas scikit-learn openai sqlalchemy psycopg2-binary \
            matplotlib seaborn python-dotenv scipy tqdm
```

### 3. Git LFS

Los datasets y notebooks se almacenan en Git LFS. Para descargarlos:

```bash
git lfs install
git lfs pull
```

---

## LLM Judge Project

El directorio `llm_judge_project/` centraliza el desarrollo iterativo del prompt de clasificación usado por el LLM Judge (`o4-mini`) como ground truth externo.

| Componente | Descripción |
|---|---|
| `prompts/prompt_v1.txt` … `prompt_v17.txt` | Historial de versiones del prompt (v17 es la versión final en producción) |
| `notebooks/prompt_alignment.ipynb` | Evaluación de alineación del prompt con el dataset de referencia Roh (2024) |
| `data/splits/` | Particiones train/test para evaluar el prompt |
| `data/results/iteration_logs/` | Predicciones y logs por cada iteración de prompt (iter_01–iter_14) |
| `data/results/final_evaluation/` | Métricas y predicciones finales (prompt_v17) |
| `data/results/external_test/` | Evaluación del prompt final sobre test set externo |

El prompt final (`prompt_v17.txt`) es el que se usa en `kapak_llm_judge.ipynb` y `kapak_gen_llm_judge.ipynb`.

---

## Pipelines

### Clasificación y embeddings

#### `kapak_indicador4_replica.ipynb`
Réplica del protocolo experimental de Roh López (2024): preprocesamiento, balanceo, data augmentation (sinónimos, BERT, GPT-4o mini) y entrenamiento de los tres modelos del ensemble.

#### `kapak_indicador4_pipeline.ipynb`
Pipeline principal de inferencia sobre el test set SIE 2022: genera embeddings con `text-embedding-3-large`, aplica M1/M2/M3 y construye el índice de riesgo agregado.

#### `kapak_gen_indicador4.ipynb`
Replica el pipeline de embeddings e inferencia sobre el dataset histórico `GEN2008_2023.csv` (659 preguntas, 111 procesos), calcula el IRC4 y el promedio de probabilidades M1 por proceso.

---

### Índice de riesgo (IRC)

#### `kapak_indicador4_indice.ipynb`
Implementación operativa del IRC4: transforma las probabilidades a nivel de pregunta en un índice continuo por proceso (escala 0–100) con componentes de intensidad, frecuencia y severidad.

#### `kapak_indicador4_irc_media.ipynb`
Construye el IRC continuo usando exclusivamente la media de las probabilidades de cada modelo: $IRC_p^{(m)} = 100 \times \frac{1}{n_p}\sum P_m(c_i)$. Sin reglas de contexto ni taxonomía.

#### `kapak_indicador4_irc_media_binaria.ipynb`
Variante del IRC anterior con predicciones binarizadas (umbral 0.5), lo que permite comparar en la misma escala conceptual que la BD y el LLM Judge.

---

### LLM Judge

#### `kapak_llm_judge.ipynb`
Evaluación asíncrona de las 4 079 preguntas de la muestra estratificada 2024 (471 procesos) usando `o4-mini` con `reasoning_effort='low'` como ground truth externo. Incluye sistema de checkpoints y resume automático.

#### `kapak_gen_llm_judge.ipynb`
Evaluación equivalente sobre el dataset histórico GEN 2008-2023 usando `o4-mini` con `reasoning_effort='medium'`. Compara el LLM Judge contra las etiquetas BD, calcula concordancia (Kappa, MCC) y genera gráficos KDE de todas las fuentes vs LLM Judge. Costo real: ~$1.65 USD.

#### `kapak_llm_judge_ablacion.ipynb`
Ablación sobre una muestra estratificada de ~32 preguntas: compara el baseline `o4-mini/low` contra `gpt-5-mini/medium` para determinar si un modelo con mayor capacidad de razonamiento cambia las clasificaciones en casos límite.

---

### Comparación y validación

#### `kapak_comparacion_gt_bd_vs_llm.ipynb`
Análisis comparativo central del entregable 3: evalúa el rendimiento de M1, M2, M3 y Ensemble en el dataset 2024 usando el LLM Judge como ground truth (no la BD). Incluye:
- Concordancia pregunta a pregunta (BD vs LLM Judge)
- Matrices de confusión y test de McNemar por modelo
- Distribuciones KDE con métricas Wasserstein, KL continua y KL discreta vs LLM Judge
- Scatter IRC por modelo vs BD y vs LLM Judge (RMSE, MAE, Spearman)
- Análisis de discrepancias BD/LLM y exportación de FP/FN por fuente

#### `kapak_muestreo_representativo.ipynb`
Determina el tamaño y composición de la muestra estratificada (2 estratos: procesos acusatorios vs no acusatorios) para la validación del IRC4.

#### `kapak_distribucion_procesos.ipynb`
Analiza la distribución de densidades de probabilidad del ensemble para cada proceso GEN clasificado durante 2024.

---

### Baseline

#### `baseline_bd/kapak_baseline_bd.ipynb`
Compara el método actual de la BD (full-text search sobre 8 palabras clave, vista `mv_gen_proceso_indicador_04`) contra los tres modelos ML usando el test set original. Soporta ejecución con o sin conexión a la BD (fallback local con regex).

---

## Baseline de la BD (referencia)

```sql
to_tsvector('spanish', pregunta) @@ to_tsquery('spanish',
    'corrupción|direccionado|limitante|vulneración
    |ocultamiento|violación|incompleto|trato<->justo')
```

Clasifica como acusatorio si contiene al menos una de las 8 palabras clave.

---

## Referencias

- Roh López, F. (2024). *Detección de comentarios acusatorios en contratación pública mediante PLN*. USFQ.
- Tituana Intriago, G. M. (2026). *Diseño e implementación de un índice de riesgo de corrupción basado en preguntas acusatorias del SOCE para procesos GEN*. USFQ.
