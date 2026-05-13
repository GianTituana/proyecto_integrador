# Índice de Señales Textuales de Acusatoriedad — Proyecto Final

**Trabajo de Fin de Carrera — Universidad San Francisco de Quito (USFQ)**

> Diseño e implementación de un índice de señales textuales de acusatoriedad
en procesos de compras públicas (GEN) mediante Procesamiento de
Lenguaje Natural

**Autor:** Gian Martín Tituana Intriago
**Tutor:** Daniel Riofrío, Ph.D.
**Quito, mayo de 2026**

---

## Contexto

El SOCE centraliza la información de los procesos contractuales del Ecuador, incluyendo preguntas y respuestas publicadas durante las licitaciones. Los procesos GEN pueden contener preguntas acusatorias —intervenciones textuales con señalamientos explícitos o implícitos de irregularidades— que actualmente se analizan con heurísticas limitadas (full-text search sobre 8 palabras clave) dentro del ecosistema **Kapak**.

Este proyecto replica y adapta el protocolo experimental de Roh López (2024), construyendo un pipeline completo de PLN para clasificar preguntas acusatorias y transformar la salida probabilística de un ensemble de modelos en un índice continuo de acusatoriedad.

---

## Estructura del repositorio

```
proyecto_integrador/
├── datasets/                                      # Conjuntos de datos (Git LFS)
│   ├── calculo_postgre_SIE_2024_no_aclaracion_clasificado.csv
│   ├── GEN2008_2023.csv
│   └── trabajo_base/
│       ├── dataset.xlsx                           # Dataset Roh (base)
│       ├── train_dataset.csv                      # Split entreno (Roh)
│       ├── test_dataset.csv                       # Split test (Roh)
│       └── embeddings/                            # Embeddings base (CSV)
│
├── pipelines/                                     # Notebooks y artefactos
│   ├── kapak_replica_modelos.ipynb                # Entrenamiento M1/M2/M3
│   ├── indice_IFS/
│   │   └── kapak_indicador4_indice.ipynb          # IFS (primera etapa, futuro indice operativo)
│   ├── procesos_SIE/
│   │   ├── kapak_indicador4_pipeline.ipynb        # Pipeline SIE 2024
│   │   ├── kapak_llm_judge.ipynb                  # LLM Judge SIE
│   │   ├── kapak_comparacion_gt_bd_vs_llm.ipynb   # Comparación BD vs LLM (GT)
│   │   ├── kapak_muestreo_representativo.ipynb    # Muestra estratificada
│   │   ├── kapak_indicador4_irc_media.ipynb       # IRC media probabilidades
│   │   └── kapak_indicador4_irc_media_binaria.ipynb
│   ├── procesos_GEN/
│   │   ├── kapak_gen_indicador4.ipynb             # Pipeline GEN 2008-2023
│   │   └── kapak_gen_llm_judge.ipynb              # LLM Judge GEN
│   ├── prompt_aligment/                           # Desarrollo de prompt LLM Judge
│   │   ├── requirements.txt
│   │   ├── prompts/
│   │   ├── notebooks/
│   │   └── data/
│   └── output_replica/                            # Salidas (ignorado por git)
│       ├── embeddings/
│       ├── evaluation/
│       │   ├── GEN/
│       │   ├── kapak_comparacion_gt/
│       │   ├── kapak_indicador4_irc_media/
│       │   └── kapak_indicador4_irc_media_binaria/
│       ├── models/
│       ├── plots/
│       ├── results_summary_replica.csv
│       └── gridsearch_*.csv
│
├── .env.example                                   # Plantilla de variables de entorno
├── .gitattributes                                 # Configuración Git LFS
└── .gitignore
```

> La carpeta `pipelines/output_replica/` se genera al ejecutar los notebooks.

---

## Modelos del ensemble

El indicador se construye sobre tres modelos con perfiles complementarios, entrenados con embeddings `text-embedding-3-large` (OpenAI, 3,072 dimensiones):

| Modelo | Algoritmo | Perfil | F1 | Recall | Precision |
|--------|-----------|--------|----|--------|-----------|
| M1 | Logistic Regression | Equilibrado | 0.677 | 0.759 | 0.611 |
| M2 | Naive Bayes | Sensible (Recall=1.0) | 0.227 | 1.000 | 0.128 |
| M3 | Random Forest | Conservador | 0.640 | 0.552 | 0.762 |

---

## Resultados resumidos (Informe Final)

- **Replicación de modelos (test):** M1 F1=0.677 (R=0.759, P=0.611), M2 R=1.000 pero F1=0.227 (P=0.128), M3 F1=0.640 (P=0.762).
- **Alineamiento del prompt (o4-mini, medium):** clase acusatoria P=0.70, R=0.72, F1=0.71; MCC=0.7032.
- **SIE 2024 (dataset completo):** 123,669 preguntas, 15,087 procesos, prevalencia acusatoria 4.78%; 78.8% de procesos sin preguntas acusatorias.
- **Muestreo estratificado:** 471 procesos; costo aproximado LLM Judge (o4-mini, low) $8.39.

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

Para el flujo de LLM Judge (prompt alignment):

```bash
pip install -r pipelines/prompt_aligment/requirements.txt
```

### 3. Git LFS

Los datasets y notebooks se almacenan en Git LFS. Para descargarlos:

```bash
git lfs install
git lfs pull
```

---

## LLM Judge Project

El directorio `pipelines/prompt_aligment/` centraliza el desarrollo iterativo del prompt de clasificación usado por el LLM Judge (`o4-mini`) como ground truth externo.

| Componente | Descripción |
|---|---|
| `prompts/` | Historial de versiones del prompt (v1–v17) |
| `notebooks/` | Alineación y evaluación de prompts |
| `data/splits/` | Particiones train/test para evaluar el prompt |
| `data/results/iteration_logs/` | Predicciones y logs por cada iteración de prompt |
| `data/results/final_evaluation/` | Métricas y predicciones finales (prompt_v17) |

El prompt final (`prompt_v17.txt`) es el que se usa en `pipelines/procesos_SIE/kapak_llm_judge.ipynb` y `pipelines/procesos_GEN/kapak_gen_llm_judge.ipynb`.

---

## Flujo recomendado de ejecución

1. **Preparación**
    - Configura `.env` con `OPENAI_API_KEY`.
    - Instala dependencias generales y las de `prompt_aligment`.
2. **Entrenamiento de modelos (M1/M2/M3)**
    - Ejecuta `pipelines/kapak_replica_modelos.ipynb`.
    - Genera modelos en `pipelines/output_replica/models/`.
3. **Pipeline SIE 2024**
    - Ejecuta `pipelines/procesos_SIE/kapak_indicador4_pipeline.ipynb`.
    - Genera `evaluacion_preguntas_completo.csv`, `evaluacion_procesos_completo.csv`, `indice_riesgo_procesos.csv` en `pipelines/output_replica/evaluation/`.
4. **Muestreo estratificado (471 procesos)**
    - Ejecuta `pipelines/procesos_SIE/kapak_muestreo_representativo.ipynb`.
    - Genera `muestra_final_471_procesos.csv`.
5. **LLM Judge SIE**
    - Ejecuta `pipelines/procesos_SIE/kapak_llm_judge.ipynb`.
    - Genera `llm_judge_results.csv` y checkpoints en `pipelines/output_replica/evaluation/`.
6. **Comparación GT (BD vs LLM)**
    - Ejecuta `pipelines/procesos_SIE/kapak_comparacion_gt_bd_vs_llm.ipynb`.
    - Genera `pipelines/output_replica/evaluation/kapak_comparacion_gt/`.
7. **IFS (primera etapa, no operativo)**
    - Ejecuta `pipelines/indice_IFS/kapak_indicador4_indice.ipynb`.
    - Este notebook alimenta un futuro indice operativo.

Para GEN 2008-2023:

1. `pipelines/procesos_GEN/kapak_gen_indicador4.ipynb`
2. `pipelines/procesos_GEN/kapak_gen_llm_judge.ipynb`

---

## Entradas y salidas principales

**Entradas clave**
- `datasets/trabajo_base/dataset.xlsx` (base Roh)
- `datasets/trabajo_base/train_dataset.csv` / `test_dataset.csv`
- `datasets/calculo_postgre_SIE_2024_no_aclaracion_clasificado.csv`
- `datasets/GEN2008_2023.csv`

**Salidas clave (se regeneran en `pipelines/output_replica/`)**
- `embeddings/` — matrices `.npz` y CSV de embeddings
- `models/` — modelos M1/M2/M3
- `evaluation/evaluacion_preguntas_completo.csv`
- `evaluation/evaluacion_procesos_completo.csv`
- `evaluation/indice_riesgo_procesos.csv`
- `evaluation/llm_judge_results.csv`
- `evaluation/kapak_comparacion_gt/` (comparaciones BD vs LLM)
- `evaluation/GEN/` (salidas del flujo GEN)

---

## Pipelines

### Clasificación y embeddings

#### `kapak_indicador4_replica.ipynb`
Réplica del protocolo experimental de Roh López (2024): preprocesamiento, balanceo, data augmentation (sinónimos, BERT, GPT-4o mini) y entrenamiento de los tres modelos del ensemble.

#### `kapak_indicador4_pipeline.ipynb`
Pipeline principal de inferencia sobre el test set SIE 2024: genera embeddings con `text-embedding-3-large`, aplica M1/M2/M3 y construye el índice de riesgo agregado.

#### `kapak_gen_indicador4.ipynb`
Replica el pipeline de embeddings e inferencia sobre el dataset histórico `GEN2008_2023.csv` (659 preguntas, 111 procesos), calcula el IRC4 y el promedio de probabilidades M1 por proceso.

---

### Índice de riesgo (IRC)

#### `kapak_indicador4_indice.ipynb`
Primera etapa del IFS (no operativo): transforma las probabilidades a nivel de pregunta en un indice continuo por proceso (escala 0-100) con componentes de intensidad, frecuencia y severidad. Este resultado sera parte de un futuro indice operativo.

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

## Baseline de la BD (referencia de acusatoriedad)

```sql
to_tsvector('spanish', pregunta) @@ to_tsquery('spanish',
    'acusatoriedad|direccionado|limitante|vulneración
    |ocultamiento|violación|incompleto|trato<->justo')
```

Clasifica como acusatorio si contiene al menos una de las 8 palabras clave.

---

## Referencias

- Roh López, F. (2024). *Detección de comentarios acusatorios en contratación pública mediante PLN*. USFQ.
- Tituana Intriago, G. M. (2026). *Diseño e implementación de un índice de señales textuales de acusatoriedad en procesos de compras públicas (GEN) mediante Procesamiento de Lenguaje Natural*. USFQ.
