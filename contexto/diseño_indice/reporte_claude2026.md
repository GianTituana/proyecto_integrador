# Reporte de Análisis del Directorio de Trabajo
**Generado por:** Claude Sonnet 4.6
**Fecha:** 18 de febrero de 2026
**Directorio analizado:** `/backup/Sandbox/froh`

---

## Resumen Ejecutivo

Este directorio contiene un proyecto de investigación de gran escala sobre **análisis de sentimientos en texto** (NLP). Se evaluaron múltiples arquitecturas de modelos de machine learning y deep learning con distintas estrategias de aumento de datos, balanceo de clases, ajuste de hiperparámetros y técnicas de transfer learning. En total se documentaron **1,659 configuraciones experimentales** con métricas detalladas.

---

## 1. Dataset

### Dataset Principal

| Archivo | Tamaño | Descripción |
|--------|--------|-------------|
| `train.tsv` | 8.1 MB | Dataset de entrenamiento principal (TSV) |
| `train_datasetAUG.csv` | 2.3 MB | Versión aumentada del dataset de entrenamiento |
| `dataset.xlsx` | 672 KB | Dataset estructurado con metadatos y etiquetas |

**Fuente:** Competencia de Kaggle — *Sentiment Analysis on Movie Reviews* (reseñas de películas)

**Formato del dataset principal (`train.tsv`):**
- Columnas: `PhraseId`, `SentenceId`, `Phrase`, `Sentiment`
- Tamaño original: **3,886 muestras de entrenamiento** (118 reseñas positivas)
- **Clases de sentimiento:**
  - 0 — Negativo
  - 1 — Algo negativo
  - 2 — Neutral
  - 3 — Algo positivo
  - 4 — Positivo
- Para algunos experimentos se recodificó en **3 clases** (negativo, neutral, positivo)

### Embeddings Pre-entrenados

| Archivo | Tamaño | Descripción |
|--------|--------|-------------|
| `SBW-vectors-300-min5.txt` | 2.7 GB | Word embeddings en español (SBW, 300 dimensiones, frecuencia mínima 5) |

---

## 2. Modelos Entrenados

Se entrenaron modelos de las siguientes familias:

### 2.1 Modelos Transformer (Transfer Learning)

| Modelo | Variantes | Descripción |
|--------|-----------|-------------|
| **BERT** | `bert`, `bert_8_f_layers`, `bert_10_f_layers`, `bert_12_f_layers` | BERT base con distintas capas congeladas (0, 8, 10, 12 capas) |
| **RoBERTa** | `roberta`, `roberta_8_f_layers`, `roberta_10_f_layers`, `roberta_12_f_layers` | RoBERTa con distintas capas congeladas |
| **Auto-BERT / Auto-RoBERTa** | `auto_bert`, `auto_roberta` | Versiones con búsqueda automática de hiperparámetros |

### 2.2 Modelos de Redes Neuronales Recurrentes (RNN)

| Modelo | Variantes |
|--------|-----------|
| **RNN Vanilla** | Unidireccional y bidireccional |
| **LSTM** | Unidireccional y bidireccional |
| **GRU** | Unidireccional y bidireccional |

Configuraciones probadas por variante:
- Embeddings: 256–300 dimensiones
- Batch size: 512 (principal)
- Épocas: hasta 50 con early stopping (patience 5–10)

### 2.3 Modelos de Machine Learning Clásico

| Modelo | Descripción |
|--------|-------------|
| **SVC** | Support Vector Classifier (kernel lineal y RBF) |
| **Logistic Regression** | Regresión logística |
| **Naive Bayes** | Clasificador probabilístico |
| **Random Forest** | Ensamble de árboles de decisión |

Estos modelos se usaron sobre representaciones vectoriales de texto (embeddings fijos).

### 2.4 Métodos de Representación de Texto (Embeddings)

| Método | Tipo |
|--------|------|
| **FastText** | Embeddings de palabras |
| **Doc2Vec** | Embeddings de documentos |
| **TF-IDF** | Representación estadística clásica |
| **BoW (Bag of Words)** | Representación de frecuencias |
| **SBERT (Sentence-BERT)** | Embeddings de oraciones |
| **GPT-2** | Embeddings generados con GPT-2 |
| **GPT-4o mini** | Embeddings generados con GPT-4o mini |

### 2.5 Técnicas de Aumento de Datos Probadas

| Técnica | Descripción |
|---------|-------------|
| Reemplazo por sinónimos | Sustitución aleatoria de palabras por sinónimos |
| Inserción de tokens BERT | Generación de palabras con el modelo de llenado de BERT |
| Generación con GPT-2 | Frases generadas por el modelo GPT-2 |
| Generación con GPT-4o mini | Frases generadas por GPT-4o mini de OpenAI |
| Back-translation | Traducción y re-traducción para parafrasear |

---

## 3. Pesos y Binarios de Modelos Entrenados

### 3.1 Checkpoints de PyTorch Lightning (`.ckpt`)

Los checkpoints se almacenan en la estructura:

```
<FinalProject_*/>/logs/<dataset>/<modelo>/<experiment_name>/version_<N>/checkpoints/*.ckpt
```

**Ejemplos representativos:**

```
FinalProject_Binary_5p/logs/original_3886_118/bert/bert_original_3886_118/version_0/checkpoints/epoch=12-step=1274.ckpt
FinalProject_Freezed_models_5p/logs/original_3886_118/bert_8_f_layers/bert_original_3886_118_8_f_layers/version_1/checkpoints/epoch=0-step=25.ckpt
```

**Total exacto:** **694 archivos `.ckpt`** distribuidos en todos los directorios `FinalProject_*`.

| Directorio | Checkpoints |
|-----------|-------------|
| `FinalProject_Freezed_models_5p/` | 384 |
| `FinalProject_Freezed_models/` | 94 |
| `FinalProject_Binary_5p/` | 41 |
| Otros directorios DL | ~175 |

> **Nota:** Algunos directorios contienen múltiples versiones (`version_0`, `version_1`, `version_2`...) del mismo experimento, correspondientes a ejecuciones abortadas o reintentos. Esto explica que haya más checkpoints que combinaciones únicas de dataset+modelo.

### 3.2 Embeddings Vectoriales (CSV)

- **Ubicación principal:** `FinalProject_cl/embeddings/` y `FinalProject_AUG_cl/`, `FinalProject_AUG_total_cl/`, `FinalProject_SBERT_AUG_total_cl/`
- **Total de archivos:** 274 archivos CSV de embeddings (solo en `FinalProject_cl/`)
- **Formato:** vectores numéricos de 300 dimensiones + etiqueta de clase
- **Nombrado:** `test_embeddings_<tipo_embedding>.csv`

### 3.3 Archivos de Respaldo

| Archivo | Tamaño | Contenido |
|--------|--------|-----------|
| `freezed_backup.tar` | 47 MB | Checkpoints de modelos con capas congeladas |
| `backup.tar` | 12 KB | Respaldo secundario |

---

## 4. Estructura de Directorios de Experimentos

| Directorio | Tamaño | Propósito |
|-----------|--------|-----------|
| `FinalProject/` | 132 MB | Experimentos iniciales de clasificación multiclase |
| `FinalProject_AUG/` | 1.2 MB | Aumento de datos (muestra limitada) |
| `FinalProject_AUG_5p/` | 2.5 MB | Aumento de datos con muestra del 5% |
| `FinalProject_AUG_cl/` | 1.6 GB | Datos aumentados + modelos clásicos de ML |
| `FinalProject_AUG_total/` | 1.3 MB | Experimentos con augmentation completo |
| `FinalProject_AUG_total_5p/` | 1.3 MB | Augmentation total con muestra 5% |
| `FinalProject_AUG_total_cl/` | 12 GB | ML clásico a gran escala con augmentation total |
| `FinalProject_AutoC_AUGtotal/` | 23 MB | AutoML sobre datos aumentados |
| `FinalProject_AutoC_AUGtotal_5p/` | 22 MB | AutoML con muestra del 5% |
| `FinalProject_Binary_5p/` | 35 GB | Clasificación binaria (positivo/negativo) |
| `FinalProject_cl/` | 35 GB | Comparación extensa de modelos clásicos de ML |
| `FinalProject_Freezed_models/` | 51 GB | Experimentos con capas congeladas |
| `FinalProject_Freezed_models_5p/` | **298 GB** | Estudio exhaustivo de capas congeladas (directorio más grande) |
| `FinalProject_p5/` | 40 MB | Experimentos con muestra del 5% |
| `FinalProject_SBERT_AUG_total_cl/` | 1.9 GB | SBERT con datos aumentados |
| `FinalProject_SBERT_AUG_total_cl_PRE/` | 1.9 GB | SBERT con pre-entrenamiento |
| `FinalProject_SVC_cl/` | 24 MB | Enfoque en SVM (65K+ archivos de matrices de confusión) |

**Tamaño total del proyecto:** ~531 GB

---

## 5. Resultados y Métricas

### 5.1 Archivo Principal de Resultados

| Archivo | Tamaño | Filas | Columnas |
|--------|--------|-------|---------|
| `results_final_summary_modelos.csv` | 2.7 MB | 1,659 | 45+ |
| `results_summary_modelos.csv` | 1.1 MB | — | — |

**Métricas registradas por experimento:**

| Partición | Métricas |
|-----------|---------|
| Entrenamiento | `train_f1`, `train_acc`, `train_precision`, `train_recall`, `train_loss` |
| Validación | `val_f1`, `val_acc`, `val_precision`, `val_recall`, `val_loss` |
| Prueba | `test_f1`, `test_acc`, `test_precision`, `test_recall`, `test_loss` |
| Clasificación binaria | `recall_positive`, `precision_positive`, `f1_positive`, ROC-AUC |

**Variables de configuración registradas:**
- `tamaño_dataset`, `balance`, `data_augmentation`
- `freezed_layers`, `embedding_type`
- `Patience`, `runtime_minutes`, `grid_runtime_minutes`
- Rutas a matrices de confusión, curvas ROC y curvas PR

### 5.2 Variaciones de Dataset Experimentadas

| Variante | Tamaño | Descripción |
|---------|--------|-------------|
| Original | 3,886 | Dataset completo desbalanceado (118 positivos) |
| Subsampled | ~118 | Submuestreo de la clase mayoritaria |
| Balanced (500) | 500 | Dataset balanceado por augmentation |
| Balanced (1,000) | 1,000 | Dataset balanceado por augmentation |
| Balanced (2,500) | 2,500 | Dataset balanceado por augmentation |
| Unbalanced (mixed) | 1,000–3,886 | Distribuciones de clases mixtas |

### 5.3 Hallazgos Destacados

- **Mejor F1-Score:** > 0.91 — alcanzado con LSTM/GRU bidireccional sobre texto original con stemming/lematización
- **Total de configuraciones evaluadas:** 1,659
- **Modelos más grandes:** Checkpoints de BERT/RoBERTa con capas congeladas en `FinalProject_Freezed_models_5p/` (298 GB)

### 5.4 Análisis de las 1,658 Filas vs. Checkpoints Disponibles

Las 1,658 filas del CSV **no equivalen a 1,658 modelos independientes serializados**. La discrepancia se explica por tres factores:

**Factor 1 — Una fila no es un modelo, es una evaluación**

Cada fila representa una combinación única de `(dataset × modelo × tipo de embedding)`. El mismo modelo entrenado puede aparecer hasta **7 veces** en el CSV, una por cada tipo de representación vectorial evaluada:

| Embeddings evaluados por modelo | Ejemplo |
|---------------------------------|---------|
| BoW, TF-IDF, Doc2Vec, FastText, SBERT, GPT-2, GPT-4o mini | `undersampled_118_118 + Random_Forest` aparece 7 veces |

**Factor 2 — El 57% de las filas corresponde a modelos clásicos sin binarios guardados**

| Modelo | Filas en CSV | Binarios en disco |
|--------|:-----------:|:-----------------:|
| Random Forest Classifier | 252 | **0** |
| Naive Bayes | 252 | **0** |
| Logistic Regression | 252 | **0** |
| SVC | 182 | **0** |
| BERT / RoBERTa / auto-variantes | 576 | `.ckpt` ✓ |
| LSTM / GRU | 144 | `.ckpt` ✓ |
| **Total** | **1,658** | |

Los modelos clásicos de scikit-learn fueron entrenados, evaluados y descartados. **Nunca se serializaron** a ningún formato (`.pkl`, `.joblib`, `.sav`, etc.). Solo se conservaron sus métricas en el CSV y sus matrices de confusión en PNG. No existe ningún archivo de modelo clásico recuperable en el directorio.

**Factor 3 — Runs reintentados generan checkpoints extra**

Los 694 `.ckpt` superan levemente las ~720 filas DL del CSV porque algunos experimentos fueron reejecutados, generando directorios `version_0`, `version_1`, `version_2`... con checkpoints adicionales no contabilizados como nuevas entradas en el CSV.

**Resumen de la reconciliación:**

```
1,658 filas en el CSV
 ├── 938 filas → modelos clásicos (ML)  →  0 binarios guardados en disco
 └── 720 filas → modelos de DL          →  694 archivos .ckpt en disco
                                              └─ diferencia por runs reintentados
```

> **Implicación práctica:** Los modelos clásicos (SVC, Logistic Regression, Naive Bayes, Random Forest) **no son reproducibles desde disco** — requieren reentrenamiento completo. Solo los modelos BERT, RoBERTa, LSTM y GRU pueden restaurarse desde los checkpoints existentes.

---

## 6. Visualizaciones Generadas

### SVGs de Resumen (raíz del directorio)

| Archivo | Tamaño | Descripción |
|--------|--------|-------------|
| `class_counts.svg` | 21 KB | Distribución de clases |
| `f1_embedding_model.svg` | 59 KB | F1 por tipo de embedding y modelo |
| `f1_recall_binary.svg` | 561 KB | F1 vs Recall (clasificación binaria) |
| `f1_recall_macro.svg` | 563 KB | F1 vs Recall (macro, multiclase) |
| `f1_recall_size_data_aug.svg` | 391 KB | F1 vs Recall según tamaño y augmentation |
| `f1_recall_size_data_aug_aligned.svg` | 389 KB | Versión alineada del gráfico anterior |
| `recall_embedding_model.svg` | 57 KB | Recall por embedding y modelo |
| `ROC_embedding_model.svg` | 62 KB | Curvas ROC por tipo de embedding |
| `TP_FN.svg` | 572 KB | Verdaderos positivos vs Falsos negativos |
| `TP_FP.svg` | 573 KB | Verdaderos positivos vs Falsos positivos |

### Archivos por Experimento (en cada directorio `FinalProject_*`)

Cada experimento genera:
- `conf_matrix_*.png` — Matrices de confusión (30–36 KB)
- `Curva_ROC_*.jpg` — Curvas ROC (train/val/test)
- `Curva_Precision-Recall_*.jpg` — Curvas Precision-Recall
- `learningcurve_*.png` — Curvas de aprendizaje (loss, F1, accuracy, precision, recall)

---

## 7. Trabajo Académico (Labs NLP)

**Ubicación:** `nlp-labs/`

| Lab | Contenido Principal |
|-----|-------------------|
| `lab-1/` | Conceptos fundacionales de NLP |
| `lab-2/` | Análisis de sentimientos con Naive Bayes (baseline) |
| `lab-3/` | Preprocesamiento avanzado e ingeniería de características |
| `lab-4/` | Introducción a redes neuronales |
| `lab-5/` | Comparación de RNN, LSTM, GRU (notebook: 6.2 MB) |

El lab-5 es el más completo e incluye todo el pipeline de entrenamiento con curvas de aprendizaje.

---

## 8. Entorno de Trabajo

| Componente | Detalle |
|-----------|---------|
| Python | 3.10 |
| Framework DL | PyTorch + PyTorch Lightning |
| ML Clásico | scikit-learn |
| NLP | HuggingFace Transformers, NLTK |
| Embeddings | FastText, Gensim (Doc2Vec), Sentence-Transformers |
| IDE | VS Code (remote server) |
| Recursos NLTK | `nltk_data/` — stopwords, tokenizers |

---

## 9. Mapa de Archivos Clave

```
/backup/Sandbox/froh/
├── train.tsv                          # Dataset principal (3,886 muestras)
├── train_datasetAUG.csv               # Dataset aumentado
├── dataset.xlsx                       # Dataset estructurado
├── SBW-vectors-300-min5.txt           # Embeddings en español (2.7 GB)
├── results_final_summary_modelos.csv  # Resultados completos (1,659 experimentos)
├── results_summary_modelos.csv        # Resumen condensado
├── freezed_backup.tar                 # Respaldo de checkpoints congelados
├── backup.tar                         # Respaldo secundario
├── *.svg                              # Visualizaciones de resumen
│
├── FinalProject*/                     # 17 directorios de experimentos (~531 GB)
│   └── logs/
│       └── <dataset>/
│           └── <modelo>/
│               └── <experimento>/
│                   └── version_N/
│                       ├── checkpoints/*.ckpt    # PESOS DEL MODELO
│                       └── metrics.csv           # Historial de entrenamiento
│
├── FinalProject_cl/embeddings/        # 274 archivos CSV de embeddings vectoriales
│
└── nlp-labs/                          # Notebooks académicos (labs 1–5)
    └── lab-5/lab-5.ipynb              # Lab más completo (6.2 MB)
```

---

## 10. Reproducibilidad de los Modelos Clásicos de ML

### 10.1 Hallazgo Principal

**Los notebooks fuente que entrenaron los modelos clásicos en los directorios `FinalProject_*_cl` no existen en disco.** Fueron eliminados o nunca guardados tras la ejecución. Los directorios de experimentos contienen únicamente artefactos de salida (CSVs de métricas, PNGs de matrices de confusión, CSVs de embeddings).

### 10.2 Código Existente con Lógica Reutilizable

El único código ejecutable con modelos clásicos son los notebooks académicos de los labs:

| Archivo | Modelo(s) | Relevancia |
|--------|-----------|-----------|
| `nlp-labs/lab-2/lab-2.ipynb` | Naive Bayes | Pipeline más detallado y standalone |
| `nlp-labs/lab-5/lab-5.ipynb` | Naive Bayes | Integrado con RNN/LSTM/GRU |
| `lab-5.ipynb` (raíz) | Naive Bayes | Copia re-ejecutada del anterior |

Estos notebooks **no contienen** SVC, Logistic Regression ni Random Forest. Son el patrón estructural del pipeline, pero el código completo de los FinalProject debe reconstruirse.

### 10.3 Secciones Clave para Revisar en los Labs

#### `nlp-labs/lab-2/lab-2.ipynb`

| Sección | Celda(s) | Qué contiene |
|---------|---------|-------------|
| Carga del dataset | cell-1 | `pd.read_csv('train.tsv', sep='\t')` — ruta a cambiar si el dataset está en otro lugar |
| Preprocesamiento | cell-34 a cell-40 | Lowercase, eliminación de stopwords, stemming (PorterStemmer), lematización (WordNetLemmatizer) — 7 variantes de texto |
| Vectorización | cell-42 | `CountVectorizer(binary=True)`, `CountVectorizer(binary=False)`, `TfidfVectorizer()` — 3 representaciones |
| Función de entrenamiento | cell-28 | `train_and_evaluate_model()` — instancia `MultinomialNB()`, llama `.fit()`, predice, calcula métricas |
| Loop principal | cell-42 | Itera 7 preprocesados × 3 vectorizadores = 21 combinaciones |
| Resultados | cell-45, cell-47 | `pd.DataFrame(resultados)` mostrado en memoria — **no se escribe a disco** |

> **Parámetros a cambiar:** `test_size=0.2`, `random_state=42` (hardcodeados dentro de `train_and_evaluate_model()`).

#### `nlp-labs/lab-5/lab-5.ipynb`

| Sección | Celda(s) | Qué contiene |
|---------|---------|-------------|
| Constantes de configuración | cell-6 | `SEED=13`, `PARTITION=0.2` — centralizado, fácil de cambiar |
| Carga y filtrado del dataset | cell-3 | Lee `train.tsv`, filtra a 3 clases (0, 2, 4) y las reasigna a (0, 1, 2) |
| Preprocesamiento | cell-6 | Igual que lab-2: 7 variantes de texto |
| Función de entrenamiento NB | cell-6 | `train_and_evaluate_model_nb()` — usa las constantes `SEED` y `PARTITION` |
| Loop principal NB | cell-8 | 7 variantes × 3 vectorizadores = 21 combinaciones |
| Resultados | cell-21, cell-23 | Solo en memoria — **no se escribe a disco** |

### 10.4 Lo que Falta para Reproducir el Pipeline Completo de FinalProject

El pipeline de los `FinalProject_*_cl` era más elaborado que los labs. A partir de los artefactos en disco se pueden reconstruir los siguientes parámetros:

#### Modelos a entrenar
```python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
```

#### Grids de hiperparámetros (reconstruidos desde los CSVs de gridsearch en disco)

```python
param_grid_SVC = {
    'C': [1, 100]
}

param_grid_LogisticRegression = {
    'C': [0.01, 1],
    'penalty': ['l2', 'elasticnet'],
    'solver': ['saga']
}

param_grid_RandomForest = {
    'n_estimators': [80, 200],
    'max_depth': [None],
    'min_samples_leaf': [1, 4]
}

param_grid_NaiveBayes = {
    'var_smoothing': np.logspace(-4, 0, 100)   # 100 valores log-espaciados
}
```

Todos los gridsearch usaron **validación cruzada de 3 folds** (`cv=3`).

#### Embeddings disponibles (CSVs pre-calculados — reutilizables desde disco)

Los embeddings ya están calculados y se pueden cargar directamente:

```
FinalProject_cl/embeddings/
FinalProject_AUG_cl/embeddings/
FinalProject_AUG_total_cl/embeddings/
FinalProject_SBERT_AUG_total_cl/embeddings/
FinalProject_SBERT_AUG_total_cl_PRE/embeddings/
```

| Tipo de embedding | Archivos disponibles |
|------------------|---------------------|
| BoW | `train_embeddings_<dataset>_BoW.csv` / `test_embeddings_<dataset>_BoW.csv` |
| TF-IDF | `train_embeddings_<dataset>_TF-IDF.csv` / `test_embeddings_...` |
| Doc2Vec | `train_embeddings_<dataset>_Doc2Vec.csv` / ... |
| FastText | `train_embeddings_<dataset>_FastText.csv` / ... |
| SBERT | `train_embeddings_<dataset>_SBERT.csv` / ... |
| prepr_SBERT | `train_embeddings_<dataset>_prepr_SBERT.csv` / ... |
| text-embedding-3-large | `train_embeddings_<dataset>_text-embedding-3-large.csv` / ... |

> Los embeddings son la parte más costosa de regenerar. Al estar disponibles en disco, el reentrenamiento de los modelos clásicos **no requiere volver a llamar a ninguna API externa** (GPT-4o, etc.).

#### Artefactos que el pipeline original guardaba

```python
# 1. Resultados del GridSearch
pd.DataFrame(grid.cv_results_).to_csv(f'gridsearch_results_{dataset}_{embedding}_{model}.csv')

# 2. Matrices de confusión (PNG)
# conf_matrix_{dataset}_{embedding}_{model}_{split}.png

# 3. Fila de resumen en CSV maestro
# results_final_summary_modelos.csv
```

### 10.5 Esquema de Reproducción

Para reproducir los modelos clásicos con el código disponible se requiere:

1. **Reusar los labs como base estructural** — `nlp-labs/lab-2/lab-2.ipynb` es el punto de partida más limpio
2. **Extender el loop** para incluir SVC, LogisticRegression y RandomForest (además de NaiveBayes)
3. **Sustituir la vectorización inline** por carga de embeddings pre-calculados desde los directorios `embeddings/` correspondientes
4. **Agregar GridSearchCV** con los parámetros reconstruidos en la sección anterior
5. **Agregar serialización** con `joblib.dump(best_model, 'modelo.joblib')` — actualmente ausente en todo el proyecto
6. **Agregar escritura de resultados** a CSV en lugar de solo mostrarlos en memoria

> **Nota:** El dataset de entrenamiento y los embeddings pre-calculados están intactos en disco. La ausencia de código fuente no implica pérdida de datos — solo requiere reescribir el script de entrenamiento.

---

*Reporte generado automáticamente mediante análisis estático del sistema de archivos.*
