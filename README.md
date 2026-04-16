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
entregable_2/
├── datasets/                          # Conjuntos de datos (Git LFS)
│   ├── train_dataset.csv              # Dataset de entrenamiento
│   └── test_dataset.csv               # Dataset de evaluación (n=1,001)
│
├── pipelines/                         # Notebooks de experimentación (Git LFS)
│   ├── kapak_indicador4_pipeline.ipynb    # Pipeline principal de clasificación
│   ├── kapak_indicador4_replica.ipynb     # Réplica del protocolo Roh (2024)
│   └── baseline_bd/
│       ├── kapak_baseline_bd.ipynb        # Comparación baseline BD vs. ensemble ML
│       └── output/                        # Métricas y gráficos generados
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
            matplotlib seaborn python-dotenv
```

### 3. Git LFS

Los datasets y notebooks se almacenan en Git LFS. Para descargarlos:

```bash
git lfs install
git lfs pull
```

---

## Pipelines

### `kapak_indicador4_replica.ipynb`
Réplica del protocolo experimental de Roh López (2024): preprocesamiento, balanceo, data augmentation (sinónimos, BERT, GPT-4o mini) y entrenamiento de los tres modelos del ensemble.

### `kapak_indicador4_pipeline.ipynb`
Pipeline principal de inferencia: carga el test set, genera embeddings, aplica los tres modelos y construye el índice de riesgo agregado.

### `baseline_bd/kapak_baseline_bd.ipynb`
Compara el método actual de la BD (full-text search sobre 8 palabras clave, vista `mv_gen_proceso_indicador_04`) contra los tres modelos ML usando el test set original. Soporta ejecución con o sin conexión a la BD (fallback local con regex).

---

## Baseline de la BD (referencia)

```sql
to_tsvector('spanish', pregunta) @@ to_tsquery('spanish',
    'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo')
```

Clasifica como acusatorio si contiene al menos una de las 8 palabras clave.

---

## Referencias

- Roh López, F. (2024). *Detección de comentarios acusatorios en contratación pública mediante PLN*. USFQ.
- Tituana Intriago, G. M. (2026). *Diseño e implementación de un índice de riesgo de corrupción basado en preguntas acusatorias del SOCE para procesos GEN*. USFQ.
