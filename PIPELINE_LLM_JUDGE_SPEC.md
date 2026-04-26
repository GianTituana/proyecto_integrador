# Especificación de Pipeline: LLM-as-a-Judge con Few-Shot Prompting

## Contexto del proyecto

Este pipeline forma parte del proyecto **"Diseño e implementación de un índice de riesgo de corrupción basado en preguntas acusatorias del Sistema Oficial de Contratación Pública del Ecuador (SOCE)"**.

El sistema actual basado en ensemble (Logistic Regression + Naive Bayes + Random Forest) presenta un subregistro sistemático: detecta 14,10% de procesos con señal acusatoria frente al 21,24% que reporta la etiqueta real de la BD oficial. Este pipeline implementa un **LLM-as-a-Judge** como capa de validación complementaria que clasifica preguntas como acusatorias o no acusatorias, aprovechando la capacidad de razonamiento semántico de los LLMs.

**Objetivo:** Alcanzar F1-score ≥ 0,90 (o accuracy ≥ 95% considerando el desbalance de clases) en la clasificación binaria de preguntas como acusatorias/no acusatorias.

---

## Estructura del proyecto a crear

```
llm_judge_project/
├── data/
│   ├── raw/
│   │   └── dataset_roh_5005.csv
│   ├── splits/
│   │   ├── train_80.csv
│   │   ├── train_inner.csv
│   │   ├── val_inner.csv
│   │   └── test_20.csv
│   └── results/
│       ├── iteration_logs/
│       └── final_evaluation/
├── prompts/
│   ├── prompt_v1.txt
│   └── prompt_final.txt
├── notebooks/
│   └── llm_judge_pipeline.ipynb
├── src/
│   ├── data_utils.py
│   ├── prompt_utils.py
│   ├── llm_client.py
│   └── evaluation.py
├── .env
└── requirements.txt
```

---

## Fase 1: Setup del entorno

### 1.1 Dependencias (requirements.txt)

```
openai>=1.40.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
tenacity>=8.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### 1.2 Variables de entorno (.env)

```
OPENAI_API_KEY=sk-...
```

---

## Fase 2: Preparación de datos

### 2.1 Carga y partición estratificada

**Entrada:** Dataset original de Roh López (2024) con 5.005 preguntas etiquetadas (~3% acusatorias, ~97% no acusatorias).

**Columnas esperadas:**
- `pregunta` (texto): el contenido textual de la pregunta del SOCE
- `etiqueta` (entero): 0 = no acusatoria, 1 = acusatoria

**División:**
- 80% (4.004 preguntas) → conjunto de entrenamiento (para iterar prompt)
- 20% (1.001 preguntas) → conjunto de test (intocable hasta evaluación final)

**Sub-división del 80%:**
- 75% (3.003 preguntas) → `train_inner` (de aquí se muestrean los ejemplos few-shot)
- 25% (1.001 preguntas) → `val_inner` (para evaluar cada iteración)

**Restricciones:**
- Usar `train_test_split` con `stratify=etiqueta` para preservar proporción de clases
- Semilla fija `random_state=42` para reproducibilidad
- El `test_20.csv` no debe tocarse durante el ajuste del prompt

---

## Fase 3: Diseño del prompt base

### 3.1 Template del prompt v1

Crear `prompts/prompt_v1.txt`:

```
Eres un experto en auditoría y análisis de procesos de contratación pública del Ecuador. Tu tarea es clasificar preguntas formuladas en el Sistema Oficial de Contratación Pública (SOCE) durante procesos de licitación.

DEFINICIONES:

Una pregunta "ACUSATORIA" es aquella que contiene señalamientos explícitos o implícitos de posibles irregularidades, favoritismos, direccionamiento, sobreprecios, requisitos restrictivos injustificados, conflictos de interés, o cualquier indicio de corrupción en el proceso contractual.

Una pregunta "NO ACUSATORIA" es una consulta legítima sobre aspectos técnicos, administrativos, plazos, especificaciones o aclaraciones del pliego, sin denunciar irregularidades.

EJEMPLOS DE REFERENCIA:

[ACUSATORIA]
Pregunta: "{ejemplo_acusatoria_1}"

[ACUSATORIA]
Pregunta: "{ejemplo_acusatoria_2}"

[ACUSATORIA]
Pregunta: "{ejemplo_acusatoria_3}"

[ACUSATORIA]
Pregunta: "{ejemplo_acusatoria_4}"

[NO ACUSATORIA]
Pregunta: "{ejemplo_no_acusatoria_1}"

[NO ACUSATORIA]
Pregunta: "{ejemplo_no_acusatoria_2}"

[NO ACUSATORIA]
Pregunta: "{ejemplo_no_acusatoria_3}"

[NO ACUSATORIA]
Pregunta: "{ejemplo_no_acusatoria_4}"

INSTRUCCIONES DE SALIDA:
- Analiza la pregunta a clasificar considerando el contexto de la contratación pública ecuatoriana
- Responde ÚNICAMENTE con una palabra: "ACUSATORIA" o "NO_ACUSATORIA"
- No añadas explicaciones, justificaciones, ni texto adicional

PREGUNTA A CLASIFICAR:
"{pregunta}"

CLASIFICACIÓN:
```

---

## Fase 4: Implementación del muestreo few-shot

### 4.1 Función de muestreo aleatorio

**Especificación:**
- Recibe el DataFrame de entrenamiento, número de ejemplos por clase, y semilla
- Realiza muestreo aleatorio sin reemplazo de 4 preguntas acusatorias y 4 no acusatorias
- Retorna dos listas separadas con los textos de las preguntas

### 4.2 Función de construcción del prompt

**Especificación:**
- Recibe el template, los ejemplos muestreados y la pregunta a clasificar
- Reemplaza los placeholders `{ejemplo_acusatoria_N}` y `{ejemplo_no_acusatoria_N}` con los textos correspondientes
- Reemplaza `{pregunta}` con el texto a clasificar
- Retorna el prompt completo listo para enviar al LLM

---

## Fase 5: Cliente LLM con paralelización

### 5.1 Configuración

- Modelo recomendado: `gpt-4o-mini` (balance costo/rendimiento)
- Temperatura: `0` (para reproducibilidad)
- Max tokens: `10` (respuesta corta esperada)
- Concurrencia máxima: `10` requests paralelas (semáforo asyncio)
- Reintentos: `3` con backoff exponencial (tenacity)

### 5.2 Función de clasificación batch

**Especificación:**
- Recibe lista de prompts ya construidos
- Ejecuta llamadas asíncronas en paralelo con límite de concurrencia
- Muestra barra de progreso (tqdm)
- Retorna lista de respuestas en el mismo orden

### 5.3 Parser de respuestas

**Especificación:**
- Convierte respuesta del LLM a etiqueta binaria
- Retorna `1` si la respuesta contiene "ACUSATORIA" sin "NO_ACUSATORIA"
- Retorna `0` si la respuesta contiene "NO_ACUSATORIA" o "NO ACUSATORIA"
- Retorna `-1` si la respuesta no es parseable (para análisis de errores)

---

## Fase 6: Loop de iteración y evaluación

### 6.1 Función `run_iteration`

**Entradas:**
- Número de iteración
- Template del prompt
- DataFrame de train_inner (para muestreo de ejemplos)
- DataFrame de val_inner (conjunto de validación)
- Semilla para muestreo aleatorio

**Proceso:**
1. Muestrear 4 ejemplos acusatorios + 4 no acusatorios de `train_inner` con la semilla
2. Construir un prompt para cada pregunta de `val_inner`
3. Ejecutar el LLM en paralelo sobre todos los prompts
4. Parsear respuestas a predicciones binarias
5. Calcular métricas: accuracy, F1, recall, precision
6. Identificar respuestas no parseables
7. Guardar log de la iteración en `data/results/iteration_logs/iter_NN.csv`
8. Guardar predicciones individuales en `iter_NN_preds.csv`

**Salida:**
- Diccionario con métricas y metadatos de la iteración

### 6.2 Análisis de errores

**Especificación:**
- Lee predicciones de una iteración
- Filtra falsos negativos (etiqueta=1, predicción=0): casos acusatorios que el LLM no detectó
- Filtra falsos positivos (etiqueta=0, predicción=1): casos no acusatorios marcados como acusatorios
- Imprime los primeros 10 ejemplos de cada categoría
- Permite identificar patrones para refinar el prompt en la siguiente iteración

### 6.3 Criterios de parada

Definir antes de iniciar:
- **Máximo de iteraciones:** 15
- **Umbral de éxito:** F1 ≥ 0,90 sobre `val_inner`
- **Sin mejora consecutiva:** detener si 3 iteraciones seguidas no mejoran F1
- **Métrica principal:** F1-score (más informativa que accuracy con desbalance)
- **Métricas secundarias:** Recall (priorizar) y Precision

---

## Fase 7: Evaluación final

### 7.1 Protocolo de evaluación final

**Solo se ejecuta UNA VEZ**, después de congelar el prompt:

1. Cargar `test_20.csv` (no se ha tocado durante el ajuste)
2. Cargar el prompt final (`prompt_final.txt`)
3. Usar la combinación de ejemplos few-shot que mejor desempeño tuvo en validación
4. Ejecutar el LLM sobre las 1.001 preguntas del test
5. Calcular y reportar:
   - Accuracy
   - F1-score
   - Recall
   - Precision
   - Matriz de confusión
   - Classification report completo (sklearn)
6. Guardar en `data/results/final_evaluation/final_metrics.csv`

### 7.2 Comparación con métodos del documento

Reportar comparativamente:
- LLM-as-a-Judge vs. Ensemble (M1+M2+M3)
- LLM-as-a-Judge vs. cada modelo individual
- LLM-as-a-Judge vs. etiqueta de la BD oficial

---

## Fase 8: Integración con el pipeline existente

Una vez validado el LLM-as-a-Judge, posibles integraciones:

1. **Como cuarto voto del ensemble:** Pasar de votación 3/3 a 4/4
2. **Como re-etiquetado del dataset SIE 2024:** Aplicarlo a las 123.669 preguntas del corpus completo para obtener una segunda referencia más allá de la etiqueta heurística de la BD
3. **Como validador de alertas:** Verificar la calidad semántica de las 247 alertas rojas y 1.719 alertas amarillas antes de escalarlas a auditores

---

## Notebook recomendado: estructura de celdas

Crear `notebooks/llm_judge_pipeline.ipynb` con la siguiente estructura:

### Celda 1 — Imports y configuración
```python
import os
import asyncio
import random
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SEMAPHORE = asyncio.Semaphore(10)
RANDOM_STATE = 42
```

### Celda 2 — Carga y exploración del dataset
- Cargar `data/raw/dataset_roh_5005.csv`
- Mostrar `df.info()`, `df.head()`, distribución de clases
- Validar que existan las columnas `pregunta` y `etiqueta`

### Celda 3 — Partición estratificada
- Dividir en train_80 / test_20
- Sub-dividir train en train_inner / val_inner
- Guardar los CSVs en `data/splits/`
- Imprimir tamaños y proporciones de clase

### Celda 4 — Carga del prompt template
- Leer `prompts/prompt_v1.txt`
- Mostrar el contenido para verificación

### Celda 5 — Funciones utilitarias
- `sample_few_shot_examples()`
- `build_prompt()`
- `parse_response()`

### Celda 6 — Cliente LLM asíncrono
- `classify_question()` con retry
- `classify_batch()` con paralelización

### Celda 7 — Función `run_iteration()`
- Implementación completa según Fase 6.1

### Celda 8 — Iteración 1 (ejemplo de uso)
```python
metrics_iter1 = await run_iteration(
    iteration_num=1,
    prompt_template=prompt_template,
    train_df=train_inner,
    val_df=val_inner,
    seed=42
)
print(metrics_iter1)
```

### Celda 9 — Análisis de errores
- Cargar predicciones de la iteración
- Mostrar falsos positivos y falsos negativos
- Identificar patrones para refinar prompt

### Celda 10 — Tabla comparativa de iteraciones
- Cargar todos los logs de iteración
- Construir DataFrame con evolución de métricas
- Graficar curvas de F1, recall, precision a lo largo de iteraciones

### Celda 11 — Evaluación final
- Cargar test_20.csv
- Ejecutar LLM con prompt final + ejemplos congelados
- Reportar métricas, matriz de confusión, classification_report
- Guardar resultados

### Celda 12 — Comparación con ensemble del proyecto
- Cargar predicciones del ensemble sobre test_20 (si están disponibles)
- Tabla comparativa: LLM Judge vs. M1, M2, M3, Ensemble, BD oficial
- Discusión de hallazgos

---

## Recomendaciones operativas

1. **Caching de respuestas:** Implementar un diccionario `response_cache` que almacene `(prompt_hash, model) -> response` para evitar gastos repetidos durante depuración.

2. **Logging estructurado:** Cada iteración debe guardar:
   - Versión del prompt usado
   - Ejemplos few-shot exactos
   - Semilla
   - Métricas obtenidas
   - Tiempo de ejecución
   - Costo aproximado

3. **Manejo de costos:** GPT-4o mini tiene un costo aproximado de $0.150 por 1M tokens de entrada y $0.600 por 1M tokens de salida. Para 1.000 preguntas con prompts de ~800 tokens y respuestas de ~10 tokens, el costo por iteración es de ~$0.13 USD.

4. **Reproducibilidad:** Fijar semillas en numpy, random y train_test_split. Usar `temperature=0` en todas las llamadas al LLM.

5. **Documentación de iteraciones:** Llevar un cuaderno de bitácora con qué cambió en cada versión del prompt y qué métricas resultaron. Esto alimenta directamente la sección metodológica del documento final.

6. **No optimizar accuracy aisladamente:** Con desbalance de 97/3, accuracy puede ser engañosa. Priorizar F1 y Recall, ya que el sistema es de alerta temprana (falsos negativos = costos altos).

---

## Criterios de éxito del pipeline

El pipeline se considera exitoso si:

1. ✅ La implementación es reproducible (semillas fijas, prompts versionados)
2. ✅ Se documenta el proceso de refinamiento iterativo (al menos 5 iteraciones logueadas)
3. ✅ El prompt final alcanza F1 ≥ 0,90 sobre val_inner
4. ✅ La evaluación final sobre test_20 confirma desempeño consistente
5. ✅ Se compara cuantitativamente con el ensemble existente
6. ✅ Los resultados se integran al documento final del proyecto

---

## Instrucciones para Claude Code

Si vas a usar Claude Code para implementar este pipeline:

1. Lee este archivo completo primero
2. Crea la estructura de carpetas según la sección "Estructura del proyecto a crear"
3. Implementa el notebook `llm_judge_pipeline.ipynb` siguiendo la estructura de celdas propuesta
4. Crea los módulos auxiliares en `src/` para reutilización
5. Genera el `prompts/prompt_v1.txt` con el template completo
6. Crea un `requirements.txt` con las dependencias
7. Genera un `.env.example` para que el usuario configure su API key
8. Antes de ejecutar la primera iteración, verifica que el dataset original tenga las columnas esperadas y la distribución de clases correcta

Si el dataset tiene nombres de columnas distintos a `pregunta` y `etiqueta`, pregúntale al usuario cómo se llaman en su CSV antes de implementar.
