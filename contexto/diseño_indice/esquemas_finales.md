# Esquemas Finales para la Construcción del Índice de Riesgo de Corrupción (Indicador 4)

## Propuesta metodológica — Proyecto Kapak

---

## Introducción

En el presente capítulo se desarrollan los dos esquemas metodológicos seleccionados para la construcción del Indicador 4 (detección de preguntas acusatorias en procesos de contratación pública GEN) con base en la evidencia empírica sistematizada por Roh López (2024). Estos esquemas no son alternativos entre sí, sino complementarios: el primero busca maximizar la robustez de la detección a través de un sistema de votación entre modelos con perfiles distintos; el segundo integra múltiples dimensiones de riesgo en un índice continuo interpretable.

Los modelos de clasificación utilizados en ambos esquemas provienen del *Appendix A: Top 15 – Metrics of Best Detection Models and Instances* de Roh López (2024), así como del análisis de las 1.658 configuraciones documentadas en el archivo `Models_Results.xlsx`. Todas las configuraciones evaluadas emplean como etapa de validación un conjunto de prueba fijo (20 % de los datos originales, estratificado), y sus métricas —AUC-ROC, Recall binario, F1 Score binario, Precision binaria— fueron obtenidas sobre ese conjunto.

---

## ESQUEMA I: Sistema de Alerta Temprana Multi-Modelo (Ensemble)

### I.1 Fundamento y motivación

El objetivo de este esquema es detectar la mayor cantidad posible de comentarios acusatorios reales sin generar una carga de revisión manual insostenible. En contextos de auditoría anticorrupción, los dos tipos de error tienen costos asimétricos:

- **Falso negativo (FN)**: un proceso con indicios de corrupción pasa inadvertido. El costo puede ser millonario en términos de recursos públicos.
- **Falso positivo (FP)**: un proceso legítimo es marcado como sospechoso. El costo es el tiempo de auditoría desperdiciado.

Ningún modelo individual logra minimizar ambos errores al mismo tiempo; la reducción de falsos negativos implica aceptar más falsos positivos y viceversa. Por esta razón se adopta un *ensemble* de tres modelos con perfiles deliberadamente distintos: uno orientado a minimizar los falsos negativos (alta sensibilidad), uno orientado a minimizar los falsos positivos (alta especificidad), y uno que balancea ambos objetivos. El mecanismo de votación entre los tres permite explotar las fortalezas de cada perfil y generar niveles de alerta diferenciados.

---

### I.2 Selección de modelos y justificación

Los tres modelos del ensemble se seleccionaron del catálogo experimental de Roh López (2024) priorizando los siguientes criterios: (1) AUC-ROC ≥ 0,90 como garantía mínima de capacidad discriminativa, (2) cada modelo debe ocupar un perfil distinto en el espacio Recall–Precision, y (3) todos deben usar la misma representación de texto para poder compartir el costo de generación de embeddings.

#### I.2.1 Modelo M1 — Equilibrado (mejor F1)

| Atributo | Valor |
|----------|-------|
| **Algoritmo** | Logistic Regression |
| **Representación** | text-embedding-3-large (3.072 dims.) |
| **Dataset de entrenamiento** | Balanced 4 (3.886 muestras por clase) |
| **Aumentación de datos** | Sinónimos (WordNet) |
| **Hiperparámetros óptimos** | C=100, penalty=l2, solver=saga |
| **AUC-ROC** | 0,940 |
| **Recall** | 0,759 |
| **F1 Score** | 0,677 |
| **Precision** | 0,611 |
| **TP / FN / FP / TN** | 22 / 7 / 14 / 958 |

**Justificación**: Este modelo obtiene el F1 Score más alto del experimento completo (0,677), lo que lo posiciona como el mejor balance entre Recall y Precision observado en las 1.658 configuraciones probadas. El dataset Balanced 4 iguala la proporción de clases positiva y negativa al nivel máximo disponible (3.886 muestras por clase), lo que permite al modelo aprender representaciones más ricas de los comentarios acusatorios. La aumentación por sinónimos (WordNet) contribuye a la generalización sin introducir el ruido semántico que genera la paráfrasis automática con GPT-2. El embedding text-embedding-3-large captura matices semánticos complejos que los modelos de representación convencional (TF-IDF, BoW) no logran. Este es el modelo central del ensemble; su predicción tiene el mayor peso en la decisión final.

---

#### I.2.2 Modelo M2 — Sensible (minimiza falsos negativos)

| Atributo | Valor |
|----------|-------|
| **Algoritmo** | Naive Bayes |
| **Representación** | text-embedding-3-large (3.072 dims.) |
| **Dataset de entrenamiento** | Consolidated Data Augmentation (3.886–21.766 muestras) |
| **Aumentación de datos** | Todas las técnicas consolidadas |
| **Hiperparámetros óptimos** | var_smoothing=0,351 |
| **AUC-ROC** | 0,940 |
| **Recall** | 1,000 |
| **F1 Score** | 0,227 |
| **Precision** | 0,128 |
| **TP / FN / FP / TN** | 29 / 0 / 198 / 774 |

**Justificación**: Este modelo alcanza un Recall de 1,000 sobre el conjunto de prueba: no produce ningún falso negativo. Esta propiedad lo hace invaluable como red de seguridad en el ensemble, ya que garantiza que ningún comentario verdaderamente acusatorio escape al sistema. El AUC-ROC de 0,940 confirma que la capacidad discriminativa a nivel de ranking es robusta, no es un resultado espurio. La combinación Naive Bayes + text-embedding-3-large aprovecha la representación semántica densa del embedding con la estimación de densidades propias de Naive Bayes, resultando en una clasificación muy permisiva que prioriza el recall. El costo de esta sensibilidad es una precision baja (0,128), lo que genera muchas alertas falsas cuando se usa en aislamiento; sin embargo, dentro del ensemble, este sesgo se compensa con los otros dos modelos mediante el sistema de votación. El dataset Consolidated Data Augmentation —el más grande disponible— maximiza la exposición del modelo a variantes de comentarios acusatorios, consolidando su capacidad de detección.

---

#### I.2.3 Modelo M3 — Conservador (minimiza falsos positivos)

| Atributo | Valor |
|----------|-------|
| **Algoritmo** | Random Forest Classifier |
| **Representación** | text-embedding-3-large (3.072 dims.) |
| **Dataset de entrenamiento** | Balanced 4 (3.886 muestras por clase) |
| **Aumentación de datos** | GPT-4o mini con prompt v1 |
| **Hiperparámetros óptimos** | max_depth=None, min_samples_leaf=1, n_estimators=200 |
| **AUC-ROC** | 0,940 |
| **Recall** | 0,552 |
| **F1 Score** | 0,640 |
| **Precision** | 0,762 |
| **TP / FN / FP / TN** | 16 / 13 / 5 / 967 |

**Justificación**: Este modelo registra la Precision más alta entre las configuraciones con F1 ≥ 0,45 y Recall ≥ 0,45 (0,762), lo que equivale a que el 76,2% de las alertas que emite corresponden a comentarios verdaderamente acusatorios. En el contexto del ensemble, su función es actuar como "árbitro conservador": cuando este modelo vota positivo, la probabilidad de que la alerta sea real es sustancialmente mayor. La aumentación con GPT-4o mini (prompt v1) genera paráfrasis de alta calidad semántica que mejoran la robustez del clasificador frente a formulaciones indirectas de acusaciones, mientras que el Random Forest con n_estimators=200 aprovecha la diversidad de árboles para reducir la varianza en la clasificación. El AUC de 0,940 es consistente con los demás modelos, reforzando la comparabilidad de las probabilidades generadas.

---

### I.3 Resumen comparativo de los tres modelos

| Métrica | M1 — Equilibrado (LR) | M2 — Sensible (NB) | M3 — Conservador (RFC) |
|---------|-----------------------|--------------------|------------------------|
| Algoritmo | Logistic Regression | Naive Bayes | Random Forest |
| Representación | Emb-3-large | Emb-3-large | Emb-3-large |
| Dataset | Balanced 4 | Consolidated DA | Balanced 4 |
| AUC-ROC | 0,940 | 0,940 | 0,940 |
| **Recall** | 0,759 | **1,000** | 0,552 |
| **Precision** | 0,611 | 0,128 | **0,762** |
| **F1 Score** | **0,677** | 0,227 | 0,640 |
| Falsos Negativos | 7 | **0** | 13 |
| Falsos Positivos | 14 | 198 | **5** |
| Rol en ensemble | Decisor central | Red de seguridad | Árbitro conservador |

*Nota: La representación text-embedding-3-large es compartida entre los tres modelos, lo que permite calcular el embedding una sola vez por comentario y reutilizarlo, reduciendo el costo de API en aproximadamente un 66 %.*

---

### I.4 Mecanismo de votación y niveles de alerta

Cada modelo emite un voto positivo si su probabilidad predicha supera su umbral de decisión. Los umbrales son asimétricos y reflejan el perfil de cada modelo:

| Modelo | Umbral de decisión | Justificación |
|--------|--------------------|---------------|
| M1 (LR, equilibrado) | 0,50 | Umbral estándar, maximiza F1 |
| M2 (NB, sensible) | 0,30 | Umbral bajo para garantizar Recall ≈ 1 |
| M3 (RFC, conservador) | 0,65 | Umbral alto para controlar falsos positivos |

El nivel de alerta se determina por el conteo de votos positivos (V = V_M1 + V_M2 + V_M3):

| Votos (V) | Nivel de alerta | Acción recomendada | Prioridad |
|-----------|-----------------|-------------------|-----------|
| 3/3 | **ROJA (Alta)** | Revisión inmediata por auditor senior | 1 |
| 2/3 | **AMARILLA (Media)** | Monitoreo activo y revisión en 48h | 2 |
| 1/3 | **VERDE (Baja)** | Registro para análisis estadístico mensual | 3 |
| 0/3 | Sin alerta | Sin acción requerida | — |

**Razonamiento del sistema de votación**: dado que M2 raramente produce falsos negativos, cualquier caso positivo real recibirá al menos un voto. Si adicionalmente M1 o M3 votan positivo (alerta media), la confianza aumenta significativamente. Si los tres votan positivo (alerta roja), se trata de un caso con señales consistentes en los tres perfiles de detección, maximizando la precisión de las alertas más urgentes.

---

### I.5 Score de confianza del ensemble

Adicionalmente al nivel de alerta, el sistema calcula un score de confianza continuo para priorizar casos dentro del mismo nivel:

```
Score_confianza(alerta ROJA)   = max(P_M1, P_M3)        — máximo entre los modelos confiables
Score_confianza(alerta AMARILLA) = (P_M1 + P_M3) / 2    — promedio de los modelos precisos
Score_confianza(alerta VERDE)  = min(P_M1, P_M3)        — conservador, usa el mínimo
```

Este score permite ordenar los comentarios al interior de cada nivel de alerta por urgencia, facilitando la priorización del trabajo de auditoría.

---

## ESQUEMA II: Índice Compuesto Multi-Nivel con Dimensiones de Intensidad, Frecuencia y Severidad

### II.1 Motivación y origen del esquema consolidado

Este esquema resulta de la integración del Esquema 3 (Índice Compuesto Multi-Nivel Jerárquico) y el Esquema B (Índice compuesto de intensidad, frecuencia y severidad). Ambos esquemas comparten el objetivo de reemplazar la lógica binaria del Indicador 4 actual por un índice continuo, pero se complementan de manera natural:

- El **Esquema 3** aporta la arquitectura de tres niveles (comentario → proceso → indicador compuesto) y el ajuste por factor de contexto, que diferencia preguntas de proveedores de aclaraciones de la entidad.
- El **Esquema B** aporta una fórmula de agregación a nivel de proceso metodológicamente más rica que una simple media o máximo, al descomponer el riesgo en tres dimensiones interpretables: intensidad media, frecuencia de señales fuertes y severidad semántica.

La consolidación consiste en reemplazar la agregación del Nivel 2 del Esquema 3 por la fórmula IRC4^(B) del Esquema B, aplicada sobre los *scores ajustados por contexto* del Nivel 1 —en lugar de sobre las probabilidades brutas del modelo.

---

### II.2 Modelo base y justificación

Para el Esquema II se utiliza como clasificador base el **Modelo M1** (Logistic Regression + text-embedding-3-large, Balanced 4, Sinónimos):

- **F1=0,677 | Recall=0,759 | Precision=0,611 | AUC=0,940**

Este modelo se selecciona porque la fórmula de agregación del Esquema B utiliza directamente las probabilidades continuas *p_j* de cada comentario (no solo la clasificación binaria), por lo que la calidad de la calibración probabilística es determinante. La Regresión Logística es el modelo que mejor combina: (1) probabilidades bien calibradas por definición de su función de pérdida, (2) máximo F1 Score en el experimento, y (3) bajo costo computacional en inferencia. El embedding text-embedding-3-large garantiza que el espacio de representación capture semántica profunda, condición necesaria para que las probabilidades del clasificador sean informativas más allá del umbral de decisión.

---

### II.3 Arquitectura del esquema consolidado

El índice se construye en tres niveles jerárquicos:

```
NIVEL 3: Indicador Compuesto (IC_proceso)
    ↑ Integra Ind_4 mejorado con todos los demás indicadores del sistema Kapak

NIVEL 2: IRC4 del proceso — fórmula compuesta B
    ↑ Agrega los scores ajustados de todos los comentarios del proceso

NIVEL 1: Score ajustado por comentario
    Score(c_i) = P_ML(c_i) × Factor_contexto(c_i)
```

---

### II.4 Nivel 1 — Score ajustado por comentario

Para cada comentario *c_i* de un proceso, se calcula:

```
Score(c_i) = P_ML(c_i) × Factor_contexto(c_i)
```

donde:

- **P_ML(c_i)**: probabilidad de acusatoriedad generada por M1 (Logistic Regression + Emb-3-large) para el comentario *i*. Esta probabilidad es continua en [0,1].

- **Factor_contexto(c_i)**: factor de ajuste basado en características observables del comentario, definido como:

```
Factor_contexto(c_i) = α₁ × F_etapa + α₂ × F_longitud + α₃ × F_técnico

Con: α₁ = 0,4 | α₂ = 0,3 | α₃ = 0,3 | Σαᵢ = 1
```

Los subfactores se definen de acuerdo al proceso GEN del SOCE (SERCOP, 2020):

**F_etapa** (tipo de interacción en el proceso GEN):

| Tipo de interacción | F_etapa | Justificación |
|---------------------|---------|---------------|
| Pregunta de proveedor | 1,3 | Mayor probabilidad de revelar irregularidades; el dataset de entrenamiento de Roh López (2024) está compuesto principalmente por preguntas de proveedores |
| Aclaración de entidad | 0,8 | Contenido típicamente técnico/administrativo, con menor carga acusatoria |

**F_longitud** (extensión del comentario):

| Longitud | F_longitud | Justificación |
|----------|------------|---------------|
| > 500 caracteres | 1,2 | Comentarios detallados suelen articular denuncias más completas |
| 100–500 caracteres | 1,0 | Longitud típica, sin ajuste |
| < 100 caracteres | 0,8 | Comentarios muy cortos con menor contenido informativo |

**F_técnico** (densidad de términos técnico-jurídicos):

```
F_técnico = min(1,0 + 0,1 × count(términos_técnicos), 1,5)

Términos considerados: especificaciones, requisitos, normativa, artículo,
ley, resolución, reglamento, procedimiento, licitación, pliego
```

El score ajustado se limita al intervalo [0,1] para preservar la interpretación como probabilidad:

```
Score(c_i) = min(P_ML(c_i) × Factor_contexto(c_i), 1,0)
```

---

### II.5 Nivel 2 — IRC4 del proceso (fórmula compuesta B)

Una vez calculados los scores ajustados Score(c_i) para todos los *n_p* comentarios del proceso *p*, se calcula el IRC4 del proceso mediante la fórmula de tres dimensiones del Esquema B, donde se reemplazan las probabilidades brutas *p_j* por los scores ajustados:

#### Dimensión 1 — Intensidad probabilística media (Ī_p)

```
Ī_p = (1 / n_p) × Σᵢ Score(c_i)
```

Captura el nivel promedio de acusatoriedad ajustado por contexto en el proceso. Un proceso donde todos los comentarios tienen scores moderados tendrá Ī_p moderada; un proceso con pocos comentarios muy acusatorios también puede tener Ī_p alta.

#### Dimensión 2 — Frecuencia de señales fuertes (F_p)

```
F_p = (1 / n_p) × Σᵢ 𝟙(Score(c_i) ≥ τ)

Umbral recomendado: τ = 0,60
```

Mide la proporción de comentarios con score ajustado superior al umbral. Este componente protege contra casos donde un solo comentario extremo eleva artificialmente Ī_p, al exigir que las señales fuertes sean frecuentes para que F_p sea alto.

#### Dimensión 3 — Severidad semántica (S_p)

```
S_p ∈ [0, 1]
```

Score construido sobre el vocabulario de mayor gravedad presente en el conjunto de comentarios del proceso. Se construye mediante reglas auditables basadas en categorías de irregularidad:

| Categoría | Peso | Términos representativos |
|-----------|------|--------------------------|
| Direccionamiento | Alto (0,9) | "dirigido a", "solo puede ganar", "favorece a", "diseñado para" |
| Restricción de competencia | Alto (0,8) | "limita participación", "vulnera libre competencia", "excluye" |
| Posible colusión | Medio-alto (0,7) | "acuerdo previo", "coordinado", "manipulación" |
| Sobreprecio | Medio (0,6) | "precio inflado", "costo excesivo", "sobrevaluado" |
| Irregularidad documental | Medio (0,5) | "documentos incompletos", "requisito injustificado", "alterado" |

S_p se calcula como el máximo ponderado detectado en el proceso, evitando que múltiples menciones del mismo tipo acumulen artificialmente el score.

#### Fórmula IRC4 del proceso

```
IRC4_p = 100 × (w₁ × Ī_p + w₂ × F_p + w₃ × S_p)

Valores iniciales recomendados para piloto:
    w₁ = 0,45   (intensidad media — componente probabilístico continuo)
    w₂ = 0,35   (frecuencia de señales fuertes — protege contra outliers)
    w₃ = 0,20   (severidad semántica — inicia con peso moderado hasta validar taxonomía)

Restricción: w₁ + w₂ + w₃ = 1
```

El IRC4_p oscila entre 0 y 100, donde valores altos indican mayor riesgo de corrupción. Para obtener el Indicador 4 en escala [0,1] alineado con los demás indicadores del sistema Kapak, se aplica la transformación complementaria:

```
Ind_4_proceso = 1 - (IRC4_p / 100)

Interpretación:
    Ind_4 ≈ 1,0  →  bajo riesgo, proceso sin señales acusatorias
    Ind_4 ≈ 0,0  →  alto riesgo, proceso con múltiples señales fuertes
```

---

### II.6 Nivel 3 — Integración en el Indicador Compuesto

El Ind_4_proceso calculado en el Nivel 2 se incorpora al Indicador Compuesto del sistema Kapak con ponderación diferenciada respecto a los demás indicadores, dado que el método ML ha demostrado una mejora de aproximadamente 40–60 % sobre el método de palabras clave utilizado actualmente (Roh López, 2024):

```
IC_proceso = [Σₖ (Ind_k × w_k)] / [Σₖ w_k] × 100

Ponderación propuesta:
    Ind_1  (plazos de entrega)               × 1,0
    Ind_2  (relación objeto-CPC)             × 1,0
    Ind_4  (preguntas acusatorias — NUEVO)   × 1,75   ← peso incrementado
    Ind_5  (reclamos)                        × 1,2
    Ind_6  (aclaraciones)                    × 1,2
    Ind_9  (número de invitados)             × 1,0
    Ind_11 (sin concurrencia)                × 1,1
    Ind_12 (relación bienes-CPC)             × 1,0
    Ind_18 (sobreprecio)                     × 1,3
    Ind_19 (proveedores multipropósito)      × 1,2
    Ind_22 (documentación)                  × 1,1
    Ind_25 (plazo adjudicación corto)        × 1,2
    Ind_27 (desierto sin motivación)         × 1,1
    Ind_28 (contratos complementarios)       × 1,2
    Ind_29 (contratos modificatorios)        × 1,2

Suma de pesos: 17,25
```

El peso de 1,75 para Ind_4 representa el punto medio del rango justificado por la evidencia (1,5–2,0) y refleja que el nuevo cálculo basado en ML produce una señal de mayor calidad y menor ruido que el método binario precedente.

---

### II.7 Propiedades metodológicas del esquema consolidado

La integración de los dos esquemas genera las siguientes ventajas que ninguno de los dos lograría por separado:

**Del Esquema 3 (jerárquico)**:
- El factor de contexto diferencia tipos de interacción en el proceso GEN (preguntas vs. aclaraciones), lo que introduce conocimiento de dominio que el modelo ML solo no captura.
- La arquitectura de tres niveles permite calcular el Ind_4 a nivel de proceso y luego incorporarlo al índice compuesto del sistema Kapak de manera natural.

**Del Esquema B (compuesto IFS)**:
- La descomposición en intensidad, frecuencia y severidad hace el índice interpretable para auditores y analistas jurídicos que necesitan explicar por qué un proceso es marcado como de alto riesgo.
- El componente F_p protege contra la manipulación del índice mediante un único comentario extremo (problema del máximo puro).
- El componente S_p incorpora gravedad jurídica diferenciada, lo que permite priorizar por tipo de irregularidad.

**De la consolidación**:
- Los scores ajustados por contexto (Nivel 1) alimentan las tres dimensiones de Nivel 2, transmitiendo el conocimiento de dominio GEN a toda la cadena de cálculo.
- El índice final es continuo, acumulativo e interpretable, superando la lógica binaria del Indicador 4 actual.

---

### II.8 Protocolo de calibración de parámetros

Los pesos w₁, w₂, w₃ y el umbral τ deben calibrarse mediante validación retrospectiva antes del despliegue en producción:

1. **Fase de baseline**: calcular IRC4 con pesos iniciales (w₁=0,45, w₂=0,35, w₃=0,20) sobre el conjunto histórico de procesos con etiqueta de irregularidad conocida.
2. **Análisis de sensibilidad**: variar τ en {0,50; 0,60; 0,70} y registrar el impacto en la distribución de alertas y la correlación con casos confirmados.
3. **Ajuste de pesos**: incrementar el peso del componente con mayor correlación con los casos confirmados de irregularidad.
4. **Validación temporal**: entrenar con datos 2020–2023 y validar en 2024, verificando que la mejora se mantenga fuera de la muestra de entrenamiento.
5. **Implementación gradual**: desplegar en paralelo con el sistema actual durante 3–6 meses para comparar resultados antes de adoptar el nuevo índice como oficial.

---

## Relación entre ambos esquemas

Los dos esquemas propuestos son complementarios y pueden operar de manera conjunta en el sistema Kapak:

| Aspecto | Esquema I (Ensemble) | Esquema II (Multi-nivel IFS) |
|---------|---------------------|------------------------------|
| **Función principal** | Alerta temprana granular por comentario | Índice de proceso interpretable |
| **Unidad de análisis** | Comentario individual | Proceso de contratación |
| **Salida** | Nivel de alerta (Roja/Amarilla/Verde) | Índice continuo IRC4 ∈ [0,100] |
| **Modelo(s) base** | M1 (LR) + M2 (NB) + M3 (RFC) | M1 (LR) únicamente |
| **Objetivo de error** | Minimizar FN (sistema completo) | Calibrar FP/FN mediante umbral τ |
| **Interpretabilidad** | Media (sistema de votación) | Alta (IFS descompuesto) |
| **Integración con Kapak** | Señal de alerta operativa | Indicador compuesto IC_proceso |

Una arquitectura recomendada consiste en que el **Esquema I** opere como capa de alerta temprana en tiempo real (generando notificaciones a auditores), mientras el **Esquema II** calcula el IRC4 a nivel de proceso para actualizar el índice compuesto mensual del sistema Kapak. De esta forma, el sistema combina la máxima sensibilidad de detección (Recall ≈ 1 del ensemble gracias a M2) con la interpretabilidad requerida por los analistas de control (IRC4 descompuesto en intensidad, frecuencia y severidad).

---

*Documento elaborado en el marco del Proyecto Kapak — Universidad San Francisco de Quito*
*Referencia principal: Roh López, F. (2024). Detección de comentarios acusatorios en procesos de contratación pública del Ecuador mediante aprendizaje automático. USFQ.*
