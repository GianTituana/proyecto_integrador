# Resumen Comparativo de Esquemas Metodológicos para el Índice de Riesgo de Corrupción

## Basado en los documentos: top_esquemas_claude.md y top_esquemas_codex.md

---

## 1. Contexto General

### Objetivo Común

Ambos documentos proponen metodologías para **reemplazar el Indicador 4 binario** (detección de preguntas acusatorias por palabras clave) por un **índice continuo de riesgo de corrupción** utilizando los modelos de machine learning desarrollados en la tesis de Francisco Roh López (USFQ, 2024).

### Modelo Base Recomendado (Consenso)

| Característica             | Valor                           |
| --------------------------- | ------------------------------- |
| **Modelo**            | Logistic Regression             |
| **Representación**   | text-embedding-3-large (OpenAI) |
| **Data Augmentation** | Synonyms (WordNet)              |
| **F1 Score**          | 0.677                           |
| **Recall**            | 0.759                           |
| **Precision**         | 0.611                           |
| **AUC-ROC**           | 0.94                            |

Este modelo se selecciona por su **balance óptimo** entre detectar casos positivos (recall) y minimizar falsos positivos (precision), con alta calibrabilidad de probabilidades.

---

## 2. Esquemas Propuestos por Documento

### Nomenclatura

| Claude (top_esquemas_claude.md) | Codex (top_esquemas_codex.md) |
| ------------------------------- | ----------------------------- |
| Esquema 1: Granular             | Esquema A: Probabilístico    |
| Esquema 3: Multi-Nivel          | Esquema B: Compuesto I+F+S    |
| Esquema 4: Ensemble             | *(sin equivalente)*         |
| *(sin equivalente)*           | Esquema C: Bayesiano          |

---

## 3. Resumen de Cada Esquema

### 3.1 ESQUEMA GRANULAR / PROBABILÍSTICO

#### Claude - Esquema 1: Índice por Pregunta Individual

**Concepto:** Cada comentario recibe un score de probabilidad P ∈ [0,1] que indica qué tan acusatorio es.

**Fórmula base:**

```
P_acusatorio(c) = σ(w^T · E(c) + b)

Donde:
- E(c) = embedding del comentario (3,072 dimensiones)
- w, b = parámetros del modelo entrenado
- σ = función sigmoide
```

**Estrategias de agregación a proceso:**

| Estrategia         | Fórmula                                 | Uso recomendado          |
| ------------------ | ---------------------------------------- | ------------------------ |
| Máximo            | max(P₁, P₂, ..., Pₙ)                  | Detectar picos de riesgo |
| Media              | (ΣPᵢ) / n                              | Nivel general de riesgo  |
| Percentil 90       | P90(P₁, ..., Pₙ)                       | Robusto a outliers       |
| **Híbrida** | α·max + β·mean + γ·P90 + δ·ratio | **Recomendada**    |

**Configuración híbrida:** α=0.4, β=0.3, γ=0.2, δ=0.1

---

#### Codex - Esquema A: Índice Probabilístico de Ocurrencia

**Concepto:** Calcula la probabilidad de que **al menos una** pregunta sea acusatoria en el proceso.

**Fórmula:**

```
IRC4_A = 1 - ∏(1 - pⱼ)
         j=1

Donde pⱼ = P(Yⱼ = 1 | Xⱼ) para cada pregunta j
```

**Interpretación:**

- Si todas las pⱼ son bajas → índice bajo
- Si una sola pⱼ es alta → índice sube significativamente
- Si varias pⱼ son moderadas → riesgo acumulado aumenta

**Limitaciones:** Asume independencia entre preguntas; puede sobreacumular riesgo si hay preguntas similares.

---

### 3.2 ESQUEMA COMPUESTO

#### Claude - Esquema 3: Índice Multi-Nivel Jerárquico

**Concepto:** Arquitectura de 3 niveles que integra información desde el comentario individual hasta el indicador compuesto.

```
┌─────────────────────────────────────────────────────────────┐
│  NIVEL 3: INDICADOR COMPUESTO                               │
│  IC = Σ(Ind_k × w_k) / Σ(w_k) × 100                        │
├─────────────────────────────────────────────────────────────┤
│  NIVEL 2: INDICADOR POR PROCESO                             │
│  Ind_4 = f(agregación de scores)                           │
├─────────────────────────────────────────────────────────────┤
│  NIVEL 1: SCORING POR COMENTARIO                            │
│  Score(c) = P_ML(c) × Factor_contexto(c)                   │
└─────────────────────────────────────────────────────────────┘
```

**Factor de Contexto:**

```
Factor_contexto = α₁×F_etapa + α₂×F_longitud + α₃×F_tecnico
```

| Factor               | Valores                                            | Justificación                                |
| -------------------- | -------------------------------------------------- | --------------------------------------------- |
| **F_etapa**    | Pregunta proveedor: 1.3 / Aclaración entidad: 0.8 | Alineado con proceso GEN del SOCE             |
| **F_longitud** | >500 chars: 1.2 / 100-500: 1.0 / <100: 0.8         | Comentarios detallados más informativos      |
| **F_tecnico**  | 1.0 + 0.1×(términos técnicos) [máx 1.5]        | Vocabulario técnico indica señal más clara |

**Pesos diferenciados por indicador:**

| Indicador                            | Peso           | Justificación                  |
| ------------------------------------ | -------------- | ------------------------------- |
| Ind_4 (preguntas acusatorias)        | **1.75** | Indicador central del análisis |
| Ind_18 (plazo entrega ofertas)       | 1.3            | Indicador de alto riesgo        |
| Ind_5, Ind_6, Ind_19, Ind_28, Ind_29 | 1.2            | Indicadores complementarios     |
| Otros                                | 1.0            | Peso base                       |

---

#### Codex - Esquema B: Índice de Intensidad, Frecuencia y Severidad

**Concepto:** Separa el riesgo en tres dimensiones interpretables para analistas.

**Componentes:**

| Componente               | Fórmula                  | Interpretación               |
| ------------------------ | ------------------------- | ----------------------------- |
| **Intensidad (I)** | I = (ΣPⱼ) / n           | Promedio de "acusatoriedad"   |
| **Frecuencia (F)** | F = (Σ𝟙[Pⱼ ≥ τ]) / n | Proporción con señal fuerte |
| **Severidad (S)**  | S ∈ [0,1] por reglas     | Gravedad del contenido        |

**Índice agregado:**

```
IRC4_B = 100 × (w₁·I + w₂·F + w₃·S)

Parametrización inicial:
- τ = 0.60 (umbral de señal fuerte)
- w₁ = 0.45 (intensidad)
- w₂ = 0.35 (frecuencia)
- w₃ = 0.20 (severidad)
```

**Vocabulario de severidad sugerido:**

- Direccionamiento, colusión, restricción de competencia, manipulación documental

---

### 3.3 ESQUEMA BAYESIANO (Solo Codex - Esquema C)

**Problema que resuelve:** Procesos con muy pocas preguntas generan índices inestables.

**Formulación:**

```
Conteos suaves:
  a = Σpⱼ (suma de probabilidades)
  b = n - Σpⱼ

Posterior:
  θ ~ Beta(α + a, β + b)

Scores posibles:
  IRC4_C1 = E[θ] = (α + a) / (α + β + n)     ← Esperanza posterior
  IRC4_C2 = P(θ > θ₀)                         ← Prob. de exceder umbral
```

**Hiperparámetros sugeridos:**

- Prior: Beta(3, 97) basado en prevalencia histórica ~3% de preguntas acusatorias
- Fuerza alta cuando n < 10 preguntas

**Ventajas:**

- Penaliza sobreconfianza en muestras pequeñas
- Proporciona intervalos de incertidumbre
- Ideal como ajuste sobre Esquema A o B

---

### 3.4 ESQUEMA ENSEMBLE (Solo Claude - Esquema 4)

**Concepto:** Combina 3 modelos con características complementarias para maximizar detección.

**Modelos del ensemble:**

| Rol                | Modelo                               | Fortaleza          | Umbral |
| ------------------ | ------------------------------------ | ------------------ | ------ |
| **Rápido**  | Random Forest + TF-IDF               | Velocidad, sin API | 0.50   |
| **Preciso**  | LogReg + text-embedding-3-large      | Mejor F1           | 0.50   |
| **Sensible** | Naive Bayes + text-embedding-3-large | Recall 96.6%       | 0.30   |

**Sistema de votación:**

```
┌─────────────────────────────────────────────────────────┐
│  VOTOS    │  NIVEL      │  ACCIÓN                       │
├───────────┼─────────────┼───────────────────────────────┤
│  3/3      │  🔴 ALTA    │  Investigar inmediatamente    │
│  2/3      │  🟡 MEDIA   │  Monitorear activamente       │
│  1/3      │  🟢 BAJA    │  Registrar para análisis      │
│  0/3      │  ⚪ NINGUNA │  Sin acción requerida         │
└─────────────────────────────────────────────────────────┘
```

**Métricas esperadas del ensemble:**

- Recall: ~98% (minimiza falsos negativos)
- Precision: ~55% (acepta más falsos positivos a cambio de cobertura)

---

## 4. Tabla Comparativa Detallada

### 4.1 Comparación por Características

| Criterio                  | Claude Esq.1  | Claude Esq.3     | Claude Esq.4 | Codex Esq.A     | Codex Esq.B | Codex Esq.C  |
| ------------------------- | ------------- | ---------------- | ------------ | --------------- | ----------- | ------------ |
| **Tipo**            | Granular      | Jerárquico      | Ensemble     | Probabilístico | Compuesto   | Bayesiano    |
| **Complejidad**     | Media         | Alta             | Alta         | Baja            | Media       | Media        |
| **Output**          | Score [0,1]   | IC [0,100]       | Nivel alerta | IRC [0,1]       | IRC [0,100] | θ posterior |
| **Agregación**     | 4 estrategias | Híbrida + pesos | Votación    | Producto        | Ponderada   | Posterior    |
| **Código Python**  | ✅ Completo   | ✅ Completo      | ✅ Completo  | ❌              | ❌          | ❌           |
| **Fórmulas LaTeX** | ✅            | ✅               | ✅           | ✅              | ✅          | ✅           |
| **Validación GEN** | ❌            | ✅               | ❌           | ❌              | ❌          | ❌           |
| **Incertidumbre**   | ❌            | ❌               | ❌           | ❌              | ❌          | ✅           |

### 4.2 Comparación por Métricas Esperadas

| Criterio                  | Claude Esq.1      | Claude Esq.3  | Claude Esq.4   | Codex Esq.A | Codex Esq.B | Codex Esq.C |
| ------------------------- | ----------------- | ------------- | -------------- | ----------- | ----------- | ----------- |
| **Recall**          | ~76%              | ~78%          | **~98%** | ~76%        | ~76%        | ~76%*       |
| **Precision**       | ~61%              | ~60%          | ~55%           | ~61%        | ~61%        | ~61%*       |
| **Costo mensual**   | ~$10-15 | ~$15-20 | ~$5-10 | ~$10 | ~$15 | ~$10    |             |             |             |
| **Implementación** | 2-3 meses         | 4-6 meses     | 3-4 meses      | 1-2 meses   | 2-3 meses   | 1 mes       |

*Esquema C usa el mismo modelo base, ajusta solo la agregación.

### 4.3 Comparación por Casos de Uso

| Escenario                                    | Mejor esquema       | Justificación                     |
| -------------------------------------------- | ------------------- | ---------------------------------- |
| **Despliegue rápido**                 | Codex A             | Implementación más simple        |
| **Máxima detección**                 | Claude 4 (Ensemble) | Recall del 98%                     |
| **Explicabilidad para auditores**      | Codex B             | Componentes I, F, S interpretables |
| **Procesos con pocas preguntas**       | Codex C             | Regularización bayesiana          |
| **Integración con Kapak existente**   | Claude 3            | Arquitectura multi-indicador       |
| **Análisis detallado por comentario** | Claude 1            | Máxima granularidad               |

---

## 5. Correspondencia Entre Esquemas

### 5.1 Esquemas Equivalentes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAPEO DE EQUIVALENCIAS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CLAUDE ESQUEMA 1              ←──── 70% similar ────→   CODEX ESQU. A │
│   (Granular)                                              (Probabilístico)
│   • Score por comentario                                  • pⱼ por pregunta
│   • Agregación flexible                                   • Producto de (1-pⱼ)
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CLAUDE ESQUEMA 3              ←──── 50% similar ────→   CODEX ESQU. B │
│   (Multi-Nivel)                                           (I + F + S)   │
│   • 3 niveles jerárquicos                                 • 3 componentes
│   • Factor de contexto                                    • Intensidad
│   • Pesos por indicador                                   • Frecuencia
│                                                           • Severidad   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CLAUDE ESQUEMA 4              ←──── SIN EQUIVALENTE                   │
│   (Ensemble)                                                            │
│   • 3 modelos complementarios                                           │
│   • Sistema de votación                                                 │
│   • Niveles de alerta                                                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                                 SIN EQUIVALENTE ────→   CODEX ESQU. C   │
│                                                         (Bayesiano)     │
│                                                         • Prior Beta    │
│                                                         • Incertidumbre │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Diferencias Clave Entre Equivalentes

#### Granular (Claude 1) vs Probabilístico (Codex A)

| Aspecto                   | Claude 1                                          | Codex A                                                      |
| ------------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| **Filosofía**      | Score individual es el producto final             | Score individual es insumo para agregación                  |
| **Agregación**     | Múltiples estrategias (max, mean, P90, híbrida) | Única fórmula: 1 - ∏(1-p)                                 |
| **Interpretación** | "Este comentario tiene X% de ser acusatorio"      | "Este proceso tiene X% prob. de tener al menos 1 acusatorio" |
| **Flexibilidad**    | Alta (configurable)                               | Baja (fórmula fija)                                         |

#### Multi-Nivel (Claude 3) vs Compuesto (Codex B)

| Aspecto                       | Claude 3                         | Codex B                       |
| ----------------------------- | -------------------------------- | ----------------------------- |
| **Estructura**          | Jerárquica (micro→meso→macro) | Plana (3 componentes sumados) |
| **Factores contexto**   | F_etapa, F_longitud, F_tecnico   | Solo el modelo base           |
| **Componentes**         | 15 indicadores ponderados        | 3 dimensiones: I, F, S        |
| **Severidad**           | No explícita                    | Componente S dedicado         |
| **Validación proceso** | ✅ Alineado con GEN del SOCE     | ❌ No validado                |

---

## 6. Fortalezas y Debilidades

### 6.1 Documento Claude (top_esquemas_claude.md)

| Fortalezas                                  | Debilidades                                       |
| ------------------------------------------- | ------------------------------------------------- |
| ✅ Código Python completo y ejecutable     | ❌ Documento extenso (1,755 líneas)              |
| ✅ Diagramas arquitectónicos detallados    | ❌ No aborda incertidumbre en muestras pequeñas  |
| ✅ Validación con proceso GEN real         | ❌ Puede ser abrumador para lectores no técnicos |
| ✅ Sistema ensemble para máxima detección | ❌ Esquema 4 requiere 3 modelos                   |
| ✅ Análisis de costos operativos           |                                                   |
| ✅ Configuraciones recomendadas             |                                                   |

### 6.2 Documento Codex (top_esquemas_codex.md)

| Fortalezas                                  | Debilidades                             |
| ------------------------------------------- | --------------------------------------- |
| ✅ Rigor matemático formal                 | ❌ Sin código de implementación       |
| ✅ Enfoque bayesiano para incertidumbre     | ❌ No valida contra estructura real GEN |
| ✅ Conciso y fácil de seguir (310 líneas) | ❌ Sin sistema ensemble                 |
| ✅ Criterios de validación explícitos     | ❌ Menos detalle en configuraciones     |
| ✅ Componente de severidad semántica       | ❌ Fórmula A asume independencia       |

---

## 7. Recomendación de Integración

### Estrategia Óptima: Combinación de Ambos Documentos

La metodología más robusta combina los mejores elementos de cada documento:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA INTEGRADA RECOMENDADA                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CAPA 1 - DETECCIÓN (Claude 4)                                          │
│  ├── Ensemble de 3 modelos                                              │
│  ├── Sistema de votación                                                │
│  └── Niveles de alerta: 🔴 🟡 🟢 ⚪                                      │
│                                                                         │
│  CAPA 2 - SCORING GRANULAR (Claude 1)                                   │
│  ├── P(acusatorio) por comentario                                       │
│  ├── Factor de contexto (F_etapa, F_longitud, F_tecnico)               │
│  └── Alineación con proceso GEN                                         │
│                                                                         │
│  CAPA 3 - AGREGACIÓN (Codex B + Claude 3)                              │
│  ├── Componente I: Intensidad probabilística                            │
│  ├── Componente F: Frecuencia de señales fuertes                        │
│  ├── Componente S: Severidad semántica                                  │
│  └── Ponderación diferenciada por indicador                             │
│                                                                         │
│  CAPA 4 - AJUSTE DE CONFIANZA (Codex C)                                │
│  ├── Prior bayesiano para procesos con n < 10                           │
│  ├── Intervalos de incertidumbre                                        │
│  └── Regularización de scores extremos                                  │
│                                                                         │
│  CAPA 5 - INTEGRACIÓN KAPAK (Claude 3)                                  │
│  ├── Indicador Compuesto con 15 indicadores                             │
│  ├── Ind_4 mejorado con peso 1.75                                       │
│  └── Dashboard y reportes automatizados                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Plan de Implementación por Fases

| Fase                             | Duración | Componentes                            | Fuente         |
| -------------------------------- | --------- | -------------------------------------- | -------------- |
| **1. Piloto**              | 1-2 meses | Esquema A (probabilístico simple)     | Codex          |
| **2. Producción básica** | 2-3 meses | Esquema 1 (granular) + Factor contexto | Claude         |
| **3. Compuesto**           | 3-4 meses | Esquema B (I+F+S) + Pesos Kapak        | Codex + Claude |
| **4. Robustez**            | 4-5 meses | Esquema C (bayesiano) para ajuste      | Codex          |
| **5. Máxima detección**  | 5-6 meses | Esquema 4 (ensemble) completo          | Claude         |

---

## 8. Validación Recomendada (Consenso)

Ambos documentos coinciden en los criterios de validación:

### Nivel Modelo

- [ ] ROC-AUC ≥ 0.90
- [ ] PR-AUC documentado
- [ ] Brier Score < 0.15 (calibración)
- [ ] Curva de calibración visual

### Nivel Agregación

- [ ] Monotonicidad: agregar evidencia NO reduce score
- [ ] Estabilidad por tamaño de proceso
- [ ] Revisión de casos frontera por expertos

### Nivel Operativo

- [ ] Carga de alertas mensual < 500
- [ ] Tasa de revisión efectiva documentada
- [ ] Proporción de alertas útiles ≥ 60%

---

## 9. Conclusión

| Pregunta                    | Respuesta                                                   |
| --------------------------- | ----------------------------------------------------------- |
| ¿Cuál documento es mejor? | **Complementarios**, no excluyentes                   |
| ¿Por dónde empezar?       | **Codex A** por simplicidad, luego **Claude 1** |
| ¿Para máxima detección?  | **Claude 4** (Ensemble)                               |
| ¿Para interpretabilidad?   | **Codex B** (I+F+S)                                   |
| ¿Para procesos pequeños?  | **Codex C** (Bayesiano)                               |
| ¿Para integrar con Kapak?  | **Claude 3** (Multi-Nivel)                            |

---

*Documento generado el 27 de febrero de 2026*
*Resumen comparativo de esquemas metodológicos para el Índice de Riesgo de Corrupción*
*Proyecto Kapak - SERCOP Ecuador*
