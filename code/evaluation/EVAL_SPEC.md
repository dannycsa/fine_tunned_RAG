# EVAL_SPEC — Contrato de evaluación (cloud vs local)

> **Para Grace:** este documento define TODO lo que tu lado (local/Ollama) debe
> hacer **idéntico** a mi lado (cloud/Bedrock). Si seguimos esto, las Tablas 2 y 3
> del paper salen comparables. Lo que cambia es **solo** quién genera el veredicto
> (tu Gemma local vs mi Nova en la nube); **todo lo demás es compartido**.

---

## 1. Los dos sistemas que comparamos

| | Sistema | Quién | Cómo infiere |
|---|---|---|---|
| **Cloud** | `amazon.nova-micro-v1:0` (zero-shot) | Adrian | AWS Bedrock Converse API |
| **Local** | `gemma-2-2b` fine-tuneado | Grace | Ollama (REST local) |

Ninguno usa RAG ni fine-tuning extra para esta evaluación: el cloud es **zero-shot**,
el local es **el modelo ya fine-tuneado**. El contexto viene dado en cada instancia.

## 2. Test set y submuestra

- Archivo único: **`grace_code/evaluacion_1200_es.jsonl`** (1.200 instancias ES, 50% alucina / 50% fiel).
- El **cloud** se evalúa sobre una **submuestra de 200** (por costo de API) — estratificada 50/50, `seed=42`.
- El **local** puede correr sobre las **1.200** (no tiene costo por token).
- ⚠️ Para la comparación directa de la Tabla 2, **ambos deben reportar también sobre las MISMAS 200** (Grace: corre score.py sobre el mismo subconjunto de 200). El muestreo está fijado en `infer_cloud.py::cargar_test` con `seed=42`; reutilízalo para elegir las mismas 200.

## 3. Prompt EXACTO (zero-shot) — `eval_common.py`

**System:**
```
Eres un auditor experto en sistemas RAG. Tu tarea es detectar si una respuesta
contiene alucinaciones, es decir, información que NO está respaldada por el
contexto. Básate ÚNICAMENTE en el contexto proporcionado, nunca en tu
conocimiento externo.
```

**User (template):**
```
Analiza si la RESPUESTA contiene alucinaciones respecto al CONTEXTO.

### Contexto y Pregunta:
{context_block}

### Respuesta a evaluar:
{response}

Responde OBLIGATORIAMENTE en este formato exacto, sin texto adicional antes:
VEREDICTO: SI    (si la respuesta contiene información NO respaldada por el contexto)
VEREDICTO: NO    (si la respuesta está totalmente respaldada por el contexto)
JUSTIFICACION: <una o dos frases explicando por qué>
```

> Grace: usa **literalmente** este system + user (impórtalos de `eval_common.py`,
> no los reescribas). Parámetros de inferencia: `temperature=0.0`, `top_p=1.0`,
> `max_tokens=512` (en Ollama: `options={"temperature":0,"top_p":1,"num_predict":512}`).

## 4. Etiquetas (cómo se derivan)

- **gold_label** (correcta): se parsea del veredicto-oro del dataset. Empieza con
  "Sí…" → `1` (alucinación); "No…" → `0` (fiel). Función: `eval_common.verdict_to_label`.
- **pred_label** (predicha): se lee la línea `VEREDICTO: SI/NO` de la salida del
  modelo. Función: `eval_common.parse_pred_label`. **Usa esta misma función**, así
  las dos partes parsean igual.
- Convención: **clase positiva = alucinación = 1**.

## 5. Esquema del archivo de predicciones (IDÉNTICO en ambos lados)

Cada línea de `preds_cloud.jsonl` / `preds_local.jsonl` es un JSON con:

| Campo | Tipo | Nota |
|---|---|---|
| `id` | int | índice de la instancia en el test set |
| `idioma` | str | `"es"` |
| `context_block` | str | contexto+pregunta → RAGAS `retrieved_contexts` |
| `response_audited` | str | la respuesta auditada |
| `task_prompt` | str | prompt completo dado al modelo → RAGAS `user_input` |
| `gold_verdict` | str | veredicto correcto |
| `gold_label` | int | 0/1 |
| `pred_verdict` | str | salida cruda del modelo → RAGAS `response` |
| `pred_label` | int | 0/1 |
| `latency_ms` | float | **wall-clock** de la llamada de inferencia |
| `server_latency_ms` | float\|null | latencia del servidor (cloud); **null en local** |
| `input_tokens` | int | tokens de entrada |
| `output_tokens` | int | tokens de salida |
| `system` | str | `"cloud:amazon.nova-micro-v1:0"` / `"local:gemma-2-2b-ft"` |

> **Grace — tokens y latencia en Ollama:** la respuesta de Ollama (`/api/generate`
> o `/api/chat`) trae `prompt_eval_count` (= `input_tokens`) y `eval_count`
> (= `output_tokens`). La latencia: mide wall-clock alrededor de la llamada
> (`time.time()` antes/después), igual que yo. `server_latency_ms = None`.

## 6. Métricas (definiciones exactas) — `score.py`

### 6.1 Detección — Precision / Recall / F1
Clase positiva = **alucinación (1)**. Sobre el conteo TP/FP/FN/TN:
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2·P·R / (P + R)
```
Reportamos:
- **Por clase**: alucinación (1) y fiel (0).
- **Macro** = promedio simple de las dos clases (P, R y F1). → el **Macro-F1 es el número principal de la Tabla 2** (el paper pide "macro-averaged P/R/F1").
- **Accuracy** = aciertos / total.

### 6.2 RAGAS — Faithfulness + Answer Relevancy
- **Faithfulness**: ¿las afirmaciones del veredicto del modelo están respaldadas
  por el contexto? (descompone la respuesta en afirmaciones y verifica cada una).
- **Answer Relevancy**: ¿el veredicto es relevante a lo que se le pidió?
- **Mapeo de campos** (decisión de diseño, ver §8):
  - `user_input`         = `task_prompt`
  - `response`           = `pred_verdict`  (el veredicto generado por el modelo)
  - `retrieved_contexts` = `[context_block]`
- **Juez (LLM)**: `amazon.nova-pro-v1:0` (vía `ChatBedrockConverse`).
- **Embeddings**: `amazon.titan-embed-text-v2:0`.
- **Región**: `us-east-1`.

> Grace: tu scoring **también** llama a Bedrock para el juez y los embeddings
> (necesitas credenciales AWS solo para esto; tu *inferencia* sigue siendo local).
> Esto es lo que garantiza que las columnas de RAGAS sean comparables.

### 6.3 Eficiencia
- **Latencia**: mediana (y media) de `latency_ms` (wall-clock por instancia).
- **Tokens**: suma de `input_tokens` y `output_tokens`. El **costo** (cloud) se
  calcula aparte con el precio de Bedrock × tokens; el local es $0.00.

## 7. Cómo se corre (los dos lados)

**Cloud (Adrian):**
```bash
cd code/evaluation
python infer_cloud.py --n 200            # -> preds_cloud.jsonl
python score.py --preds preds_cloud.jsonl
```

**Local (Grace):**
```bash
# 1) genera preds_local.jsonl con el MISMO esquema (§5) desde Ollama,
#    usando el MISMO prompt (§3) y la MISMA función parse_pred_label (§4)
python infer_local.py                    # tu script -> preds_local.jsonl
# 2) corre EL MISMO score.py (no lo modifiques)
python score.py --preds preds_local.jsonl
```

## 8. Checklist de "debe ser idéntico" ✅
- [ ] Mismo test set y mismas 200 instancias (seed=42) para la comparación.
- [ ] Mismo system + user prompt (importados de `eval_common.py`).
- [ ] Mismos parámetros: temperature=0, top_p=1, max_tokens=512.
- [ ] Misma función de parseo de etiqueta (`parse_pred_label`).
- [ ] Mismo esquema de `preds_*.jsonl`.
- [ ] Mismo `score.py`, mismo juez (`nova-pro`), mismos embeddings (`titan v2`).

## 9. Limitaciones a declarar en el paper (honestidad metodológica)
1. **RAGAS sobre una tarea de detección:** RAGAS está diseñado para evaluar
   *respuestas generadas* en QA. Aquí lo aplicamos al *veredicto* del modelo
   (mide qué tan fundamentada y relevante es su justificación). Por eso la
   **métrica principal de detección es el Macro-F1**, y RAGAS es secundaria.
2. **El juez comparte proveedor con el sistema cloud:** juez = Nova Pro, sistema
   cloud = Nova Micro (mismo proveedor, distinto tamaño). Frente al sistema local
   (Gemma) el juez es independiente. Si se habilita un modelo no-Nova (Claude
   3.5+/GPT-4) como juez, es cambiar una línea en `eval_common.py`. (Los Claude 3
   están bloqueados como *legacy* en la cuenta actual.)
3. **Solo 2 de las 4 métricas de RAGAS** se reportan (faithfulness + answer
   relevancy): Context Precision/Recall evalúan el *retriever*, y aquí no hay
   retrieval (el contexto viene dado), así que serían idénticas en ambos sistemas.
