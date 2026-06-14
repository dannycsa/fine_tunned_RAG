"""
eval_common.py — CONTRATO COMPARTIDO de evaluación (cloud Adrian / local Grace)
===============================================================================
TODO lo que DEBE ser idéntico entre la inferencia en la nube y la local vive aquí:
  - el PROMPT exacto de detección (zero-shot)
  - cómo se parsea el test set
  - cómo se deriva la etiqueta binaria (gold y predicha)
  - los IDs de los modelos juez/embeddings de RAGAS
  - el ESQUEMA del archivo de predicciones

Si cambias algo aquí, cámbialo en AMBOS lados o los números no son comparables.
"""

# ===========================================================================
# 1. MODELOS (fijos para los dos lados)
# ===========================================================================
# Sistema en la NUBE (la línea base del paper). Adrian usa este.
CLOUD_SYSTEM_MODEL = "amazon.nova-micro-v1:0"
# Sistema LOCAL (Grace) — solo informativo aquí; ella lo corre en Ollama.
LOCAL_SYSTEM_MODEL = "gemma-2-2b-ft (Ollama)"

# JUEZ y EMBEDDINGS de RAGAS: IDÉNTICOS para ambos lados (los dos llaman a Bedrock
# SOLO para puntuar; la inferencia de Grace es local, pero su scoring usa esto).
# NOTA: en esta cuenta los Claude están bloqueados (legacy), así que el juez es
# Nova Pro (el Nova más capaz). Limitación documentada en EVAL_SPEC.md: el juez
# comparte proveedor con el sistema cloud (Nova-micro), aunque difiere en tamaño;
# frente al sistema local (Gemma) el juez es totalmente independiente.
RAGAS_JUDGE_MODEL = "amazon.nova-pro-v1:0"   # juez (distinto tamaño del sistema)
RAGAS_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
AWS_REGION = "us-east-1"

# Parámetros de inferencia del sistema (deterministas para reproducibilidad)
INFER_TEMPERATURE = 0.0
INFER_TOP_P = 1.0
INFER_MAX_TOKENS = 512

# ===========================================================================
# 2. PROMPT DE DETECCIÓN (zero-shot) — EXACTAMENTE el mismo en cloud y local
# ===========================================================================
SYSTEM_PROMPT = (
    "Eres un auditor experto en sistemas RAG. Tu tarea es detectar si una "
    "respuesta contiene alucinaciones, es decir, información que NO está "
    "respaldada por el contexto. Básate ÚNICAMENTE en el contexto proporcionado, "
    "nunca en tu conocimiento externo."
)

USER_PROMPT_TEMPLATE = """Analiza si la RESPUESTA contiene alucinaciones respecto al CONTEXTO.

### Contexto y Pregunta:
{context_block}

### Respuesta a evaluar:
{response}

Responde OBLIGATORIAMENTE en este formato exacto, sin texto adicional antes:
VEREDICTO: SI    (si la respuesta contiene información NO respaldada por el contexto)
VEREDICTO: NO    (si la respuesta está totalmente respaldada por el contexto)
JUSTIFICACION: <una o dos frases explicando por qué>"""


def build_user_prompt(context_block, response):
    return USER_PROMPT_TEMPLATE.format(context_block=context_block, response=response)


# ===========================================================================
# 3. PARSEO DEL TEST SET (evaluacion_1200_es.jsonl -> campos separados)
# ===========================================================================
H_CTX = "### Contexto y Pregunta:"
H_RESP = "### Respuesta a evaluar:"
H_VER = "### Veredicto del Auditor:"


def parse_test_text(text):
    """Devuelve (context_block, response_audited, gold_verdict, gold_label).
    gold_label: 1 = alucinación, 0 = fiel."""
    ctx = text.split(H_CTX, 1)[1]
    ctx, rest = ctx.split(H_RESP, 1)
    resp, gold = rest.split(H_VER, 1)
    context_block = ctx.strip()
    response_audited = resp.strip()
    gold_verdict = gold.strip()
    gold_label = verdict_to_label(gold_verdict)
    return context_block, response_audited, gold_verdict, gold_label


def verdict_to_label(verdict_text):
    """Etiqueta GOLD: el template empieza con 'Sí, se detectó...' (alucina=1)
    o 'No, la respuesta es correcta...' (fiel=0). EN: 'Yes,'/'No,'."""
    t = verdict_text.strip().lower()
    if t.startswith(("sí", "si", "yes")):
        return 1
    return 0


def parse_pred_label(model_output):
    """Etiqueta PREDICHA: leemos la línea 'VEREDICTO: SI/NO'. Con respaldos por si
    el modelo no respeta el formato."""
    t = model_output.strip().lower()
    # 1) formato pedido
    if "veredicto:" in t:
        seg = t.split("veredicto:", 1)[1].lstrip()
        if seg.startswith(("si", "sí")):
            return 1
        if seg.startswith("no"):
            return 0
    # 2) respaldo por palabras clave
    if "veredicto: si" in t or "sí, se detect" in t or "alucinaci" in t and "no" not in t[:20]:
        return 1
    if t.startswith(("no", "veredicto: no")) or "es correcta" in t or "respaldad" in t:
        return 0
    # 3) por defecto, fiel (clase mayoritaria conservadora) — se cuenta como no-detección
    return 0


# ===========================================================================
# 4. ESQUEMA del archivo de predicciones (.jsonl) — MISMO en cloud y local
# ===========================================================================
# Cada línea del archivo de predicciones DEBE tener estas claves:
PREDICTION_SCHEMA = [
    "id",               # int: índice de la instancia en el test set
    "idioma",           # "es"
    "context_block",    # str: contexto+pregunta (RAGAS retrieved_contexts)
    "response_audited", # str: la respuesta que se está auditando
    "task_prompt",      # str: prompt completo dado al modelo (RAGAS user_input)
    "gold_verdict",     # str: veredicto correcto
    "gold_label",       # int 0/1
    "pred_verdict",     # str: salida cruda del modelo (RAGAS response)
    "pred_label",       # int 0/1
    "latency_ms",       # float: wall-clock de la llamada de inferencia
    "server_latency_ms",# float|null: latencia reportada por Bedrock (cloud); null en local
    "input_tokens",     # int
    "output_tokens",    # int
    "system",           # str: "cloud:amazon.nova-micro-v1:0" o "local:gemma-2-2b-ft"
]
