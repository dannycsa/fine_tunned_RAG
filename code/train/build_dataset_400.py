"""
build_dataset_400.py
=====================
Genera de forma REPRODUCIBLE el split 400 ES + 400 EN (train) y un set de
validación balanceado, a partir de `ragtruth_qa_filtrado_es.jsonl`.

Por qué existe este script (arregla 2 problemas):

  FIX #1 (datos "stale"): los ragtruth_train.jsonl/ragtruth_val.jsonl que estaban
         versionados venían de una corrida vieja (4730 filas), NO del 400/400 que
         pide la tarea. Este script regenera los archivos correctos.

  FIX #2 (procedencia / no determinismo): el notebook asignaba el idioma con
         random.random() SIN semilla, por lo que cada ejecución daba un dataset
         distinto e irreproducible, y además leía un archivo fuente (5913 filas)
         que no está en el repo. Aquí fijamos todas las semillas y leemos el
         archivo que SÍ está en el repo, con asserts que avisan si algo no cuaja.

Uso:
    python build_dataset_400.py
"""

import os
import json
import ast
import random

SEED = 42
N_TRAIN_PER_LANG = 400   # 400 ES + 400 EN  (requisito de la tarjeta)
N_VAL_PER_LANG = 200     # validación balanceada y acotada (ver nota más abajo)

# --- Localizar el archivo fuente (el que SÍ está en el repo) ---
AQUI = os.path.dirname(os.path.abspath(__file__))
CANDIDATOS = [
    os.path.join(AQUI, "ragtruth_qa_filtrado_es.jsonl"),
    os.path.join(AQUI, "..", "..", "dataset", "ragtruth_qa_filtrado_es.jsonl"),
]
FUENTE = next((p for p in CANDIDATOS if os.path.exists(p)), None)
if FUENTE is None:
    raise FileNotFoundError(
        "No encontré 'ragtruth_qa_filtrado_es.jsonl'. Busqué en:\n  - "
        + "\n  - ".join(CANDIDATOS)
    )


def parsear_etiquetas(x):
    """labels_originales puede venir como lista JSON o como string; normalizamos."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(x)
                return v if isinstance(v, list) else []
            except Exception:
                continue
    return []


def construir_texto(row):
    """Replica EXACTAMENTE el formato del notebook (### Tarea / Veredicto)."""
    etiquetas = parsear_etiquetas(row.get("labels_originales", []))
    tiene_aluc = len(etiquetas) > 0

    # idioma asignado con moneda al aire, pero AHORA con semilla -> reproducible
    idioma = "es" if random.random() < 0.5 else "en"

    if idioma == "es":
        prompt = row.get("prompt_es", "")
        respuesta = row.get("response_es", "")
        if tiene_aluc:
            tipo = etiquetas[0].get("label_type", "Alucinación general")
            frase = etiquetas[0].get("text", "Texto no especificado")
            veredicto = (
                f"Sí, se detectó una alucinación del tipo '{tipo}'. "
                f"El modelo generó la siguiente información sin respaldo en el "
                f"contexto: \"{frase}\"."
            )
        else:
            veredicto = ("No, la respuesta es correcta, segura y está totalmente "
                         "respaldada por los pasajes del contexto.")
        texto = (
            "### Tarea: Analiza si la siguiente respuesta contiene alucinaciones "
            "basándote en el contexto.\n\n"
            f"### Contexto y Pregunta:\n{prompt}\n\n"
            f"### Respuesta a evaluar:\n{respuesta}\n\n"
            f"### Veredicto del Auditor:\n{veredicto}"
        )
    else:
        prompt = row.get("prompt_en", "")
        respuesta = row.get("response_en", "")
        if tiene_aluc:
            tipo = etiquetas[0].get("label_type", "General Hallucination")
            frase = etiquetas[0].get("text", "Unspecified text")
            veredicto = (
                f"Yes, a hallucination of type '{tipo}' was detected. "
                f"The model generated the following unsupported information: "
                f"\"{frase}\"."
            )
        else:
            veredicto = ("No, the response is correct, safe, and fully supported "
                         "by the context passages.")
        texto = (
            "### Task: Analyze if the following response contains hallucinations "
            "based on the context.\n\n"
            f"### Context and Question:\n{prompt}\n\n"
            f"### Response to evaluate:\n{respuesta}\n\n"
            f"### Auditor Verdict:\n{veredicto}"
        )

    return idioma, texto


def main():
    random.seed(SEED)  # <-- determinismo total (idioma + muestreo)

    es_rows, en_rows = [], []
    with open(FUENTE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idioma, texto = construir_texto(row)
            (es_rows if idioma == "es" else en_rows).append(texto)

    print(f"Fuente: {os.path.relpath(FUENTE, AQUI)}")
    print(f"Disponibles -> ES: {len(es_rows)} | EN: {len(en_rows)}")

    # Asserts: si no hay suficientes por idioma, avisar claro (en vez de fallar raro)
    minimo = N_TRAIN_PER_LANG + N_VAL_PER_LANG
    assert len(es_rows) >= minimo, f"ES insuficiente: {len(es_rows)} < {minimo}"
    assert len(en_rows) >= minimo, f"EN insuficiente: {len(en_rows)} < {minimo}"

    random.shuffle(es_rows)
    random.shuffle(en_rows)

    train = es_rows[:N_TRAIN_PER_LANG] + en_rows[:N_TRAIN_PER_LANG]
    val = (es_rows[N_TRAIN_PER_LANG:N_TRAIN_PER_LANG + N_VAL_PER_LANG]
           + en_rows[N_TRAIN_PER_LANG:N_TRAIN_PER_LANG + N_VAL_PER_LANG])

    random.shuffle(train)
    random.shuffle(val)

    out_train = os.path.join(AQUI, "ragtruth_train.jsonl")
    out_val = os.path.join(AQUI, "ragtruth_val.jsonl")
    with open(out_train, "w", encoding="utf-8") as f:
        for t in train:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with open(out_val, "w", encoding="utf-8") as f:
        for t in val:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print(f"\nEscrito:")
    print(f"  {os.path.basename(out_train)} -> {len(train)} filas "
          f"(ES={N_TRAIN_PER_LANG}, EN={N_TRAIN_PER_LANG})")
    print(f"  {os.path.basename(out_val)}   -> {len(val)} filas "
          f"(ES={N_VAL_PER_LANG}, EN={N_VAL_PER_LANG})")
    print("\nNota: la validación se acotó a un set balanceado (200/200) en lugar de "
          "'todo el resto' (~miles). Así la evaluación cada 50 pasos es rápida y la "
          "métrica eval_loss es comparable entre idiomas.")


if __name__ == "__main__":
    main()
