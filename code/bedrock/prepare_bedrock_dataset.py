"""
prepare_bedrock_dataset.py
==========================
Convierte los JSONL de entrenamiento/validación (campo "text" con el formato
"### Tarea ... ### Veredicto del Auditor: ...") al formato que exige el
fine-tuning de Amazon Bedrock (Amazon Nova): el esquema conversacional
`bedrock-conversation-2024` (system + messages user/assistant).

Entrada : ../train/ragtruth_train.jsonl  y  ../train/ragtruth_val.jsonl
Salida  : ./bedrock_train.jsonl  y  ./bedrock_val.jsonl

Uso:
    python prepare_bedrock_dataset.py
"""

import os
import json

AQUI = os.path.dirname(os.path.abspath(__file__))
TRAIN_IN = os.path.join(AQUI, "..", "train", "ragtruth_train.jsonl")
VAL_IN = os.path.join(AQUI, "..", "train", "ragtruth_val.jsonl")
TRAIN_OUT = os.path.join(AQUI, "bedrock_train.jsonl")
VAL_OUT = os.path.join(AQUI, "bedrock_val.jsonl")

# Marcadores donde se separa el "prompt" (entrada) de la "respuesta" (salida que
# el modelo debe aprender a generar). Soportamos español e inglés.
MARCADORES = ["### Veredicto del Auditor:", "### Auditor Verdict:"]

SYSTEM_ES = ("Eres un auditor experto en detección de alucinaciones en sistemas "
             "RAG. Analiza la respuesta frente al contexto y emite un veredicto.")
SYSTEM_EN = ("You are an expert auditor for hallucination detection in RAG "
             "systems. Analyze the response against the context and give a verdict.")


def convertir(texto):
    """Divide el 'text' en (system, user, assistant) y arma el objeto Nova."""
    marcador = next((m for m in MARCADORES if m in texto), None)
    if marcador is None:
        return None  # línea malformada -> se omite

    before, _, after = texto.partition(marcador)
    user_text = before.strip()
    completion = after.strip()
    if not user_text or not completion:
        return None

    system = SYSTEM_EN if marcador == "### Auditor Verdict:" else SYSTEM_ES

    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": system}],
        "messages": [
            {"role": "user", "content": [{"text": user_text}]},
            {"role": "assistant", "content": [{"text": completion}]},
        ],
    }


def procesar(entrada, salida):
    n_ok, n_skip = 0, 0
    with open(entrada, "r", encoding="utf-8") as fin, \
         open(salida, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            texto = json.loads(line).get("text", "")
            obj = convertir(texto)
            if obj is None:
                n_skip += 1
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"  {os.path.basename(entrada)} -> {os.path.basename(salida)}: "
          f"{n_ok} ejemplos ({n_skip} omitidos)")
    return n_ok


def main():
    print("Convirtiendo al formato Amazon Bedrock (Nova, bedrock-conversation-2024)...")
    n_train = procesar(TRAIN_IN, TRAIN_OUT)
    n_val = procesar(VAL_IN, VAL_OUT)
    # Bedrock exige un mínimo de ejemplos (Nova: ~100). Avisamos si no llega.
    if n_train < 100:
        print(f"\n⚠️ ADVERTENCIA: solo {n_train} ejemplos de train. "
              f"Bedrock/Nova suele exigir >= 100. Tienes 800, debería estar OK.")
    print("\nListo. Sube estos dos .jsonl a S3 con finetune_bedrock.py.")


if __name__ == "__main__":
    main()
