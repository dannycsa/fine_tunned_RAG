"""
infer_local.py — INFERENCIA LOCAL (parte de Grace) — PLANTILLA LISTA
====================================================================
Corre el Gemma fine-tuneado en Ollama sobre el MISMO test set, con el MISMO
prompt, y escribe preds_local.jsonl con el MISMO esquema que el lado cloud.
Así score.py produce métricas directamente comparables.

Requisitos:
    - Ollama corriendo con tu modelo fine-tuneado, p. ej.:
        ollama create gemma-ragtruth -f Modelfile   # tu modelo merged
    - pip install requests

Uso:
    python infer_local.py --model gemma-ragtruth --n 200     # mismas 200 que cloud
    python infer_local.py --model gemma-ragtruth --n 1200    # corpus completo
"""
import os
import json
import time
import random
import argparse

import requests
import eval_common as C

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--testset", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "grace_code", "evaluacion_1200_es.jsonl"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "preds_local.jsonl"))
    p.add_argument("--model", required=True, help="nombre del modelo en Ollama")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cargar_test(path, n, seed):
    """IDÉNTICO al de infer_cloud.py: mismo seed=42 -> MISMAS instancias."""
    filas = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            ctx, resp, gold_v, gold_l = C.parse_test_text(text)
            filas.append({"id": i, "context_block": ctx, "response_audited": resp,
                          "gold_verdict": gold_v, "gold_label": gold_l})
    random.seed(seed)
    pos = [r for r in filas if r["gold_label"] == 1]
    neg = [r for r in filas if r["gold_label"] == 0]
    random.shuffle(pos); random.shuffle(neg)
    k = n // 2
    sel = pos[:k] + neg[:n - k]
    random.shuffle(sel)
    return sel


def invocar(model, ctx, resp):
    user = C.build_user_prompt(ctx, resp)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": C.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": C.INFER_TEMPERATURE, "top_p": C.INFER_TOP_P,
                    "num_predict": C.INFER_MAX_TOKENS},
    }
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    wall_ms = (time.time() - t0) * 1000.0
    r.raise_for_status()
    d = r.json()
    out = d["message"]["content"]
    # Ollama devuelve los conteos de tokens:
    tin = d.get("prompt_eval_count")
    tout = d.get("eval_count")
    return user, out, wall_ms, tin, tout


def main():
    a = parse_args()
    filas = cargar_test(a.testset, a.n, a.seed)
    print(f"Test cargado: {len(filas)} | modelo Ollama: {a.model}")

    n_ok = 0
    with open(a.out, "w", encoding="utf-8") as fout:
        for idx, r in enumerate(filas):
            try:
                user, out, wall, tin, tout = invocar(a.model, r["context_block"], r["response_audited"])
            except Exception as e:
                print(f"  [{idx}] error: {e} — salto")
                continue
            rec = {
                "id": r["id"], "idioma": "es",
                "context_block": r["context_block"],
                "response_audited": r["response_audited"],
                "task_prompt": user,
                "gold_verdict": r["gold_verdict"], "gold_label": r["gold_label"],
                "pred_verdict": out, "pred_label": C.parse_pred_label(out),
                "latency_ms": round(wall, 1), "server_latency_ms": None,
                "input_tokens": tin, "output_tokens": tout,
                "system": f"local:{a.model}",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += 1
            if (idx + 1) % 20 == 0:
                print(f"  {idx+1}/{len(filas)}...")

    print(f"\nListo: {n_ok} -> {a.out}")
    print("Ahora puntúa con costo local 0:")
    print("  python score.py --preds preds_local.jsonl --price-in 0 --price-out 0")


if __name__ == "__main__":
    main()
