"""
infer_cloud.py — INFERENCIA EN LA NUBE (parte de Adrian)
========================================================
Corre el modelo de AWS Bedrock (zero-shot, sin fine-tuning ni RAG) sobre el test
set de 1.200, captura el veredicto, la latencia y los tokens, y escribe un archivo
de predicciones con el ESQUEMA compartido (eval_common.PREDICTION_SCHEMA).

Usa la Converse API de Bedrock: una sola interfaz para Nova/Claude/Llama y, sobre
todo, devuelve el conteo de tokens (input/output) y la latencia del servidor.

Uso:
    python infer_cloud.py --n 200                 # subsample de 200 (como el paper)
    python infer_cloud.py --n 1200                # corpus completo
    python infer_cloud.py --model amazon.nova-lite-v1:0 --n 50
"""
import os
import csv
import json
import time
import random
import argparse

import boto3
import eval_common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--testset", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "grace_code", "evaluacion_1200_es.jsonl"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "preds_cloud.jsonl"))
    p.add_argument("--model", default=C.CLOUD_SYSTEM_MODEL)
    p.add_argument("--n", type=int, default=200, help="cuántas instancias (subsample)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cargar_test(path, n, seed):
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
    # subsample estratificado por etiqueta (mantiene 50/50) y reproducible
    random.seed(seed)
    pos = [r for r in filas if r["gold_label"] == 1]
    neg = [r for r in filas if r["gold_label"] == 0]
    random.shuffle(pos); random.shuffle(neg)
    k = n // 2
    sel = pos[:k] + neg[:n - k]
    random.shuffle(sel)
    return sel


def invocar(client, model_id, ctx, resp):
    user = C.build_user_prompt(ctx, resp)
    t0 = time.time()
    r = client.converse(
        modelId=model_id,
        system=[{"text": C.SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": C.INFER_MAX_TOKENS,
                         "temperature": C.INFER_TEMPERATURE,
                         "topP": C.INFER_TOP_P},
    )
    wall_ms = (time.time() - t0) * 1000.0
    out = r["output"]["message"]["content"][0]["text"]
    usage = r.get("usage", {})
    server_ms = r.get("metrics", {}).get("latencyMs")
    return user, out, wall_ms, server_ms, usage.get("inputTokens"), usage.get("outputTokens")


def main():
    a = parse_args()
    client = boto3.client("bedrock-runtime", region_name=C.AWS_REGION)
    filas = cargar_test(a.testset, a.n, a.seed)
    print(f"Test cargado: {len(filas)} instancias | modelo: {a.model}")

    n_ok = 0
    with open(a.out, "w", encoding="utf-8") as fout:
        for idx, r in enumerate(filas):
            try:
                user, out, wall, srv, tin, tout = invocar(
                    client, a.model, r["context_block"], r["response_audited"])
            except Exception as e:
                print(f"  [{idx}] error: {e} — reintento en 3s")
                time.sleep(3)
                try:
                    user, out, wall, srv, tin, tout = invocar(
                        client, a.model, r["context_block"], r["response_audited"])
                except Exception as e2:
                    print(f"  [{idx}] error definitivo: {e2} — salto")
                    continue

            rec = {
                "id": r["id"], "idioma": "es",
                "context_block": r["context_block"],
                "response_audited": r["response_audited"],
                "task_prompt": user,
                "gold_verdict": r["gold_verdict"], "gold_label": r["gold_label"],
                "pred_verdict": out, "pred_label": C.parse_pred_label(out),
                "latency_ms": round(wall, 1), "server_latency_ms": srv,
                "input_tokens": tin, "output_tokens": tout,
                "system": f"cloud:{a.model}",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += 1
            if (idx + 1) % 20 == 0:
                print(f"  {idx+1}/{len(filas)} procesadas...")

    print(f"\nListo: {n_ok} predicciones -> {a.out}")
    print("Siguiente paso: python score.py --preds preds_cloud.jsonl")


if __name__ == "__main__":
    main()
