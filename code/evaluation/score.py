"""
score.py — SCORING COMPARTIDO (lo corren IGUAL Adrian y Grace)
==============================================================
Lee un archivo de predicciones (mismo esquema en cloud y local) y calcula:
  - Detección:  Precision / Recall / F1 (por clase y macro) + Accuracy
  - RAGAS:      Faithfulness + Answer Relevancy  (juez=Claude, embeddings=Titan)
  - Eficiencia: latencia (mediana/media) + tokens totales (input/output)

Como AMBOS lados corren ESTE MISMO script con el MISMO juez/embeddings (fijados en
eval_common.py), las métricas son directamente comparables.

Uso:
    python score.py --preds preds_cloud.jsonl
    python score.py --preds preds_cloud.jsonl --max-ragas 50   # limita coste de RAGAS
    python score.py --preds preds_local.jsonl --no-ragas       # solo F1+latencia+tokens
"""
import os
import json
import argparse
import statistics

import eval_common as C


# ----------------------- Detección: P / R / F1 -----------------------------
def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def detection_metrics(rows):
    gold = [r["gold_label"] for r in rows]
    pred = [r["pred_label"] for r in rows]
    n = len(rows)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0

    # Clase positiva = alucinación (1)
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
    p1, r1, f1_pos = prf(tp, fp, fn)
    # Clase negativa = fiel (0)
    p0, r0, f1_neg = prf(tn, fn, fp)
    macro_f1 = (f1_pos + f1_neg) / 2
    macro_p = (p1 + p0) / 2
    macro_r = (r1 + r0) / 2
    return {
        "n": n, "accuracy": acc,
        "halluc_precision": p1, "halluc_recall": r1, "halluc_f1": f1_pos,
        "faithful_precision": p0, "faithful_recall": r0, "faithful_f1": f1_neg,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ----------------------- Eficiencia: latencia / tokens ---------------------
# Precios on-demand Bedrock por 1.000 tokens (us-east-1, feb 2026). El sistema
# LOCAL no tiene costo de API -> usar --price-in 0 --price-out 0 (costo = $0.00).
PRICE_IN_NOVA_MICRO = 0.000035   # USD / 1k tokens entrada
PRICE_OUT_NOVA_MICRO = 0.00014   # USD / 1k tokens salida


def efficiency_metrics(rows, price_in, price_out):
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    tin = sum(r.get("input_tokens") or 0 for r in rows)
    tout = sum(r.get("output_tokens") or 0 for r in rows)
    n = len(rows)
    cost_total = (tin / 1000) * price_in + (tout / 1000) * price_out
    return {
        "latency_ms_median": statistics.median(lat) if lat else None,
        "latency_ms_mean": statistics.mean(lat) if lat else None,
        "total_input_tokens": tin,
        "total_output_tokens": tout,
        "total_tokens": tin + tout,
        "avg_tokens_per_instance": (tin + tout) / n if n else 0,
        "cost_total_usd": round(cost_total, 6),
        "cost_per_1k_evals_usd": round((cost_total / n) * 1000, 4) if n else 0,
    }


# ----------------------- RAGAS: faithfulness + answer relevancy ------------
def ragas_metrics(rows, max_n):
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

    # ChatBedrockConverse usa la Converse API -> compatible con Amazon Nova.
    judge = LangchainLLMWrapper(ChatBedrockConverse(
        model=C.RAGAS_JUDGE_MODEL, region_name=C.AWS_REGION,
        temperature=0.0, max_tokens=1024))
    emb = LangchainEmbeddingsWrapper(BedrockEmbeddings(
        model_id=C.RAGAS_EMBED_MODEL, region_name=C.AWS_REGION))

    sub = rows[:max_n] if max_n else rows
    # MAPEO (documentado en EVAL_SPEC.md):
    #   user_input         = prompt completo dado al modelo
    #   response           = veredicto generado por el modelo (su justificación)
    #   retrieved_contexts = [bloque de contexto+pregunta]
    samples = [{
        "user_input": r["task_prompt"],
        "response": r["pred_verdict"],
        "retrieved_contexts": [r["context_block"]],
    } for r in sub]

    ds = EvaluationDataset.from_list(samples)
    result = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy()],
                      llm=judge, embeddings=emb)
    df = result.to_pandas()
    out = {"ragas_n": len(sub)}
    for col in df.columns:
        if col in ("faithfulness", "answer_relevancy", "response_relevancy"):
            vals = [v for v in df[col].tolist() if v == v]  # descarta NaN
            out[col] = sum(vals) / len(vals) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--max-ragas", type=int, default=0, help="0 = todas")
    ap.add_argument("--price-in", type=float, default=PRICE_IN_NOVA_MICRO,
                    help="USD/1k tokens entrada (0 para local)")
    ap.add_argument("--price-out", type=float, default=PRICE_OUT_NOVA_MICRO,
                    help="USD/1k tokens salida (0 para local)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.preds, encoding="utf-8") if l.strip()]
    print(f"Predicciones cargadas: {len(rows)} | sistema: {rows[0].get('system')}")

    results = {"system": rows[0].get("system"),
               "detection": detection_metrics(rows),
               "efficiency": efficiency_metrics(rows, args.price_in, args.price_out)}

    print("\n===== DETECCIÓN (P/R/F1) =====")
    d = results["detection"]
    print(f"  Accuracy           : {d['accuracy']:.3f}")
    print(f"  Macro  P/R/F1      : {d['macro_precision']:.3f} / {d['macro_recall']:.3f} / {d['macro_f1']:.3f}")
    print(f"  Alucinación P/R/F1 : {d['halluc_precision']:.3f} / {d['halluc_recall']:.3f} / {d['halluc_f1']:.3f}")
    print(f"  Matriz             : {d['confusion']}")

    print("\n===== EFICIENCIA =====")
    e = results["efficiency"]
    print(f"  Latencia mediana   : {e['latency_ms_median']} ms")
    print(f"  Tokens totales     : {e['total_tokens']} (in={e['total_input_tokens']}, out={e['total_output_tokens']})")
    print(f"  Costo total        : ${e['cost_total_usd']}  |  Costo/1k evals: ${e['cost_per_1k_evals_usd']}")

    if not args.no_ragas:
        print("\n===== RAGAS (faithfulness + answer relevancy) — usando Claude juez... =====")
        results["ragas"] = ragas_metrics(rows, args.max_ragas)
        for k, v in results["ragas"].items():
            print(f"  {k}: {round(v,3) if isinstance(v,float) else v}")

    out = args.out or args.preds.replace(".jsonl", "_results.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nResultados -> {out}")


if __name__ == "__main__":
    main()
