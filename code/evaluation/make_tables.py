"""
make_tables.py — arma las Tablas 2 y 3 del paper comparando ambos sistemas
==========================================================================
Lee los *_results.json que produce score.py (uno por sistema) e imprime las filas
de la Tabla 2 (detección + eficiencia) y la Tabla 3 (RAGAS), en texto y en LaTeX.

Uso:
    python make_tables.py preds_cloud_results.json preds_local_results.json
"""
import sys
import json


def load(paths):
    out = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        out.append(d)
    return out


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    if len(sys.argv) < 2:
        print("uso: python make_tables.py results1.json [results2.json ...]")
        return
    results = load(sys.argv[1:])

    print("\n================= TABLA 2: Detección + Eficiencia =================")
    print(f"{'Sistema':<32}{'Macro-F1':>9}{'Prec':>7}{'Rec':>7}{'Cost/1k':>10}{'Lat(ms)':>9}")
    for r in results:
        d, e = r["detection"], r["efficiency"]
        print(f"{r['system']:<32}{fmt(d['macro_f1']):>9}{fmt(d['macro_precision']):>7}"
              f"{fmt(d['macro_recall']):>7}{'$'+fmt(e['cost_per_1k_evals_usd'],4):>9}"
              f"{fmt(e['latency_ms_median'],0):>9}")

    print("\n================= TABLA 3: RAGAS =================")
    print(f"{'Sistema':<32}{'Faithfulness':>13}{'Answer Rel.':>13}")
    for r in results:
        rg = r.get("ragas", {}) or {}
        faith = rg.get("faithfulness")
        ar = rg.get("answer_relevancy") or rg.get("response_relevancy")
        print(f"{r['system']:<32}{fmt(faith):>13}{fmt(ar):>13}")

    # --- LaTeX listo para pegar ---
    print("\n--- LaTeX Tabla 2 ---")
    for r in results:
        d, e = r["detection"], r["efficiency"]
        print(f"{r['system']} & {fmt(d['macro_f1'])} & {fmt(d['macro_precision'])} & "
              f"{fmt(d['macro_recall'])} & \\${fmt(e['cost_per_1k_evals_usd'],4)} & "
              f"{fmt(e['latency_ms_median'],0)} \\\\")
    print("\n--- LaTeX Tabla 3 ---")
    for r in results:
        rg = r.get("ragas", {}) or {}
        ar = rg.get("answer_relevancy") or rg.get("response_relevancy")
        print(f"{r['system']} & {fmt(rg.get('faithfulness'))} & {fmt(ar)} \\\\")


if __name__ == "__main__":
    main()
