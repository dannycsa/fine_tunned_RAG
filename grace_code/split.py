import pandas as pd
import json
import ast
from sklearn.model_selection import train_test_split

print("=== INICIANDO PIPELINE DE DATASET BLINDADO (CROSS-LINGUAL) ===")

# ==========================================
# 1. CARGAR ESPAÑOL (NUESTRA BASE)
# ==========================================
print("\n[1] Cargando el dataset en Español (ragtruth_qa_filtrado_es_ult.jsonl)...")
data_es = []
with open("ragtruth_qa_filtrado_es_ult.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data_es.append(json.loads(line))

df_es = pd.DataFrame(data_es)

def parsear_etiquetas(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except: return []
    return x if isinstance(x, list) else []

df_es['parsed_labels'] = df_es['labels_originales'].apply(parsear_etiquetas)
df_es['tiene_alucinacion'] = df_es['parsed_labels'].apply(lambda x: len(x) > 0)
df_es['idioma_asignado'] = 'es'

# ==========================================
# 2. SEPARAR EL TEST (1200) Y EL TRAIN_ES (462)
# ==========================================
df_es_alucina = df_es[df_es['tiene_alucinacion'] == True]
df_es_limpio = df_es[df_es['tiene_alucinacion'] == False]

print("\n[2] Extrayendo los Splits en Español...")
test_es_alucina = df_es_alucina.sample(n=600, random_state=42)
test_es_limpio = df_es_limpio.sample(n=600, random_state=42)
df_test_final = pd.concat([test_es_alucina, test_es_limpio]).sample(frac=1, random_state=42).reset_index(drop=True)

df_es_alucina_ft = df_es_alucina.drop(test_es_alucina.index).sample(n=231, random_state=42)
df_es_limpio_ft = df_es_limpio.drop(test_es_limpio.index).sample(n=231, random_state=42)
df_es_ft = pd.concat([df_es_alucina_ft, df_es_limpio_ft]).sample(frac=1, random_state=42)

ids_espanol = set(df_test_final['id'].astype(str).tolist() + df_es_ft['id'].astype(str).tolist())

# ==========================================
# 3. CARGAR INGLÉS CON MATEMÁTICA DINÁMICA
# ==========================================
print("\n[3] Cargando Inglés y aplicando filtro Anti-Fuga...")
df_en = pd.read_json("ragtruth_en_internet.jsonl", lines=True)

def tiene_alucinacion_en(row):
    labels = row.get('hallucination_labels_processed', {})
    if isinstance(labels, dict):
        return (labels.get('evident_conflict', 0) > 0) or (labels.get('baseless_info', 0) > 0)
    return False

df_en['tiene_alucinacion'] = df_en.apply(tiene_alucinacion_en, axis=1)
df_en['idioma_asignado'] = 'en'

df_en_filtrado = df_en[~df_en['id'].astype(str).isin(ids_espanol)]
df_en_alucina = df_en_filtrado[df_en_filtrado['tiene_alucinacion'] == True]
df_en_limpio = df_en_filtrado[df_en_filtrado['tiene_alucinacion'] == False]

# MATEMÁTICA DINÁMICA PARA NO CRASHEAR
max_en_alucina = len(df_en_alucina)
max_en_limpios = len(df_en_limpio)
objetivo_en = 4389 # Lo ideal para lograr el 95%

target_final_en = min(objetivo_en, max_en_alucina, max_en_limpios)

print(f"  - Extrayendo exactamente {target_final_en} limpios y {target_final_en} alucinaciones en Inglés...")
en_alucina_ft = df_en_alucina.sample(n=target_final_en, random_state=42)
en_limpio_ft = df_en_limpio.sample(n=target_final_en, random_state=42)
df_en_ft = pd.concat([en_alucina_ft, en_limpio_ft]).sample(frac=1, random_state=42)

# ==========================================
# 4. SPLIT 80/20 ESTRATIFICADO CRUZADO
# ==========================================
print("\n[4] Realizando split 80/20...")
train_es, val_es = train_test_split(df_es_ft, test_size=0.20, random_state=42, stratify=df_es_ft['tiene_alucinacion'])
train_en, val_en = train_test_split(df_en_ft, test_size=0.20, random_state=42, stratify=df_en_ft['tiene_alucinacion'])

df_train_raw = pd.concat([train_es, train_en]).sample(frac=1, random_state=42).reset_index(drop=True)
df_val_raw = pd.concat([val_es, val_en]).sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# 5. FORMATEO DE TEXTOS
# ==========================================
def formatear_texto(row):
    lang = row['idioma_asignado']
    tiene_aluc = row['tiene_alucinacion']

    if lang == 'es':
        prompt = row['prompt_es']
        respuesta = row['response_es']
        if tiene_aluc:
            etiquetas = row['parsed_labels']
            tipo_aluc = etiquetas[0].get("label_type", "Alucinación general") if etiquetas else "Alucinación general"
            frase = etiquetas[0].get("text", "Texto inventado") if etiquetas else "Texto inventado"
            veredicto = f"Sí, se detectó una alucinación del tipo '{tipo_aluc}'. El modelo generó la siguiente información sin respaldo en el contexto: \"{frase}\"."
        else:
            veredicto = "No, la respuesta es correcta, segura y está totalmente respaldada por los pasajes del contexto."
        texto_final = f"### Tarea: Analiza si la siguiente respuesta contiene alucinaciones basándote en el contexto.\n\n### Contexto y Pregunta:\n{prompt}\n\n### Respuesta a evaluar:\n{respuesta}\n\n### Veredicto del Auditor:\n{veredicto}"
    else:
        q = row.get("query", "")
        c = row.get("context", "")
        respuesta = row.get("output", "")
        prompt = f"Question: {q}\n\nContext:\n{c}"
        if tiene_aluc:
            veredicto = "Yes, a hallucination was detected. The model generated unsupported information."
        else:
            veredicto = "No, the response is correct, safe, and fully supported by the context passages."
        texto_final = f"### Task: Analyze if the following response contains hallucinations based on the context.\n\n### Context and Question:\n{prompt}\n\n### Response to evaluate:\n{respuesta}\n\n### Auditor Verdict:\n{veredicto}"

    row['text'] = texto_final
    return row

print("[5] Ensamblando textos finales...")
df_test_final = df_test_final.apply(formatear_texto, axis=1)
df_train_final = df_train_raw.apply(formatear_texto, axis=1)
df_val_final = df_val_raw.apply(formatear_texto, axis=1)

# ==========================================
# 6. EXPORTAR Y VALIDAR MÉTRICAS
# ==========================================
print("\n" + "="*50)
print(" VALIDACIÓN MATEMÁTICA DEL DATASET ")
print("="*50)

print(f"\n1. TAMAÑOS FINALES:")
print(f"   Train: {len(df_train_final)} | Validation: {len(df_val_final)} | Test Eval: {len(df_test_final)}")

print("\n2. BALANCE DE ALUCINACIONES (Debería ser 50% en TODOS lados):")
print(f"   Test Eval:  Alucinan {sum(df_test_final['tiene_alucinacion'])} | Limpios {sum(~df_test_final['tiene_alucinacion'])}")
print(f"   Train:      Alucinan {sum(df_train_final['tiene_alucinacion'])} | Limpios {sum(~df_train_final['tiene_alucinacion'])}")
print(f"   Validation: Alucinan {sum(df_val_final['tiene_alucinacion'])} | Limpios {sum(~df_val_final['tiene_alucinacion'])}")

print("\n3. DISTRIBUCIÓN CROSS-LINGUAL (Debería ser ~95% EN / 5% ES):")
print("   En TRAIN:")
print(df_train_final['idioma_asignado'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
print("   En VALIDATION:")
print(df_val_final['idioma_asignado'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Guardar con nuevos nombres inconfundibles
df_test_final[['text']].to_json("evaluacion_1200_es.jsonl", orient="records", lines=True, force_ascii=False)
df_train_final[['text']].to_json("train_95en_5es.jsonl", orient="records", lines=True, force_ascii=False)
df_val_final[['text']].to_json("val_95en_5es.jsonl", orient="records", lines=True, force_ascii=False)

print("\n¡Archivos guardados con éxito!")