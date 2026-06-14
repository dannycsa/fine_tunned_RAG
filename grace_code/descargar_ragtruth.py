from datasets import load_dataset
import pandas as pd

print("Descargando RAGTruth original en inglés desde Hugging Face...")
# Descargamos la versión procesada del dataset original
dataset = load_dataset("wandb/RAGTruth-processed", split="train")

# Lo convertimos a Pandas para guardarlo fácil
df_en = dataset.to_pandas()

# Guardamos el archivo como JSONL
df_en.to_json("ragtruth_descargado_en.jsonl", orient="records", lines=True, force_ascii=False)

print(f"¡Listo! Se guardaron {len(df_en)} ejemplos en 'ragtruth_descargado_en.jsonl'.")