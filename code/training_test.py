import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("=== INICIANDO ENTRENAMIENTO PRINCIPAL (FASE 3) ===")

# 1. Iniciar monitoreo en WandB
wandb.init(project="ragtruth-fine-tuning", name="gemma-2b-bilingue-jsonl")

# 2. Cargar modelo base comprimido
print("\n[1] Cargando modelo base (Gemma 2B)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 3. Cargar tu Dataset Bilingüe Local (.jsonl)
print("\n[2] Cargando dataset local (ragtruth_qa_filtrado_es.jsonl)...")
# Usamos "json" como formato y le pasamos el nombre de tu archivo
dataset = load_dataset("json", data_files="ragtruth_qa_filtrado_es.jsonl", split="train") 

# 4. Inyectar LoRA
print("\n[3] Inyectando adaptadores LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Formatear datos usando tus columnas exactas
def formatear_prompt(ejemplo):
    # Unimos el prompt en español y la respuesta esperada
    texto_formateado = f"{ejemplo['prompt_es']}\nRespuesta: {ejemplo['response_es']}"
    return {"text": texto_formateado}

dataset = dataset.map(formatear_prompt)

# 6. Hiperparámetros de ENTRENAMIENTO COMPLETO
print("\n[4] Configurando entrenamiento completo...")
args_entrenamiento = SFTConfig(
    output_dir="./resultados_ragtruth",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,          
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    logging_steps=5,             
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    report_to="wandb"            
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=args_entrenamiento
)

# 7. ¡Entrenar!
print("\n[5] ¡Iniciando fine-tuning completo! Revisa tu panel web de WandB.")
trainer.train()

# 8. Guardar Modelo
print("\n[6] Guardando el modelo entrenado en tu disco duro...")
trainer.save_model("./modelo_final_ragtruth_bilingue")
wandb.finish()

print("\n✅ ¡FASE 3 COMPLETADA! Tienes tu propio modelo de IA bilingüe guardado.")
