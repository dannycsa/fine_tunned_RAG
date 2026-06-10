import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("=== INICIANDO CORRIDA INICIAL DE FINE-TUNING CON RAGTRUTH ===")

# 1. Configurar compresión a 4-bits (SIN usar la RAM de la computadora)
print("\n[1] Cargando modelo base comprimido (Gemma 2B)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    # ELIMINADO: llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",  # <--- CAMBIO AL MODELO LIGERO
    quantization_config=bnb_config, 
    device_map="auto"     # Ahora todo cabrá 100% en tu tarjeta gráfica
)

# 2. Descargar Dataset RAGTruth
print("\n[2] Descargando dataset RAGTruth de Hugging Face...")
dataset = load_dataset("wandb/RAGTruth-processed", split="train")
print(f"¡Dataset descargado exitosamente! {len(dataset)} ejemplos encontrados.")

# 3. Configurar LoRA (Módulos estándar)
print("\n[3] Inyectando adaptadores LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # <--- Volvemos a los módulos normales
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Formatear los datos
def formatear_prompt(ejemplo):
    texto = f"Aprende a detectar alucinaciones con este ejemplo: {str(ejemplo)[:500]}" 
    return {"text": texto}

dataset = dataset.map(formatear_prompt)

# 5. Configurar el Entrenador 
print("\n[4] Configurando hiperparámetros de entrenamiento...")
args_entrenamiento = SFTConfig(
    output_dir="./resultados_ragtruth",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    logging_steps=2,
    optim="paged_adamw_8bit",
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=args_entrenamiento
)

# 6. ¡Entrenar!
print("\n[5] ¡Iniciando fine-tuning de prueba (10 pasos)!")
trainer.train()

print("\n✅ ¡CORRIDA INICIAL DE FINE-TUNING COMPLETADA CON ÉXITO! La memoria de video se mantuvo estable.")
