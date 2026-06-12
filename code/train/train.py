import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print("=== INICIANDO PIPELINE DE ENTRENAMIENTO PRINCIPAL ===")

wandb.init(project="ragtruth-fine-tuning", name="gemma-2b-cross-lingual")

MODEL_ID = "google/gemma-2-2b"
MAX_SEQ_LEN = 256

print("\n[1] Cargando tokenizer y modelo base optimizado (Gemma 2B)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="eager",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print("\n[2] Cargando y pre-procesando dataset (truncación forzada a 256 tokens)...")
dataset_raw = load_dataset(
    "json",
    data_files={"train": "ragtruth_train.jsonl", "validation": "ragtruth_val.jsonl"}
)

# SOLUCIÓN CLAVE: tokenizar manualmente con truncation=True
# Así garantizamos que NINGUNA secuencia supere 256 tokens antes de llegar al modelo
def tokenize_and_truncate(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,        # <-- esto es lo que faltaba
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    # En CLM el label es el mismo input desplazado
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

cols_a_borrar = dataset_raw["train"].column_names  # elimina columna "text" original

dataset_train = dataset_raw["train"].map(
    tokenize_and_truncate,
    batched=True,
    remove_columns=cols_a_borrar,
    desc="Tokenizando train"
)

dataset_val = dataset_raw["validation"].map(
    tokenize_and_truncate,
    batched=True,
    remove_columns=cols_a_borrar,
    desc="Tokenizando validación"
)

print("\n[3] Configurando adaptadores LoRA de bajo rango...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("\n[4] Estableciendo hiperparámetros de SFT...")
args_entrenamiento = SFTConfig(
    output_dir="./resultados_ragtruth",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    dataloader_pin_memory=False,
    # dataset_text_field ya NO se pasa: el dataset ya está tokenizado
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    peft_config=lora_config,
    args=args_entrenamiento,
)

torch.cuda.empty_cache()

print("\n[5] Iniciando fine-tuning. Monitorea en WandB.")
trainer.train()

print("\n[6] Guardando adaptadores LoRA entrenados...")
trainer.save_model("./modelo_final_ragtruth_bilingue")
wandb.finish()
print("\n¡ENTRENAMIENTO EXITOSO!")