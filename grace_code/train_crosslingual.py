import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import gc
torch.cuda.empty_cache()
gc.collect()
print("=== INICIANDO PIPELINE DE ENTRENAMIENTO CROSS-LINGUAL (VRAM 6GB OPTIMIZED) ===")

# Limpiamos caché de CUDA por si quedó basura del intento anterior
torch.cuda.empty_cache()

wandb.init(project="ragtruth-fine-tuning", name="gemma-2b-crosslingual-3epochs-1024")
MODEL_ID = "google/gemma-2-2b"

# REDUCIDO A 1024 PARA SOBREVIVIR A LOS 6GB DE VRAM
MAX_SEQ_LEN = 1024 

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
    attn_implementation="sdpa",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print("\n[2] Cargando y pre-procesando dataset Cross-Lingual (95% EN / 5% ES)...")
dataset_raw = load_dataset(
    "json",
    data_files={"train": "train_95en_5es.jsonl", "validation": "val_95en_5es.jsonl"}
)

def tokenize_and_truncate(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

cols_a_borrar = dataset_raw["train"].column_names

dataset_train = dataset_raw["train"].map(tokenize_and_truncate, batched=True, remove_columns=cols_a_borrar, desc="Tokenizando train")
dataset_val = dataset_raw["validation"].map(tokenize_and_truncate, batched=True, remove_columns=cols_a_borrar, desc="Tokenizando val")

print("\n[3] Configurando adaptadores LoRA ultra-ligeros (r=4)...")
lora_config = LoraConfig(
    r=4,               # <--- Reducido de 8 a 4 (Ahorro masivo de VRAM)
    lora_alpha=8,      # <--- Reducido de 16 a 8 para mantener la proporción
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("\n[4] Estableciendo hiperparámetros de SFT (3 Épocas, Checkpoints cada 150)...")
args_entrenamiento = SFTConfig(
    output_dir="./resultados_crosslingual",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,       
    eval_accumulation_steps=1,          
    gradient_accumulation_steps=8,
    num_train_epochs=3,                 
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,                   
    eval_strategy="steps",
    eval_steps=150,                     
    save_strategy="steps",
    save_steps=150,                     
    optim="paged_adamw_8bit", # CLAVE PARA NO QUEDARNOS SIN VRAM
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    peft_config=lora_config,
    args=args_entrenamiento,
)

# Última limpieza profunda antes de iniciar
torch.cuda.empty_cache()

print("\n[5] Iniciando fine-tuning. Monitorea en WandB.")
trainer.train()

print("\n[6] Guardando adaptadores LoRA entrenados finales...")
trainer.save_model("./modelo_final_crosslingual")
wandb.finish()
print("\n¡ENTRENAMIENTO EXITOSO!")