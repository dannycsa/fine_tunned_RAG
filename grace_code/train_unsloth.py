
# LAS PRIMERAS LÍNEAS DE TU SCRIPT DEBEN SER ESTAS:
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

import os
import wandb

print("=== INICIANDO PIPELINE CROSS-LINGUAL CON UNSLOTH (MAGIA PARA 6GB VRAM) ===")

wandb.init(project="ragtruth-fine-tuning", name="gemma-2b-crosslingual-unsloth-1024")

MAX_SEQ_LEN = 1024
# Modifica esta línea en tu train_unsloth.py:
MODEL_ID = "unsloth/gemma-2-2b-bnb-4bit"
print("\n[1] Cargando modelo y tokenizer con Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = MAX_SEQ_LEN,
    dtype = None, # Autodetecta bf16
    load_in_4bit = True, # QLoRA automático
)

print("\n[2] Configurando adaptadores LoRA ultra-optimizados...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 4, # Mantenemos r=4 para cuidar la memoria
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 8,
    lora_dropout = 0, # Unsloth requiere 0 para máximo rendimiento
    bias = "none",
    use_gradient_checkpointing = "unsloth", # ESTA ES LA CLAVE PARA NO QUEDAR OOM
    random_state = 3407,
)

print("\n[3] Cargando y pre-procesando dataset...")
dataset_raw = load_dataset(
    "json",
    data_files={"train": "train_95en_5es.jsonl", "validation": "val_95en_5es.jsonl"}
)

def tokenize_and_truncate(examples):
    # Unsloth requiere que simplemente pasemos el texto, SFTTrainer se encarga del resto
    return {"text": examples["text"]}

dataset_train = dataset_raw["train"].map(tokenize_and_truncate, batched=True, desc="Procesando train")
dataset_val = dataset_raw["validation"].map(tokenize_and_truncate, batched=True, desc="Procesando val")

print("\n[4] Estableciendo hiperparámetros de SFT...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    eval_dataset = dataset_val,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LEN,
    dataset_num_proc = 2,
    args = SFTConfig(               # <--- ¡ESTE ES EL CAMBIO MÁGICO!
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        eval_strategy = "steps",
        eval_steps = 150,
        save_strategy = "steps",
        save_steps = 150,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "resultados_unsloth",
        report_to = "wandb",
    ),
)

torch.cuda.empty_cache()

print("\n[5] Iniciando fine-tuning ultra-rápido...")
trainer.train()

print("\n[6] Guardando modelo final...")
model.save_pretrained_merged("modelo_final_unsloth", tokenizer, save_method = "lora")
wandb.finish()
print("\n¡ENTRENAMIENTO EXITOSO!")