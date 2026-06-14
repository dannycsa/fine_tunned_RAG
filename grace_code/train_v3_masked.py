# train_v3_masked.py
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import wandb

wandb.init(project="ragtruth-fine-tuning", name="gemma-2b-v3-final")

MAX_SEQ_LEN = 1024
SEPARADOR   = "### Auditor Verdict:\n"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/gemma-2-2b-bnb-4bit",
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)
EOS = tokenizer.eos_token
print(f"EOS token: {repr(EOS)}")

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_alpha     = 32,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 3407,
)

def tokenizar_con_mascara(examples):
    input_ids_lista, attention_mask_lista, labels_lista = [], [], []
    for texto in examples["text"]:
        texto = texto.replace("### Veredicto del Auditor:\n",  SEPARADOR)
        texto = texto.replace("### Tarea: Analiza si",         "### Task: Analyze if")
        texto = texto.replace("### Contexto y Pregunta:\n",    "### Context and Question:\n")
        texto = texto.replace("### Respuesta a evaluar:\n",    "### Response to evaluate:\n")
        texto = texto + EOS

        partes = texto.split(SEPARADOR)
        if len(partes) < 2:
            continue

        prompt    = partes[0] + SEPARADOR
        respuesta = partes[1]

        prompt_ids = tokenizer(prompt, truncation=True,
                               max_length=MAX_SEQ_LEN-10,
                               add_special_tokens=True)["input_ids"]
        resp_ids   = tokenizer(respuesta, truncation=False,
                               add_special_tokens=False)["input_ids"]

        full_ids = (prompt_ids + resp_ids)[:MAX_SEQ_LEN]
        labels   = ([-100]*len(prompt_ids) + resp_ids)[:MAX_SEQ_LEN]
        pad_len  = MAX_SEQ_LEN - len(full_ids)

        input_ids_lista.append(full_ids   + [0]*pad_len)
        attention_mask_lista.append([1]*len(full_ids) + [0]*pad_len)
        labels_lista.append(labels        + [-100]*pad_len)

    return {"input_ids"      : input_ids_lista,
            "attention_mask" : attention_mask_lista,
            "labels"         : labels_lista}

dataset_raw = load_dataset("json", data_files={
    "train":      "train_95en_5es.jsonl",
    "validation": "val_95en_5es.jsonl"
})

print("Tokenizando datasets...")
dataset_train = dataset_raw["train"].map(
    tokenizar_con_mascara, batched=True,
    remove_columns=dataset_raw["train"].column_names)
dataset_val = dataset_raw["validation"].map(
    tokenizar_con_mascara, batched=True,
    remove_columns=dataset_raw["validation"].column_names)
dataset_train.set_format("torch")
dataset_val.set_format("torch")

# Verificar que las máscaras están bien
ej = dataset_train[0]
labels_reales = [l for l in ej["labels"].tolist() if l != -100]
print(f"\n✅ El modelo aprenderá SOLO esto:")
print(f"   {repr(tokenizer.decode(labels_reales))}")
print(f"   (Debe verse Yes.../No... con <eos> al final)\n")

args = TrainingArguments(
    output_dir                  = "resultados_v3",
    max_steps                   = 800,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 8,
    warmup_steps                = 40,
    learning_rate               = 2e-4,
    fp16  = not is_bfloat16_supported(),
    bf16  = is_bfloat16_supported(),
    logging_steps               = 10,
    eval_strategy               = "steps",
    eval_steps                  = 267,
    save_strategy               = "steps",
    save_steps                  = 267,
    save_total_limit            = 3,
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",
    seed                        = 3407,
    report_to                   = "wandb",
    remove_unused_columns       = False,
)

trainer = Trainer(
    model         = model,
    args          = args,
    train_dataset = dataset_train,
    eval_dataset  = dataset_val,
)

torch.cuda.empty_cache()
print("="*50)
print("INICIANDO ENTRENAMIENTO V3")
print("Max steps : 800  (~2 horas)")
print("Val en    : steps 267, 534, 801")
print("Guarda en : steps 267, 534, 801")
print("="*50)
trainer.train()

print("\n¡ENTRENAMIENTO COMPLETO!")
print("Siguiente paso: python test_rapido_v3.py")
wandb.finish()