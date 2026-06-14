# Fine-tuning en AWS Bedrock (alternativa "nube", no on-premise)

Esta carpeta migra el entrenamiento del modelo (que localmente era Gemma-2-2B +
LoRA con `bitsandbytes`, **on-premise**, requiere tu GPU) a **fine-tuning
gestionado de Amazon Bedrock**: subes tu dataset a S3, Bedrock entrena su modelo
base (Amazon Nova / Titan / Llama) y tú lo consumes por API. No manejas GPU.

## Flujo

```
ragtruth_train.jsonl / ragtruth_val.jsonl   (los 400/400 que genera ../train/build_dataset_400.py)
        │
        ▼  prepare_bedrock_dataset.py
bedrock_train.jsonl / bedrock_val.jsonl      (formato Nova: bedrock-conversation-2024)
        │
        ▼  finetune_bedrock.py  (sube a S3 → create_model_customization_job → polling)
modelo customizado (ARN)  +  métricas en W&B (train/loss, eval/loss)
```

## Pasos

1. **Generar los datos base** (si no lo hiciste):
   ```bash
   cd ../train && python build_dataset_400.py && cd ../bedrock
   ```

2. **Convertir al formato Bedrock**:
   ```bash
   python prepare_bedrock_dataset.py
   # -> bedrock_train.jsonl (800) y bedrock_val.jsonl (400)
   ```

3. **Averiguar el ID exacto del modelo base** fine-tuneable:
   ```bash
   python list_finetunable_models.py
   ```

4. **Lanzar el fine-tuning** (3 épocas, con validación, métricas a W&B):
   ```bash
   python finetune_bedrock.py \
     --bucket   TU_BUCKET_S3 \
     --role-arn arn:aws:iam::TU_CUENTA:role/BedrockCustomizationRole \
     --base-model amazon.nova-micro-v1:0:128k \
     --epochs   3 \
     --wandb-project ragtruth-fine-tuning
   ```

## Prerequisitos (importantes)

- **Acceso al modelo habilitado** en la consola de Bedrock (Model access) para el
  modelo que vayas a tunear.
- **Bucket S3** en la misma región (por defecto `us-east-1`, igual que el
  `qa.py` de traducción).
- **Rol IAM** (`--role-arn`) con:
  - *trust policy* que permita a `bedrock.amazonaws.com` asumir el rol;
  - permisos `s3:GetObject`/`s3:PutObject`/`s3:ListBucket` sobre tu bucket.
- `pip install boto3 wandb` y `wandb login` (o `WANDB_API_KEY`).

## Notas / límites honestos

- **"Guardar cada 200 pasos":** Bedrock **no** expone esa cadencia como
  parámetro. Controla los checkpoints internamente y al terminar deja las
  métricas paso a paso en S3
  (`training_artifacts/step_wise_training_metrics.csv` y
  `validation_artifacts/post_fine_tuning_validation/validation_metrics.csv`).
  El script las lee y las sube a W&B, que es el objetivo real de la tarjeta.
- **Validación / eval_loss:** la genera la familia clásica (Titan, **Nova 1.0**
  = `nova-micro/lite/pro v1`, Llama). En **Nova 2.0 SFT** el set de validación
  puede no usarse durante el entrenamiento → si quieres curva de `eval_loss`,
  usa Nova Micro **v1** o Titan.
- **Costo:** el fine-tuning se cobra por tokens procesados, y **para invocar** el
  modelo resultante necesitas *Provisioned Throughput*
  (`create_provisioned_model_throughput`), que se factura por hora mientras esté
  activo. Acuérdate de borrarlo al terminar.
- No se puede tunear "Gemma" en Bedrock: aquí el modelo es el del proveedor
  (Nova/Titan/Llama). Si el requisito fuera específicamente Gemma, la ruta sería
  GPU en la nube (Colab/SageMaker) con `../train/train.py`, no Bedrock.
```
