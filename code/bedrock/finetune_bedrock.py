"""
finetune_bedrock.py
===================
Lanza un job de fine-tuning (SFT) en Amazon Bedrock con el dataset 400/400,
3 épocas, usando datos de validación, y al terminar empuja las métricas
(training_loss y validation_loss por paso) a Weights & Biases.

Equivalencias con la tarjeta de Trello:
  - "400 ES + 400 EN, 3 épocas"  -> hyperParameters={"epochCount": "3"} sobre
    bedrock_train.jsonl (800 ejemplos).
  - "guardar cada 200 pasos eval_loss" -> Bedrock NO expone un knob "cada N pasos";
    él decide la cadencia y emite automáticamente las métricas paso a paso en S3
    (validation_metrics.csv). Las leemos y las mandamos a W&B (FIX equivalente a
    eval_steps del entrenamiento local).
  - "mandar a W&B" -> se loguea train/loss y val/loss por paso al final del job.

IMPORTANTE sobre validación:
  Bedrock genera validation_metrics.csv para la familia clásica de customización
  (Amazon Titan, Amazon Nova 1.0 -> nova-micro/lite/pro v1, Meta Llama). En
  Amazon Nova **2.0** SFT el set de validación puede NO usarse durante el
  entrenamiento. Si necesitas la curva de eval_loss, elige un base model que
  soporte validación (p. ej. Nova Micro v1 o Titan). Verifica el ID exacto con
  list_finetunable_models.py.

Prerequisitos: ver README.md (rol IAM, bucket S3, acceso al modelo habilitado).

Uso:
    python finetune_bedrock.py \
        --bucket   mi-bucket-bedrock \
        --role-arn arn:aws:iam::<cuenta>:role/BedrockCustomizationRole \
        --base-model amazon.nova-micro-v1:0:128k \
        --region   us-east-1 \
        --epochs   3 \
        --wandb-project ragtruth-fine-tuning
"""

import os
import csv
import time
import argparse

import boto3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True, help="Bucket S3 (sin s3://)")
    p.add_argument("--prefix", default="ragtruth-ft", help="Prefijo/carpeta en S3")
    p.add_argument("--role-arn", required=True,
                   help="ARN del rol IAM que Bedrock asume para leer/escribir S3")
    p.add_argument("--base-model", required=True,
                   help="baseModelIdentifier (usa list_finetunable_models.py)")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--job-name", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--wandb-project", default="ragtruth-fine-tuning")
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def upload(s3, local, bucket, key):
    print(f"  subiendo {os.path.basename(local)} -> s3://{bucket}/{key}")
    s3.upload_file(local, bucket, key)
    return f"s3://{bucket}/{key}"


def descargar_csv(s3, bucket, key):
    """Devuelve lista de dicts desde un CSV en S3, o [] si no existe."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        contenido = obj["Body"].read().decode("utf-8").splitlines()
        return list(csv.DictReader(contenido))
    except s3.exceptions.NoSuchKey:
        return []
    except Exception as e:
        print(f"  (no pude leer s3://{bucket}/{key}: {e})")
        return []


def main():
    args = parse_args()
    aqui = os.path.dirname(os.path.abspath(__file__))
    train_local = os.path.join(aqui, "bedrock_train.jsonl")
    val_local = os.path.join(aqui, "bedrock_val.jsonl")
    for f in (train_local, val_local):
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Falta {f}. Ejecuta primero: python prepare_bedrock_dataset.py")

    # Nombres únicos sin depender de time.time() en runtime de notebooks:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"ragtruth-sft-{stamp}"
    model_name = args.model_name or f"ragtruth-custom-{stamp}"

    s3 = boto3.client("s3", region_name=args.region)
    bedrock = boto3.client("bedrock", region_name=args.region)

    # 1) Subir datasets a S3
    print("\n[1] Subiendo datasets a S3...")
    train_uri = upload(s3, train_local, args.bucket, f"{args.prefix}/train/bedrock_train.jsonl")
    val_uri = upload(s3, val_local, args.bucket, f"{args.prefix}/val/bedrock_val.jsonl")
    output_prefix = f"{args.prefix}/output/"
    output_uri = f"s3://{args.bucket}/{output_prefix}"

    # 2) Crear el job de fine-tuning
    print("\n[2] Creando job de customización (FINE_TUNING)...")
    resp = bedrock.create_model_customization_job(
        jobName=job_name,
        customModelName=model_name,
        roleArn=args.role_arn,
        baseModelIdentifier=args.base_model,
        customizationType="FINE_TUNING",
        trainingDataConfig={"s3Uri": train_uri},
        validationDataConfig={"validators": [{"s3Uri": val_uri}]},
        outputDataConfig={"s3Uri": output_uri},
        hyperParameters={
            "epochCount": str(args.epochs),   # <-- 3 épocas
            # Otros knobs opcionales según el modelo base:
            # "batchSize": "1",
            # "learningRate": "0.00002",
            # "learningRateWarmupSteps": "0",
        },
    )
    job_arn = resp["jobArn"]
    print(f"  jobArn: {job_arn}")

    # 3) Esperar a que termine
    print("\n[3] Esperando a que el job termine (puede tardar)...")
    while True:
        info = bedrock.get_model_customization_job(jobIdentifier=job_arn)
        status = info["status"]
        print(f"  estado: {status}")
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(args.poll_seconds)

    if status != "Completed":
        print(f"\n❌ El job terminó en estado {status}: "
              f"{info.get('failureMessage', 'sin detalle')}")
        return

    # Carpeta de salida concreta: Bedrock usa el ID del training job
    job_id = job_arn.split("/")[-1]
    base = f"{output_prefix}model-customization-job-{job_id}"
    train_metrics_key = f"{base}/training_artifacts/step_wise_training_metrics.csv"
    val_metrics_key = f"{base}/validation_artifacts/post_fine_tuning_validation/validation_metrics.csv"

    # 4) Leer métricas y mandarlas a W&B
    print("\n[4] Leyendo métricas de S3 y enviando a Weights & Biases...")
    train_rows = descargar_csv(s3, args.bucket, train_metrics_key)
    val_rows = descargar_csv(s3, args.bucket, val_metrics_key)
    print(f"  train steps: {len(train_rows)} | val steps: {len(val_rows)}")

    try:
        import wandb
        wandb.init(project=args.wandb_project, name=job_name,
                   config={"base_model": args.base_model, "epochs": args.epochs,
                           "platform": "aws-bedrock"})
        for r in train_rows:
            step = int(float(r.get("step_number", r.get("step", 0))))
            loss = r.get("training_loss")
            if loss not in (None, ""):
                wandb.log({"train/loss": float(loss)}, step=step)
        for r in val_rows:
            step = int(float(r.get("step_number", r.get("step", 0))))
            loss = r.get("validation_loss")
            if loss not in (None, ""):
                wandb.log({"eval/loss": float(loss)}, step=step)
        wandb.summary["custom_model_arn"] = info.get("outputModelArn")
        wandb.finish()
        print("  ✅ métricas enviadas a W&B.")
    except ImportError:
        print("  (wandb no instalado: `pip install wandb`. Métricas no enviadas.)")

    print(f"\n✅ Listo. Modelo customizado: {info.get('outputModelArn')}")
    print("Para INVOCARLO necesitas Provisioned Throughput "
          "(create_provisioned_model_throughput). Ver README.md.")


if __name__ == "__main__":
    main()
