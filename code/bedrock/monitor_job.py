"""
monitor_job.py
==============
Vigila el job de fine-tuning (lee last_job.txt), y al terminar descarga las
métricas paso a paso a CSV local. Si wandb está instalado y autenticado, también
las sube a Weights & Biases.

Uso (en segundo plano):
    python monitor_job.py
"""
import os
import time
import json
import boto3

REGION = "us-east-1"
AQUI = os.path.dirname(os.path.abspath(__file__))

arn, bucket, prefix = open(os.path.join(AQUI, "last_job.txt")).read().split("\n")[:3]
bedrock = boto3.client("bedrock", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

print(f"Monitoreando: {arn}")
while True:
    info = bedrock.get_model_customization_job(jobIdentifier=arn)
    status = info["status"]
    print(f"[{time.strftime('%H:%M:%S')}] status = {status}", flush=True)
    if status in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(180)

if status != "Completed":
    print(f"Terminó en {status}: {info.get('failureMessage', 'sin detalle')}")
    raise SystemExit(2)

job_id = arn.split("/")[-1]
base = f"{prefix}/output/model-customization-job-{job_id}"
keys = {
    "train_metrics.csv": f"{base}/training_artifacts/step_wise_training_metrics.csv",
    "val_metrics.csv": f"{base}/validation_artifacts/post_fine_tuning_validation/validation_metrics.csv",
}
for local, key in keys.items():
    try:
        s3.download_file(bucket, key, os.path.join(AQUI, local))
        print(f"descargado: {local}")
    except Exception as e:
        print(f"no pude bajar {key}: {e}")

print("outputModelArn:", info.get("outputModelArn"))
print("LISTO. Métricas locales en code/bedrock/. Para W&B: pip install wandb && wandb login, luego push_to_wandb.py")
