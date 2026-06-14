"""
provision_and_run.py
=====================
Bootstrap COMPLETO y ejecutable del fine-tuning en Bedrock:
  1. crea (si no existe) un bucket S3
  2. sube bedrock_train.jsonl / bedrock_val.jsonl
  3. crea (si no existe) el rol IAM que Bedrock asume (trust + permisos S3)
  4. lanza create_model_customization_job (Nova Micro v1, 3 épocas, con validación)
  5. imprime el jobArn

Es idempotente: si el bucket o el rol ya existen, los reutiliza.

Uso:
    python provision_and_run.py [--base-model amazon.nova-micro-v1:0:128k] [--epochs 3]
"""

import os
import json
import time
import argparse

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
ROLE_NAME = "BedrockRagtruthCustomizationRole"
AQUI = os.path.dirname(os.path.abspath(__file__))


def ensure_bucket(s3, bucket):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  bucket ya existe: {bucket}")
    except ClientError:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket)  # us-east-1 NO admite LocationConstraint
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={"LocationConstraint": REGION})
        print(f"  bucket creado: {bucket}")


def ensure_role(iam, account, bucket):
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account},
                "ArnEquals": {
                    "aws:SourceArn":
                    f"arn:aws:bedrock:{REGION}:{account}:model-customization-job/*"
                },
            },
        }],
    }
    perms = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
        }],
    }
    try:
        arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"  rol ya existe: {arn}")
    except ClientError:
        arn = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Rol para fine-tuning RAGTruth en Bedrock",
        )["Role"]["Arn"]
        print(f"  rol creado: {arn}")
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="S3AccessRagtruth",
        PolicyDocument=json.dumps(perms),
    )
    print("  política S3 adjuntada.")
    return arn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="amazon.nova-micro-v1:0:128k")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    train_local = os.path.join(AQUI, "bedrock_train.jsonl")
    val_local = os.path.join(AQUI, "bedrock_val.jsonl")
    for f in (train_local, val_local):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Falta {f}. Corre prepare_bedrock_dataset.py")

    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"bedrock-ragtruth-ft-{account}"
    prefix = "ragtruth-ft"

    s3 = boto3.client("s3", region_name=REGION)
    iam = boto3.client("iam")
    bedrock = boto3.client("bedrock", region_name=REGION)

    print("\n[1] Bucket S3...")
    ensure_bucket(s3, bucket)

    print("\n[2] Subiendo datasets...")
    train_key = f"{prefix}/train/bedrock_train.jsonl"
    val_key = f"{prefix}/val/bedrock_val.jsonl"
    s3.upload_file(train_local, bucket, train_key)
    s3.upload_file(val_local, bucket, val_key)
    print(f"  s3://{bucket}/{train_key}")
    print(f"  s3://{bucket}/{val_key}")

    print("\n[3] Rol IAM...")
    role_arn = ensure_role(iam, account, bucket)
    print("  esperando propagación de IAM (15s)...")
    time.sleep(15)

    print("\n[4] Lanzando job de fine-tuning...")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"ragtruth-sft-{stamp}"
    model_name = f"ragtruth-custom-{stamp}"
    output_uri = f"s3://{bucket}/{prefix}/output/"

    # Reintento por si IAM aún no propagó (error de validación del rol)
    last_err = None
    for intento in range(6):
        try:
            resp = bedrock.create_model_customization_job(
                jobName=job_name,
                customModelName=model_name,
                roleArn=role_arn,
                baseModelIdentifier=args.base_model,
                customizationType="FINE_TUNING",
                trainingDataConfig={"s3Uri": f"s3://{bucket}/{train_key}"},
                validationDataConfig={"validators": [{"s3Uri": f"s3://{bucket}/{val_key}"}]},
                outputDataConfig={"s3Uri": output_uri},
                hyperParameters={"epochCount": str(args.epochs)},
            )
            job_arn = resp["jobArn"]
            print(f"\n[OK] JOB LANZADO")
            print(f"  jobName : {job_name}")
            print(f"  jobArn  : {job_arn}")
            print(f"  output  : {output_uri}")
            with open(os.path.join(AQUI, "last_job.txt"), "w") as f:
                f.write(job_arn + "\n" + bucket + "\n" + prefix + "\n")
            return
        except ClientError as e:
            last_err = e
            msg = str(e)
            if "validation" in msg.lower() or "assume" in msg.lower() or "role" in msg.lower():
                print(f"  intento {intento+1}: IAM aún propagando, reintento en 10s...")
                time.sleep(10)
                continue
            raise
    raise last_err


if __name__ == "__main__":
    main()
