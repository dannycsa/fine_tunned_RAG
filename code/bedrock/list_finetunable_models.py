"""
list_finetunable_models.py
==========================
Lista los modelos base de Amazon Bedrock que admiten FINE-TUNING en tu región.
Úsalo para obtener el `baseModelIdentifier` EXACTO (los IDs cambian y traen
sufijos como ':0:128k'), en vez de adivinarlo.

Uso:
    python list_finetunable_models.py            # región us-east-1 por defecto
    AWS_REGION=us-west-2 python list_finetunable_models.py
"""

import os
import boto3

region = os.environ.get("AWS_REGION", "us-east-1")
bedrock = boto3.client("bedrock", region_name=region)

resp = bedrock.list_foundation_models(byCustomizationType="FINE_TUNING")
modelos = resp.get("modelSummaries", [])

print(f"Modelos fine-tuneables en {region} ({len(modelos)}):\n")
for m in modelos:
    print(f"  modelId          : {m.get('modelId')}")
    print(f"  modelName        : {m.get('modelName')}")
    print(f"  provider         : {m.get('providerName')}")
    print(f"  customizations   : {m.get('customizationsSupported')}")
    print("  " + "-" * 50)

print("\nCopia el 'modelId' del modelo que quieras (p. ej. una variante de "
      "Amazon Nova Micro o Titan) y pásalo a finetune_bedrock.py como "
      "--base-model.")
