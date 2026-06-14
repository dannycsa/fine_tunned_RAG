"""
push_to_wandb.py
===============
Sube a Weights & Biases las métricas del fine-tuning de Bedrock
(train_metrics.csv / val_metrics.csv que deja monitor_job.py).

Requiere:  pip install wandb  &&  wandb login   (o WANDB_API_KEY en el entorno)

Uso:
    python push_to_wandb.py [--project ragtruth-fine-tuning]
"""
import os
import csv
import argparse

AQUI = os.path.dirname(os.path.abspath(__file__))


def leer(path, loss_col):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        step = r.get("step_number") or r.get("step") or r.get("epoch_number") or 0
        loss = r.get(loss_col)
        if loss not in (None, ""):
            out.append((int(float(step)), float(loss)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="ragtruth-fine-tuning")
    ap.add_argument("--name", default="bedrock-nova-micro-400x2")
    args = ap.parse_args()

    import wandb
    train = leer(os.path.join(AQUI, "train_metrics.csv"), "training_loss")
    val = leer(os.path.join(AQUI, "val_metrics.csv"), "validation_loss")
    print(f"train points: {len(train)} | val points: {len(val)}")

    wandb.init(project=args.project, name=args.name,
               config={"platform": "aws-bedrock",
                       "base_model": "amazon.nova-micro-v1:0:128k", "epochs": 3})
    # Ejes propios para cada curva: así eval/loss (steps 12,24,36) NO choca con el
    # step global del train y se ven TODOS los puntos de validación.
    wandb.define_metric("train/step")
    wandb.define_metric("train/loss", step_metric="train/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/loss", step_metric="eval/step")
    for step, loss in train:
        wandb.log({"train/step": step, "train/loss": loss})
    for step, loss in val:
        wandb.log({"eval/step": step, "eval/loss": loss})
    wandb.finish()
    print("Listo: métricas en W&B.")


if __name__ == "__main__":
    main()
