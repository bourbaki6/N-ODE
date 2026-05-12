import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional


def get_device() -> torch.device:
  
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok = True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    
    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location = map_location)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}: {path}")
    return checkpoint


class MetricsLogger:

    def __init__(self, log_dir: str = "logs", run_name: str = "run"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents = True, exist_ok = True)   
        self.run_name = run_name
        self.records: list = []
        self.start_time = time.time()

    def log(self, epoch: int, **kwargs):
        record = {
            "epoch": epoch,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            **{
                k: round(float(v), 6) if isinstance(v, (float, int, np.floating)) else v
                for k, v in kwargs.items()
            },
        }
        self.records.append(record)

        parts = [f"Epoch {epoch:3d}"]
        for k, v in kwargs.items():
            parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print("  " + " | ".join(parts))

    def save(self):
        path = self.log_dir / f"{self.run_name}.json"
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"  Logs saved to {path}")

    @staticmethod
    def load(path: str) -> list:
        with open(path) as f:
            return json.load(f)


def compute_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def measure_nfe(model, loader, device: torch.device, num_batches: int = 10) -> float:
    
    model.eval()
    nfe_list = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
            nfe_list.append(model.nfe)

    return float(np.mean(nfe_list))


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
   
    print(f"\n{model_name}")
   
    if hasattr(model, "count_parameters"):
        params = model.count_parameters()
        for k, v in params.items():
            if k != "total":
                print(f"  {k:<20} {v:>8,} params")
        print(f"  {'─' * 30}")
        print(f"  {'total':<20} {params['total']:>8,} params")
    else:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total: {total:,} trainable parameters")
    print()


def print_comparison_table(results: dict):
    header = f"  {'Model':<25} {'Accuracy':>10} {'Params':>12} {'NFE':>8}"
    print(f"\n{header}")
    print(f"  {'─' * 57}")

    for name, metrics in results.items():
        acc = f"{metrics.get('accuracy', 0) * 100:.2f}%"
        par = f"{metrics.get('params', 0):,}"
        nfe = f"{metrics['nfe']:.1f}" if metrics.get("nfe") else "—"
        print(f"  {name:<25} {acc:>10} {par:>12} {nfe:>8}")
    print()
