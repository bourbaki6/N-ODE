

import argparse
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.classifier import NeuralODEClassifier
from data.dataset import get_dataloaders
from utils import get_device, load_checkpoint


def benchmark_solver(
    model: NeuralODEClassifier,
    loader,
    device: torch.device,
    solver: str,
    num_steps: int = None,
    rtol: float = None,
    n_batches: int = 20,
) -> dict:
   
    model.set_solver(solver, num_steps=num_steps)
    if rtol is not None:
        model.ode_block.rtol = rtol
        model.ode_block.atol = rtol * 0.1

    model.eval()

    correct = 0
    total = 0
    nfe_list = []
    times = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= n_batches:
                break
            images, labels = images.to(device), labels.to(device)

            t0 = time.perf_counter()
            log_probs = model(images)
            t1 = time.perf_counter()

            preds = log_probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            nfe_list.append(model.nfe)
            times.append((t1 - t0) * 1000 / images.size(0)) 

    return {
        "accuracy": correct / total,
        "avg_time_ms": np.mean(times),
        "avg_nfe": np.mean(nfe_list),
    }


def run_comparison(model, test_loader, device, save_dir: str):

    results = {}

    print("\n Benchmarking Euler solver")
    euler_steps = [2, 4, 6, 8, 10, 15, 20]
    results["euler"] = []
    for steps in euler_steps:
        r = benchmark_solver(model, test_loader, device, "euler", num_steps = steps)
        r["num_steps"] = steps
        results["euler"].append(r)
        print(f"  Euler steps = {steps:3d}: acc = {r['accuracy']*100:.2f}%  "
              f"time = {r['avg_time_ms']:.2f}ms  NFE={r['avg_nfe']:.0f}")

    print("\n Benchmarking RK4 solver")
    rk4_steps = [2, 4, 6, 8, 10, 15, 20]
    results["rk4"] = []
    for steps in rk4_steps:
        r = benchmark_solver(model, test_loader, device, "rk4", num_steps = steps)
        r["num_steps"] = steps
        results["rk4"].append(r)
        print(f"  RK4   steps={steps:3d}: acc={r['accuracy']*100:.2f}%  "
              f"time={r['avg_time_ms']:.2f}ms  NFE = {r['avg_nfe']:.0f}")

    print("\n Benchmarking RK45 solver")
    tolerances = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
    results["rk45"] = []
    for tol in tolerances:
        r = benchmark_solver(model, test_loader, device, "rk45", rtol=tol)
        r["rtol"] = tol
        results["rk45"].append(r)
        print(f"  RK45  tol={tol:.0e}: acc={r['accuracy']*100:.2f}%  "
              f"time={r['avg_time_ms']:.2f}ms  NFE={r['avg_nfe']:.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Solver Comparison: Accuracy / Speed / NFE Tradeoffs", fontsize = 13, fontweight = "bold")


    ax = axes[0]
    for solver, color in [("euler", "tomato"), ("rk4", "steelblue"), ("rk45", "seagreen")]:
        accs = [r["accuracy"] * 100 for r in results[solver]]
        nfes = [r["avg_nfe"] for r in results[solver]]
        ax.plot(nfes, accs, "o-", color = color, label = solver.upper(), linewidth = 2)

    ax.set_xlabel("Average NFE per forward pass", fontsize = 11)
    ax.set_ylabel("Test Accuracy (%)", fontsize = 11)
    ax.set_title("Accuracy vs NFE", fontsize = 10)
    ax.legend()
    ax.grid(True, alpha = 0.3)
    ax.annotate("\n Higher accuracy\n and  Fewer evaluations\n(top-left corner is ideal)",
                xy = (0.02, 0.02), xycoords="axes fraction", fontsize = 8,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha = 0.8))

    ax2 = axes[1]
    for solver, color in [("euler", "tomato"), ("rk4", "steelblue"), ("rk45", "seagreen")]:
        accs = [r["accuracy"] * 100 for r in results[solver]]
        times = [r["avg_time_ms"] for r in results[solver]]
        ax2.plot(times, accs, "o-", color = color, label = solver.upper(), linewidth = 2)

    ax2.set_xlabel("Avg inference time (ms/sample)", fontsize = 11)
    ax2.set_ylabel("Test Accuracy (%)", fontsize = 11)
    ax2.set_title("Accuracy vs Speed", fontsize = 10)
    ax2.legend()
    ax2.grid(True, alpha = 0.3)


    ax3 = axes[2]
    for solver, color in [("euler", "tomato"), ("rk4", "steelblue")]:
        steps = [r["num_steps"] for r in results[solver]]
        nfes = [r["avg_nfe"] for r in results[solver]]
        ax3.plot(steps, nfes, "o-", color = color, label = solver.upper(), linewidth = 2)

    ax3.set_xlabel("num_steps", fontsize = 11)
    ax3.set_ylabel("NFE", fontsize = 11)
    ax3.set_title("NFE vs num_steps (fixed solvers)\nSlope: Euler=1×, RK4=4×", fontsize = 10) 
    ax3.legend()
    ax3.grid(True, alpha = 0.3)
    ax3.annotate("Slope ratio shows\nRK4 costs 4× NFE\nper step vs Euler",
                xy = (0.6, 0.05), xycoords = "axes fraction", fontsize = 8,
                bbox = dict(boxstyle = "round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    save_path = f"{save_dir}/solver_comparison.png"
    plt.savefig(save_path, dpi = 150, bbox_inches = "tight")
    print(f"\nPlot saved: {save_path}")
    plt.show()

    return results


def main():
    parser = argparse.ArgumentParser(description = "Solver Comparison")
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--config", type = str, default = "configs/mnist.yaml")
    parser.add_argument("--save_dir", type = str, default = "analysis/plots")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()

    m_cfg = cfg["model"]
    model = NeuralODEClassifier(
        hidden_dim = m_cfg["hidden_dim"],
        num_classes = m_cfg["num_classes"],
        solver = m_cfg["solver"],
        num_steps = m_cfg["num_steps"],
    ).to(device)
    load_checkpoint(model, args.checkpoint, device=device)

    _, test_loader = get_dataloaders(
        dataset_name = cfg["dataset"]["name"],
        batch_size = cfg["dataset"]["batch_size"],
    )

    run_comparison(model, test_loader, device, args.save_dir)


if __name__ == "__main__":
    main()