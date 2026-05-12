#---Plotting NFE evolves over training epochs and 
#   its distribution through all of training ---#


import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

def load_log(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def plot_nfe_over_training(log_paths: list, labels: list, save_path: str = None):
   
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neural ODE: NFE Dynamics Over Training",
                 fontsize=13, fontweight="bold")

    colors = plt.cm.tab10.colors
    ax_nfe, ax_acc = axes

    for i, (path, label) in enumerate(zip(log_paths, labels)):
        records = load_log(path)
        epochs = [r["epoch"] for r in records]
        color = colors[i % len(colors)]

        nfes = [r.get("nfe") for r in records]
        nfe_vals = [n for n in nfes if n is not None]
        nfe_epochs = [e for e, n in zip(epochs, nfes) if n is not None]

        if nfe_vals:
            ax_nfe.plot(
                nfe_epochs, nfe_vals,
                color = color, linewidth = 2, label = label,
                marker = "o", markersize = 3,
            )
            
            window = 5
            if len(nfe_vals) >= window:
                trend = np.convolve(nfe_vals, np.ones(window) / window, mode="valid")
                ax_nfe.plot(
                    nfe_epochs[window - 1:], trend,
                    color=color, linewidth=1.5, linestyle="--", alpha=0.6,
                )

        accs = [r.get("test_acc", 0) * 100 for r in records]
        ax_acc.plot(epochs, accs, color=color, linewidth=2, label=label)

    ax_nfe.set_xlabel("Epoch", fontsize=11)
    ax_nfe.set_ylabel("Avg NFE per forward pass", fontsize=11)
    ax_nfe.set_title(
        "NFE over Training\n(decreasing = smoother vector field = implicit regularisation)",
        fontsize=10,
    )
    ax_nfe.legend()
    ax_nfe.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_nfe.annotate(
        "Lower NFE = learned smoother\nvector field = implicit regularisation",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax_acc.set_title("Test Accuracy over Training", fontsize=10)
    ax_acc.legend()

    all_accs = []
    for path in log_paths:
        recs = load_log(path)
        all_accs += [r.get("test_acc", 0) * 100 for r in recs]
    if all_accs:
        ax_acc.set_ylim(bottom=max(0, min(all_accs) - 5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        default = "nfe_over_training.png"
        plt.savefig(default, dpi=150, bbox_inches="tight")
        print(f"  Saved: {default}")
    plt.close()

def plot_nfe_histogram(log_path: str, save_path: str = None):
    
    records = load_log(log_path)
    nfes = [r.get("nfe") for r in records if "nfe" in r]

    if not nfes:
        print("No NFE data in log.")
        return

    n = len(nfes)
    early = nfes[:n//3]
    mid = nfes[n//3 : 2*n//3]
    late = nfes[2*n//3:]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("NFE Distribution: Early vs Late Training", fontsize=13)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("NFE Distribution: Early vs Mid vs Late Training",
                 fontsize=13, fontweight="bold")

    lo = min(nfes) - 1
    hi = max(nfes) + 1

    bins = np.linspace(lo, hi, max(20, int(hi - lo) + 2))
    
    ax.hist(early, bins = bins, alpha = 0.5, label = "Early epochs", color = "tomato")
    ax.hist(mid, bins = bins, alpha = 0.5, label = "Mid epochs", color ="orange")
    ax.hist(late, bins = bins, alpha = 0.5, label = "Late epochs",  color  = "steelblue")

    ax.axvline(np.mean(early), color="tomato", linestyle = "--", alpha = 0.8, label = f"Early mean: {np.mean(early):.1f}")
    ax.axvline(np.mean(late), color="steelblue", linestyle = "--", alpha = 0.8, label = f"Late mean: {np.mean(late):.1f}")

    ax.set_xlabel("NFE per forward pass", fontsize = 11)
    ax.set_ylabel("Count (epochs)", fontsize = 11)
    
    ax.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        default = "nfe_histogram.png"
        plt.savefig(default, dpi=150, bbox_inches="tight")
        print(f"  Saved: {default}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description = "NFE Analysis")
    parser.add_argument("--logs", nargs = "+", type = str, required = True,
                        help = "Paths to JSON log files from training")
    parser.add_argument("--labels", nargs = "+", type = str, default = None,
                        help = "Display labels for each log file")
    parser.add_argument("--save_dir", type = str, default = "analysis/plots")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents = True, exist_ok = True)

    labels = args.labels or [Path(p).stem for p in args.logs]

    plot_nfe_over_training(
        args.logs, labels,
        save_path = f"{args.save_dir}/nfe_over_training.png",
    )

    if len(args.logs) >= 1:
        plot_nfe_histogram(
            args.logs[0],
            save_path = f"{args.save_dir}/nfe_histogram.png",
        )


if __name__ == "__main__":
    main()