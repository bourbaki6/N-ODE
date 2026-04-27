
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path


def load_log(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def plot_nfe_over_training(log_paths: list, labels: list, save_path: str = None):
    
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    fig.suptitle("\n Neural ODE: NFE Dynamics Over Training", fontsize = 13, fontweight = "bold")

    colors = plt.cm.tab10.colors

    ax_nfe = axes[0]
    ax_acc = axes[1]

    for i, (path, label) in enumerate(zip(log_paths, labels)):
        records = load_log(path)
        epochs = [r["epoch"] for r in records]
        color = colors[i % len(colors)]

        if any("nfe" in r for r in records):
            nfes = [r.get("nfe", None) for r in records]
            nfes_clean = [n for n in nfes if n is not None]
            epochs_nfe = [e for e, n in zip(epochs, nfes) if n is not None]

            ax_nfe.plot(
                epochs_nfe, nfes_clean,
                color = color, linewidth = 2, label = label,
                marker = "o", markersize = 3,
            )

            if len(nfes_clean) >= 5:
                window = 5
                trend = np.convolve(nfes_clean, np.ones(window)/window, mode = "valid")
                trend_epochs = epochs_nfe[window-1:]
                ax_nfe.plot(
                    trend_epochs, trend,
                    color = color, linewidth = 1.5, linestyle = "--", alpha = 0.6,
                )

        if any("test_acc" in r for r in records):
            accs = [r.get("test_acc", 0) * 100 for r in records]
            ax_acc.plot(
                epochs, accs,
                color = color, linewidth = 2, label = label,
            )

    ax_nfe.set_xlabel("\n Epoch", fontsize = 11)
    ax_nfe.set_ylabel("\n Avg NFE per forward pass", fontsize = 11)
    ax_nfe.set_title("\n NFE over Training\n(decreasing = simpler dynamics = better)", fontsize=10)
    ax_nfe.legend()
    ax_nfe.grid(True, alpha = 0.3)
    ax_nfe.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax_nfe.annotate(
        " Lower NFE = learned smoother\nvector field = implicit regularisation",
        xy =(0.97, 0.97), xycoords = "axes fraction",
        ha = "right", va = "top", fontsize = 8,
        bbox = dict(boxstyle = "round,pad = 0.3", facecolor = "lightyellow", alpha = 0.8),
    )

    ax_acc.set_xlabel("Epoch", fontsize = 11)
    ax_acc.set_ylabel("Test Accuracy (%)", fontsize = 11)
    ax_acc.set_title("Test Accuracy over Training", fontsize = 10)
    ax_acc.legend()
    ax_acc.grid(True, alpha  = 0.3)
    ax_acc.set_ylim(bottom =  85)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches = "tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_nfe_histogram(log_path: str, save_path: str = None):
    
    records = load_log(log_path)
    nfes = [r.get("nfe") for r in records if "nfe" in r]
    epochs = [r["epoch"] for r in records if "nfe" in r]

    if not nfes:
        print("No NFE data found in log. Make sure you trained with rk45 solver.")
        return

    n = len(nfes)
    early = nfes[:n//3]
    mid = nfes[n//3 : 2*n//3]
    late = nfes[2*n//3:]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("NFE Distribution: Early vs Late Training", fontsize=13)

    bins = np.linspace(min(nfes)-1, max(nfes)+1, 20)
    ax.hist(early, bins = bins, alpha = 0.5, label = "Early epochs", color = "tomato")
    ax.hist(mid, bins = bins, alpha = 0.5, label = "Mid epochs", color ="orange")
    ax.hist(late, bins = bins, alpha = 0.5, label = "Late epochs",  color  = "steelblue")

    ax.axvline(np.mean(early), color="tomato", linestyle = "--", alpha = 0.8, label = f"Early mean: {np.mean(early):.1f}")
    ax.axvline(np.mean(late), color="steelblue", linestyle = "--", alpha = 0.8, label = f"Late mean: {np.mean(late):.1f}")

    ax.set_xlabel("NFE per forward pass", fontsize = 11)
    ax.set_ylabel("Count (epochs)", fontsize = 11)
    ax.legend()
    ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches = "tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


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