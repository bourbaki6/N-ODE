
import yaml
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from models.classifier import NeuralODEClassifier
from baseline.resnet import ResNetBaseline
from data.dataset import get_dataloaders, get_class_names, get_input_dim
from utils import get_device, load_checkpoint, print_model_summary, measure_nfe


def evaluate_full(model, test_loader, device, class_names):

    model.eval()

    all_preds = []
    all_labels = []
    all_nfe = []

    with torch.no_grad():

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            log_probs  = model(images)
            preds = log_probs.argmax(dim = 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if hasattr(model, "nfe"):
                all_nfe.append(model.nfe)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    report = classification_report(
        all_labels, all_preds,
        target_names = class_names,
        digits = 4,
    )

    conf_mat = confusion_matrix(all_labels, all_preds)
    avg_nfe = float(np.mean(all_nfe)) if all_nfe else None

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_mat,
        "avg_nfe": avg_nfe,
    }

def plot_confusion_matrix(conf_mat, class_names, save_path):

    norm = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims = True)

    fig, ax = plt.subplots(figsize = (9, 7))
    im = ax.imshow(norm, interpolation = "nearest", cmap = "Blues", vmin = 0, vmax =1)
    fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation = 45, ha = "right", fontsize = 8)
    ax.set_yticklabels(class_names, fontsize = 8)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{norm[i, j]:.2f}",
                ha="center", va="center", fontsize=6,
                color="white" if norm[i, j] > thresh else "black",
            )

    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Normalised Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--config", type = str, default = "configs/mnist.yaml")
    parser.add_argument("--solver",  type = str, default = None,
        help = "Override solver for inference (euler|rk4|rk45|adjoint)",
    )
    parser.add_argument("--num_steps",  type = int, default = None)
    parser.add_argument("--model",     type = str, default="ode",
        choices = ["ode", "resnet"],
    )
    parser.add_argument("--num_blocks", type=int, default=6,
                        help="ResNet depth (only used with --model resnet)")
    parser.add_argument("--save_dir",   type=str, default="outputs/eval",
                        help="Directory for confusion matrix figure")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(args.save_dir).mkdir(parents = True, exist_ok = True)
    device = get_device()

    m_cfg = cfg["model"]
    dataset_name = cfg["dataset"]["name"]

    input_dim = get_input_dim(dataset_name)

    if args.model == "resnet":
        model = ResNetBaseline(
            hidden_dim = m_cfg["hidden_dim"],
            num_blocks = args.num_blocks,
            num_classes = m_cfg["num_classes"],
            input_dim = input_dim,          
        )
    else:
        model = NeuralODEClassifier(
            hidden_dim = m_cfg["hidden_dim"],
            num_classes = m_cfg["num_classes"],
            solver = m_cfg["solver"],
            num_steps = m_cfg["num_steps"],
            input_dim = input_dim,
        )

    load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)

    if args.solver and hasattr(model, "set_solver"):
        print(f"\nOverriding solver: {m_cfg['solver']} ->  {args.solver}"
              + (f" (num_steps={args.num_steps})" if args.num_steps else ""))
        model.set_solver(args.solver, num_steps=args.num_steps)

    print_model_summary(model, model.__class__.__name__)

    _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=cfg["dataset"]["batch_size"],
    )
    class_names = get_class_names(dataset_name)

    print("\nRunning evaluation…")
    results = evaluate_full(model, test_loader, device, class_names)

    print(f"\nTest Accuracy : {results['accuracy'] * 100:.2f}%")
    if results["avg_nfe"] is not None:
        print(f"Average NFE   : {results['avg_nfe']:.1f}")

    print(f"\n{results['report']}")

    cm_path = str(Path(args.save_dir) / "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], class_names, cm_path)


if __name__ == "__main__":
    main()
