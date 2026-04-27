
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from models import NeuralODEClassifier
from baseline import ResNetBaseline
from data import get_dataloaders, get_class_names
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
            preds = log_probs.argmax(dim=1)

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

    avg_nfe = np.mean(all_nfe) if all_nfe else None

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_mat,
        "avg_nfe": avg_nfe,
    }


def main():
    parser = argparse.ArgumentParser(description = "Evaluate Neural ODE Classifier")
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--config", type = str, default = "configs/mnist.yaml")
    parser.add_argument("--solver", type = str, default = None,
                        help = "Override solver for inference")
    parser.add_argument("--num_steps", type = int, default = None)
    parser.add_argument("--model", type = str, default = "ode")
    parser.add_argument("--num_blocks", type = int, default = 6)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    m_cfg = cfg["model"]
    if args.model == "resnet":
        model = ResNetBaseline(
            hidden_dim = m_cfg["hidden_dim"],
            num_blocks = args.num_blocks,
            num_classes = m_cfg["num_classes"],
        )
    else:
        model = NeuralODEClassifier(
            hidden_dim = m_cfg["hidden_dim"],
            num_classes = m_cfg["num_classes"],
            solver = m_cfg["solver"],
            num_steps = m_cfg["num_steps"],
        )

    load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)

    if args.solver and hasattr(model, "set_solver"):
        print(f"\nOverriding solver: {m_cfg['solver']} -> {args.solver}")
        model.set_solver(args.solver, num_steps=args.num_steps)

    print_model_summary(model)


    _, test_loader = get_dataloaders(
        dataset_name = cfg["dataset"]["name"],
        batch_size = cfg["dataset"]["batch_size"],
    )
    class_names = get_class_names(cfg["dataset"]["name"])


    print("\nRunning evaluation")
    results = evaluate_full(model, test_loader, device, class_names)

    print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
    if results["avg_nfe"] is not None:
        print(f"Average NFE:   {results['avg_nfe']:.1f}")
    print(f"\n{results['report']}")

    print("Confusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()