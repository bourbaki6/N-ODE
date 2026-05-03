
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from models import NeuralODEClassifier
from baseline import ResNetBaseline
from data import get_dataloaders
from utils import (
    get_device, save_checkpoint, MetricsLogger,
    compute_accuracy, measure_nfe,
    print_model_summary, print_comparison_table,
)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        log_probs = model(images)             
        loss = criterion(log_probs, labels)  

        loss.backward()
    
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss  += loss.item() * images.size(0)
        predictions = log_probs.argmax(dim = 1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            batch_acc = (predictions == labels).float().mean().item()
            print(f" Batch {batch_idx+1}/{len(loader)} | "
                  f"loss: {loss.item():.4f} | acc: {batch_acc:.3f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
   
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            predictions = log_probs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(cfg: dict, model: nn.Module, device: torch.device):
    
    train_loader, test_loader = get_dataloaders(
        dataset_name = cfg["dataset"]["name"],
        data_dir = cfg["dataset"]["data_dir"],
        batch_size = cfg["dataset"]["batch_size"],
        num_workers = cfg["dataset"]["num_workers"],
        augment_train = cfg["dataset"].get("augment_train", False),
    )

    model = model.to(device)
    print_model_summary(model, model.__class__.__name__)

    t_cfg = cfg["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr = t_cfg["lr"],
        weight_decay = t_cfg.get("weight_decay", 1e-5),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_cfg["epochs"],
        eta_min=t_cfg.get("lr_min", 1e-5),
    )

    criterion = nn.NLLLoss()

    l_cfg = cfg["logging"]
    logger = MetricsLogger(log_dir = l_cfg["log_dir"], run_name = l_cfg["run_name"])

    print(f"\nStarting training for {t_cfg['epochs']} epochs on {device}.")

    best_test_acc = 0.0

    for epoch in range(1, t_cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{t_cfg['epochs']}")
        current_lr = scheduler.get_last_lr()[0]

    
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=t_cfg.get("grad_clip", 1.0),
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        nfe = None
        if hasattr(model, "nfe"):
            nfe = measure_nfe(model, test_loader, device, num_batches=5)

        log_kwargs = dict(
            train_loss = train_loss,
            test_loss = test_loss,
            train_acc = train_acc,
            test_acc = test_acc,
            lr = current_lr,
        )
        if nfe is not None:
            log_kwargs["nfe"] = nfe

        logger.log(epoch = epoch, **log_kwargs)

        scheduler.step()

        if epoch % l_cfg.get("save_every", 5) == 0:
            ckpt_path = f"{l_cfg['checkpoint_dir']}/{l_cfg['run_name']}_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, log_kwargs, ckpt_path)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_path = f"{l_cfg['checkpoint_dir']}/{l_cfg['run_name']}_best.pt"
            save_checkpoint(model, optimizer, epoch, log_kwargs, best_path)


    logger.save()
    print(f"Training complete. Best test accuracy: {best_test_acc*100:.2f}%")

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Neural ODE Classifier")
    parser.add_argument("--config", type = str, default = "configs/mnist.yaml")
    parser.add_argument("--solver", type = str, default = None,
                        help="Override solver: euler|rk4|rk45|adjoint")
    parser.add_argument("--num_steps", type = int, default = None,
                        help="Override num_steps for euler/rk4")
    parser.add_argument("--batch_size", type = int, default = None)
    parser.add_argument("--epochs", type = int,  default = None)
    parser.add_argument("--model", type = str, default = "ode",
                        help="'ode' for NeuralODE, 'resnet' for ResNet baseline")
    parser.add_argument("--num_blocks", type = int, default = 6,
                        help="ResNet blocks (only used with --model resnet)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.solver:
        cfg["model"]["solver"] = args.solver
    elif args.num_steps:
        cfg["model"]["num_steps"] = args.num_steps
    elif args.batch_size:
        cfg["dataset"]["batch_size"] = args.batch_size
    elif args.epochs:
        cfg["training"]["epochs"] = args.epochs

    device = get_device()

    m_cfg = cfg["model"]

    if args.model == "resnet":
        model = ResNetBaseline(
            hidden_dim=m_cfg["hidden_dim"],
            num_blocks=args.num_blocks,
            num_classes=m_cfg["num_classes"],
            input_dim=m_cfg["input_dim"],
        )
    
        cfg["logging"]["run_name"] += f"_resnet{args.num_blocks}"
    else:
        model = NeuralODEClassifier(
            hidden_dim = m_cfg["hidden_dim"],
            num_classes = m_cfg["num_classes"],
            solver = m_cfg["solver"],
            num_steps = m_cfg["num_steps"],
            input_dim = m_cfg["input_dim"],
        )
        
        cfg["logging"]["run_name"] += f"_{m_cfg['solver']}"

    train(cfg, model, device)


if __name__ == "__main__":
    main()