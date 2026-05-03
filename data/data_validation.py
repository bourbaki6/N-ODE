
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from data.dataset import get_dataloaders, get_class_names, DATASET_STATS

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

def ok(msg):   print(f" {GREEN}ok{RESET}  {msg}")
def fail(msg): print(f" {RED}no{RESET}  {msg}")
def warn(msg): print(f"{YELLOW}warn{RESET}  {msg}")
def info(msg): print(f" {BLUE}info{RESET}  {msg}")



def check_download(dataset_name: str) -> tuple:
    
    print(f"\n{BOLD}[1] Download / Cache{RESET}")
    try:
        t0 = time.time()
        train_loader, test_loader = get_dataloaders(
            dataset_name = dataset_name,
            batch_size = 128,
            num_workers = 0,     
        )
        elapsed = time.time() - t0
        ok(f"DataLoaders constructed in {elapsed:.1f}s")
        ok(f"Train batches : {len(train_loader)}")
        ok(f"Test  batches : {len(test_loader)}")
        return train_loader, test_loader
    except Exception as e:
        fail(f"Failed to build DataLoaders: {e}")
        raise


def check_shapes(loader, split: str):

    print(f"\n{BOLD}[2] Batch Shape & Dtype — {split}{RESET}")
    images, labels = next(iter(loader))

    if images.ndim == 4 and images.shape[1] == 1 and images.shape[2:] == torch.Size([28, 28]):
        ok(f"images shape : {tuple(images.shape)}  ok [batch, 1, 28, 28]")
    else:
        fail(f"images shape : {tuple(images.shape)}  expected [batch, 1, 28, 28]")

    if labels.ndim == 1:
        ok(f"labels shape : {tuple(labels.shape)}  ok [batch]")
    else:
        fail(f"labels shape : {tuple(labels.shape)}  expected [batch]")

    if images.dtype == torch.float32:
        ok(f"images dtype : {images.dtype}")
    else:
        fail(f"images dtype : {images.dtype}  expected torch.float32")

    if labels.dtype == torch.int64:
        ok(f"labels dtype : {labels.dtype}")
    else:
        warn(f"labels dtype : {labels.dtype}  expected torch.int64 — may cause issues with NLLLoss")

    return images, labels

def check_normalisation(images: torch.Tensor, dataset_name: str):
    
    print(f"\n{BOLD}[3] Normalisation{RESET}")

    mean_val = images.mean().item()
    std_val = images.std().item()
    min_val = images.min().item()
    max_val = images.max().item()

    info(f"Pixel mean : {mean_val:+.4f}  (target ≈ 0.0)")
    info(f"Pixel std : {std_val:.4f}   (target ≈ 1.0)")
    info(f"Pixel range : [{min_val:.3f}, {max_val:.3f}]")


    if abs(mean_val) < 0.3:
        ok("Mean is near zero — normalisation applied correctly")
    else:
        fail(f"Mean {mean_val:.4f} is far from zero — check normalisation")

    if 0.5 < std_val < 2.0:
        ok("Std is near 1.0 — normalisation applied correctly")
    else:
        warn(f"Std {std_val:.4f} is unusual — verify normalisation parameters")

    if min_val < 0:
        ok("Negative pixel values present — ToTensor + Normalize confirmed")
    else:
        fail("No negative values — Normalize may not have been applied")

    stats = DATASET_STATS[dataset_name]
    info(f"Expected stats -> mean={stats['mean'][0]}, std={stats['std'][0]}")


def check_label_distribution(train_loader, test_loader, dataset_name: str):
    
    print(f"\n{BOLD}[4] Label Distribution{RESET}")
    class_names = get_class_names(dataset_name)

    for split, loader in [("Train", train_loader), ("Test", test_loader)]:
        counts = torch.zeros(10, dtype = torch.long)
        for _, labels in loader:
            for c in range(10):
                counts[c] += (labels == c).sum()

        total = counts.sum().item()
        min_count = counts.min().item()
        max_count = counts.max().item()
        imbalance_ratio = max_count / min_count

        info(f"{split} set — total: {total:,}, per-class range: [{min_count}, {max_count}]")

        if imbalance_ratio < 1.2:
            ok(f"{split}: balanced (imbalance ratio {imbalance_ratio:.2f})")
        elif imbalance_ratio < 2.0:
            warn(f"{split}: mild imbalance (ratio {imbalance_ratio:.2f})")
        else:
            fail(f"{split}: severe imbalance (ratio {imbalance_ratio:.2f}) — investigate")

        print(f"\n  {'Class':<20} {'Count':>8}  {'Bar'}")
        for i, (name, count) in enumerate(zip(class_names, counts.tolist())):
            bar_len = int(count / max_count * 30)
            bar = "█" * bar_len
            pct = count / total * 100
            print(f"  {name:<20} {count:>8,}  {bar}  {pct:.1f}%")


def check_loader_speed(train_loader, n_batches: int = 20):
    
    print(f"\n{BOLD}[5] DataLoader Speed{RESET}")

    times = []
    t_start = time.perf_counter()
    for i, (images, labels) in enumerate(train_loader):
        if i >= n_batches:
            break
        t_end = time.perf_counter()
        if i > 0:  
            times.append((t_end - t_start) * 1000)
        t_start = t_end

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)

    info(f"Avg batch load time : {avg_ms:.1f} ± {std_ms:.1f} ms")
    info(f"Range : [{min_ms:.1f}, {max_ms:.1f}] ms")

    if avg_ms < 10:
        ok("Very fast — data likely fully cached in RAM")
    elif avg_ms < 50:
        ok("Acceptable speed for CPU training")
    elif avg_ms < 150:
        warn(f"Moderate speed ({avg_ms:.0f}ms/batch) — consider num_workers > 0")
    else:
        fail(f"Slow loading ({avg_ms:.0f}ms/batch) — will bottleneck GPU training. "
             f"Try num_workers=4 and pin_memory=True")


def check_visual(images: torch.Tensor, labels: torch.Tensor,
                 dataset_name: str, save_path: str = "data_sample.png"):
   
    print(f"\n{BOLD}[6] Visual Sanity Check{RESET}")
    class_names = get_class_names(dataset_name)

    stats = DATASET_STATS[dataset_name]
    mean = stats["mean"][0]
    std = stats["std"][0]

    display = images[:32].clone()
    display = display * std + mean        
    display = display.clamp(0, 1)           

    fig, axes = plt.subplots(4, 8, figsize = (14, 7))
    fig.suptitle(
        f"{dataset_name.upper()} — Sample Images (de-normalised for display)\n"
        f"If these look wrong (inverted, scrambled, blank), investigate immediately.",
        fontsize = 11
    )

    for idx, ax in enumerate(axes.flat):
        if idx >= len(display):
            ax.axis("off")
            continue
        img = display[idx].squeeze().numpy()
        ax.imshow(img, cmap = "gray", vmin= 0, vmax = 1)
        ax.set_title(class_names[labels[idx].item()], fontsize = 7)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi = 120, bbox_inches = "tight")
    plt.close()

    ok(f"Sample grid saved to: {save_path}")

def check_full_epoch(train_loader):
    
    print(f"\n{BOLD}[7] Full Epoch Iteration{RESET}")
    info("Iterating full training set")

    t0 = time.time()
    n_batches = 0
    n_samples = 0
    batch_sizes = []

    try:
        for images, labels in train_loader:
            n_batches  += 1
            n_samples  += images.size(0)
            batch_sizes.append(images.size(0))

            if torch.isnan(images).any():
                fail(f"NaN values found in batch {n_batches} — corrupt sample or bad normalisation")
                return
            if torch.isinf(images).any():
                fail(f"Inf values found in batch {n_batches} — check normalisation std (is it zero?)")
                return

    except Exception as e:
        fail(f"Loader raised exception at batch {n_batches}: {e}")
        return

    elapsed = time.time() - t0
    ok(f"Full epoch: {n_batches} batches, {n_samples:,} samples in {elapsed:.1f}s")
    ok(f"Avg throughput: {n_samples/elapsed:.0f} samples/sec")

    unique_sizes = set(batch_sizes)
    if len(unique_sizes) <= 2:   
        ok(f"Batch sizes consistent: {sorted(unique_sizes)}")
    else:
        warn(f"Inconsistent batch sizes: {sorted(unique_sizes)} — check drop_last setting")

    ok("\n No NaN or Inf values found across entire training set")


def check_model_compatibility(train_loader):
   
    print(f"\n{BOLD}[8] Model Compatibility (forward pass smoke test){RESET}")

    try:
        from models import NeuralODEClassifier
        model = NeuralODEClassifier(solver = "euler", num_steps = 2)
        model.eval()

        images, labels = next(iter(train_loader))

        with torch.no_grad():
            log_probs = model(images)

        expected_shape = (images.size(0), 10)
        if tuple(log_probs.shape) == expected_shape:
            ok(f"Output shape : {tuple(log_probs.shape)} ")
        else:
            fail(f"Output shape : {tuple(log_probs.shape)}  expected {expected_shape}")

        if (log_probs <= 0).all():
            ok("All outputs ≤ 0  ok  (valid log-probabilities)")
        else:
            fail("Some outputs > 0 — model is not returning log-probabilities")

        probs_sum = log_probs.exp().sum(dim = 1)
        if (probs_sum - 1.0).abs().max().item() < 1e-4:
            ok("Probabilities sum to 1.0 per sample  ok")
        else:
            fail(f"Probabilities don't sum to 1.0 — max deviation: {(probs_sum-1).abs().max():.6f}")

        criterion = torch.nn.NLLLoss()
        loss = criterion(log_probs, labels)
        if not torch.isnan(loss) and not torch.isinf(loss):
            ok(f"NLLLoss computable : {loss.item():.4f}  ok")
        else:
            fail(f"NLLLoss returned {loss.item()} — check label range and model output")

        ok(f"NFE after forward  : {model.nfe} (Euler, 2 steps -> expect 2)")

    except ImportError:
        warn("models/ not found — skipping model compatibility check")
        warn("Run this again after writing models/classifier.py")
    except Exception as e:
        fail(f"Forward pass failed: {e}")


def validate(dataset_name: str):

    print(f"Dataset Validation — {dataset_name.upper()}")

    train_loader, test_loader = check_download(dataset_name)

    train_images, train_labels = check_shapes(train_loader, "Train")
    check_shapes(test_loader, "Test")

    check_normalisation(train_images, dataset_name)
    check_label_distribution(train_loader, test_loader, dataset_name)
    check_loader_speed(train_loader)
    check_visual(train_images, train_labels, dataset_name,
                 save_path = f"data_sample_{dataset_name}.png")
    check_full_epoch(train_loader)
    check_model_compatibility(train_loader)

    print(f"\n{'='*60}")
    print(f" Validation complete for {dataset_name.upper()}")
    print(f" Open data_sample_{dataset_name}.png and visually verify")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description = "Validate dataset pipeline")
    parser.add_argument(
        "--dataset", type = str, default = "both",
        choices=["mnist", "fashion_mnist", "both"],
        help="Which dataset to validate"
    )
    args = parser.parse_args()

    datasets = (
        ["mnist", "fashion_mnist"] if args.dataset == "both"
        else [args.dataset]
    )

    for name in datasets:
        validate(name)


if __name__ == "__main__":
    main()