#--- MNIST as the primary dataset, but to test for complexity the FashionMNIST dataset---# 

"""
dataset.py
==========
DataLoader setup for MNIST and FashionMNIST.

DATASET DETAILS
---------------
MNIST:
    - 60,000 training, 10,000 test images
    - 28×28 grayscale, 10 digit classes (0–9)
    - Mean=0.1307, Std=0.3081 (precomputed over training set)
    - "Solved" benchmark — any reasonable model hits >98%
    - Use this to verify your implementation is correct

FashionMNIST:
    - Same format as MNIST (28×28, 10 classes, same split sizes)
    - Classes: T-shirt, Trouser, Pullover, Dress, Coat,
               Sandal, Shirt, Sneaker, Bag, Ankle boot
    - Significantly harder: state-of-the-art is ~96%, simple models ~88-91%
    - This is where Neural ODE vs ResNet comparisons become meaningful
    - Mean=0.2860, Std=0.3530

NORMALISATION
-------------
We normalise to zero-mean unit-variance using the training set statistics.
This is important for ODE stability:
    - If h(0) has very large values, the ODE derivative dh/dt can be large,
      requiring very small Δt for accurate integration → high NFE
    - Normalised inputs → smaller initial h(0) → smoother dynamics → lower NFE

DATA AUGMENTATION (OPTIONAL)
------------------------------
For FashionMNIST we can add horizontal flipping (RandomHorizontalFlip).
Note: horizontal flipping is valid for fashion items but NOT for MNIST digits
(a flipped "6" is not a "9" in the standard labelling scheme).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


# ── Dataset-specific normalisation statistics ─────────────────────────────────
DATASET_STATS = {
    "mnist": {
        "mean": (0.1307,),
        "std":  (0.3081,),
    },
    "fashion_mnist": {
        "mean": (0.2860,),
        "std":  (0.3530,),
    },
}


def get_dataloaders(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment_train: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and test DataLoaders for MNIST or FashionMNIST.

    Args:
        dataset_name (str):  'mnist' or 'fashion_mnist'.
        data_dir (str):      Directory to cache downloaded data.
        batch_size (int):    Batch size for both loaders.
                             Larger batches → more stable gradients but more memory.
                             128 is a good default for Neural ODE on CPU.
        num_workers (int):   Background workers for data loading.
                             Set to 0 if you get multiprocessing errors on Windows.
        augment_train (bool): If True, apply horizontal flip to FashionMNIST
                              training data. Has no effect on MNIST.

    Returns:
        (train_loader, test_loader): Configured PyTorch DataLoaders.

    Usage:
        train_loader, test_loader = get_dataloaders('fashion_mnist', batch_size=64)
        for images, labels in train_loader:
            # images: [batch, 1, 28, 28], float32, normalised
            # labels: [batch], int64, class indices 0–9
    """
    assert dataset_name in DATASET_STATS, (
        f"Unknown dataset '{dataset_name}'. "
        f"Choose from: {list(DATASET_STATS.keys())}"
    )

    stats = DATASET_STATS[dataset_name]
    mean, std = stats["mean"], stats["std"]

    # ── Training transforms ────────────────────────────────────────────────────
    train_transforms = [transforms.ToTensor()]

    if augment_train and dataset_name == "fashion_mnist":
        # Random horizontal flip for FashionMNIST only (valid for clothing)
        # Placed before normalisation so flipping acts on raw pixel values
        train_transforms.insert(0, transforms.RandomHorizontalFlip(p=0.5))

    # Normalise last: ToTensor already converts to [0,1], then we shift to
    # zero-mean unit-variance using the training set statistics
    train_transforms.append(transforms.Normalize(mean, std))
    train_transform = transforms.Compose(train_transforms)

    # ── Test transforms: no augmentation ──────────────────────────────────────
    # CRITICAL: Never apply random augmentation to test data.
    # Test set must be deterministic for reproducible evaluation.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Select dataset class ──────────────────────────────────────────────────
    DatasetClass = {
        "mnist":         datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
    }[dataset_name]

    # ── Download and instantiate ──────────────────────────────────────────────
    train_dataset = DatasetClass(
        root=data_dir, train=True,
        transform=train_transform, download=True,
    )
    test_dataset = DatasetClass(
        root=data_dir, train=False,
        transform=test_transform, download=True,
    )

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True,        # faster CPU→GPU transfer (no-op on CPU-only)
        drop_last=True,         # drop incomplete final batch (stabilises GroupNorm)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # can use larger batch for eval (no grad needed)
        shuffle=False,              # deterministic evaluation
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset: {dataset_name}")
    print(f"  Training: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Test:     {len(test_dataset):,}  samples, {len(test_loader):,} batches")
    print(f"  Batch size: {batch_size}")

    return train_loader, test_loader


def get_class_names(dataset_name: str) -> list:
    """Return human-readable class names for a dataset."""
    if dataset_name == "mnist":
        return [str(i) for i in range(10)]
    elif dataset_name == "fashion_mnist":
        return [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")