#--- Extended dataset.py: supports grayscale (MNIST, FashionMNIST)
#   AND full RGB image datasets (CIFAR-10, CIFAR-100, STL-10)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

DATASET_STATS = {
    "mnist": {
        "mean":(0.1307,),
        "std": (0.3081,),
        "channels": 1,
        "image_size": 28,
        "num_classes": 10,
    },
    "fashion_mnist": {
        "mean":(0.2860,),
        "std": (0.3530,),
        "channels": 1,
        "image_size": 28,
        "num_classes": 10,
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":(0.2470, 0.2435, 0.2616),
        "channels": 3,
        "image_size": 32,
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "channels": 3,
        "image_size": 32,
        "num_classes": 100,
    },
    "stl10": {
        "mean":(0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "channels": 3,
        "image_size": 32,  
        "num_classes": 10,
    },
}

_DATASET_CLASS = {
    "mnist":  datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100":datasets.CIFAR100,
    "stl10": datasets.STL10,
}


_USES_SPLIT_ARG = {"stl10"}


def get_input_dim(dataset_name: str) -> int:
    
    info = DATASET_STATS[dataset_name]
    return info["channels"] * info["image_size"] * info["image_size"]


def get_num_classes(dataset_name: str) -> int:

    return DATASET_STATS[dataset_name]["num_classes"]


def _build_transforms(dataset_name: str, train: bool, augment_train: bool) -> transforms.Compose:
    info = DATASET_STATS[dataset_name]
    mean = info["mean"]
    std = info["std"]
    size = info["image_size"]
    is_rgb = info["channels"] == 3

    tfm_list = []

    if dataset_name == "stl10":
        tfm_list.append(transforms.Resize(size))

    if train and augment_train:
        if is_rgb:
    
            tfm_list.append(transforms.RandomCrop(size, padding = 4))
            tfm_list.append(transforms.RandomHorizontalFlip(p = 0.5))
        else:
    
            if dataset_name == "fashion_mnist":
                tfm_list.append(transforms.RandomHorizontalFlip(p = 0.5))

    tfm_list.append(transforms.ToTensor())

    tfm_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(tfm_list)


def _load_dataset(dataset_name: str,data_dir: str, train: bool, transform):
    cls = _DATASET_CLASS[dataset_name]

    if dataset_name in _USES_SPLIT_ARG:
        #--- STL-10 uses split='train' or split='test' ---#
        return cls(
            root = data_dir,
            split = "train" if train else "test",
            transform = transform,
            download = True,
        )
    else:
        return cls(
            root = data_dir,
            train = train,
            transform  = transform,
            download = True,
        )


def get_dataloaders(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment_train: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    
    assert dataset_name in DATASET_STATS, (
        f"Unknown dataset '{dataset_name}'. "
        f"Supported: {list(DATASET_STATS.keys())}"
    )

    train_transform = _build_transforms(dataset_name, train = True,  augment_train = augment_train)
    test_transform = _build_transforms(dataset_name, train = False, augment_train = False)

    train_dataset = _load_dataset(dataset_name, data_dir, train = True,  transform = train_transform)
    test_dataset = _load_dataset(dataset_name, data_dir, train = False, transform = test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True,    
        drop_last = True,     
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size * 2,  
        shuffle = False,             
        num_workers= num_workers,
        pin_memory = True,
    )

    info = DATASET_STATS[dataset_name]
    print(f"Dataset : {dataset_name}")
    print(f"Image size  : {info['channels']}x{info['image_size']}x{info['image_size']}  ->  input_dim={get_input_dim(dataset_name)}")
    print(f"Classes: {info['num_classes']}")
    print(f" Training: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f" Test: {len(test_dataset):,} samples,  {len(test_loader):,} batches")
    print(f" Batch size : {batch_size} (train) / {batch_size*2} (test)")
    print(f" Augment: {augment_train}")

    return train_loader, test_loader


def get_class_names(dataset_name: str) -> list:

    if dataset_name == "mnist":
        return [str(i) for i in range(10)]

    elif dataset_name == "fashion_mnist":
        return [
            "T-shirt/top", "Trouser",  "Pullover", "Dress", "Coat",
            "Sandal","Shirt","Sneaker",  "Bag", "Ankle boot",
        ]

    elif dataset_name in ("cifar10", "stl10"):
        #--- STL-10 = 10 class names as CIFAR-10 ---#
        return [
            "airplane", "automobile", "bird",  "cat", "deer",
            "dog", "frog","horse", "ship", "truck",
        ]

    elif dataset_name == "cifar100":
        #--- all 100 fine-grained CIFAR-100 classes in label order; from official CIFAR website---#
        return [
            "apple",          "aquarium_fish",  "baby",           "bear",
            "beaver",         "bed",            "bee",            "beetle",
            "bicycle",        "bottle",         "bowl",           "boy",
            "bridge",         "bus",            "butterfly",      "camel",
            "can",            "castle",         "caterpillar",    "cattle",
            "chair",          "chimpanzee",     "clock",          "cloud",
            "cockroach",      "couch",          "crab",           "crocodile",
            "cup",            "dinosaur",       "dolphin",        "elephant",
            "flatfish",       "forest",         "fox",            "girl",
            "hamster",        "house",          "kangaroo",       "keyboard",
            "lamp",           "lawn_mower",     "leopard",        "lion",
            "lizard",         "lobster",        "man",            "maple_tree",
            "motorcycle",     "mountain",       "mouse",          "mushroom",
            "oak_tree",       "orange",         "orchid",         "otter",
            "palm_tree",      "pear",           "pickup_truck",   "pine_tree",
            "plain",          "plate",          "poppy",          "porcupine",
            "possum",         "rabbit",         "raccoon",        "ray",
            "road",           "rocket",         "rose",           "sea",
            "seal",           "shark",          "shrew",          "skunk",
            "skyscraper",     "snail",          "snake",          "spider",
            "squirrel",       "streetcar",      "sunflower",      "sweet_pepper",
            "table",          "tank",           "telephone",      "television",
            "tiger",          "tractor",        "train",          "trout",
            "tulip",          "turtle",         "wardrobe",       "whale",
            "willow_tree",    "wolf",           "woman",          "worm",
        ]

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(DATASET_STATS.keys())}"
        )