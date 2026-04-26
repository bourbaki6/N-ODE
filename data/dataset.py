#--- MNIST as the primary dataset, but to test for complexity the FashionMNIST dataset---# 

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def dataloaders(dataset="mnist", batch_size = 128):
    transform = transforms.ToTensor()

    if dataset == "mnist":
        train = datasets.MNIST(
            root = "./data",
            train = True,
            download = True,
            transform = transform
        )

        test = datasets.MNIST(
            root = "./data",
            train = False,
            download = True,
            transform = transform
        )

    elif dataset == "fashion":
        train = datasets.FashionMNIST(
            root = "./data",
            train = True,
            download = True,
            transform = transform
        )

        test = datasets.FashionMNIST(
            root = "./data",
            train = False,
            download = True,
            transform = transform
        )

    else:
        raise ValueError("\n dataset not matching")

    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader


# --- test run--- #
if __name__ == "__main__":
    train_loader, test_loader = dataloaders(dataset = "mnist", batch_size = 128)

    print("\n Number of training batches:", len(train_loader))
    print("\n Number of test batches:", len(test_loader))

    for images, labels in train_loader:
        print("\nImage batch shape:", images.shape)
        print("\n Label batch shape:", labels.shape)
        break