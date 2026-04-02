"""Utilities for downloading MNIST and working with image tensors."""

from pathlib import Path

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


def _dataset(root: str, split: str) -> MNIST:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    return MNIST(root=root, train=(split == "train"), download=True, transform=ToTensor())


def train(root: str = "data") -> list[torch.Tensor]:
    """Return a list of 10 image tensors from the MNIST training split."""
    dataset = _dataset(root=root, split="train")
    return [dataset[idx][0] for idx in range(10)]


def test(root: str = "data") -> list[torch.Tensor]:
    """Return a list of 10 image tensors from the MNIST test split."""
    dataset = _dataset(root=root, split="test")
    return [dataset[idx][0] for idx in range(10)]


def generate_all_images(root: str = "data", split: str = "train") -> Path:
    """Download MNIST and export every image in the selected split."""
    dataset = _dataset(root=root, split=split)
    output_dir = Path(root) / "images" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_tensor, label) in enumerate(dataset):
        filename = output_dir / f"{idx:05d}_label-{label}.png"
        save_image(image_tensor, filename)

    return output_dir


if __name__ == "__main__":
    train_dir = generate_all_images(split="train")
    test_dir = generate_all_images(split="test")
    print(f"Saved train images to: {train_dir}")
    print(f"Saved test images to: {test_dir}")
