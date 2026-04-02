"""Starter entrypoint for students to edit and experiment with MNIST."""

import torch

from mnist import test, train


def main() -> None:
    # Students can edit this section to customize their workflow.
    print(f"Using torch version: {torch.__version__}")
    train_images = train(root="data")
    test_images = test(root="data")

    # Return format note:
    # - train_images/test_images are `list[torch.Tensor]`
    # - each tensor is a single MNIST image with shape [1, 28, 28]
    # - pixel values are normalized to [0.0, 1.0] by ToTensor
    print(f"Loaded {len(train_images)} train image tensors and {len(test_images)} test image tensors.")


if __name__ == "__main__":
    main()
