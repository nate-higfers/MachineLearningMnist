"""Starter entrypoint for students to edit and experiment with MNIST."""

import torch

from mnist import test, train


def main() -> None:
    # Students can edit this section to customize their workflow.
    print(f"Using torch version: {torch.__version__}")
    train_samples = train(root="data")
    test_samples = test(root="data")

    # Return format note:
    # - train_samples/test_samples are `list[tuple[torch.Tensor, int]]`
    # - each item is `(image_tensor, label)`
    # - image tensor shape is [1, 28, 28], pixel values are [0.0, 1.0]
    first_image, first_label = train_samples[0]
    print(
        f"Loaded {len(train_samples)} train samples and {len(test_samples)} test samples. "
        f"First train label: {first_label}, image shape: {tuple(first_image.shape)}"
    )


if __name__ == "__main__":
    main()
