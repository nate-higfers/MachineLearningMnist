"""Starter entrypoint for students to edit and experiment with MNIST."""

import torch

from mnist import test, train


def main() -> None:
    # Students can edit this section to customize their workflow.
    print(f"Using torch version: {torch.__version__}")
    train_images = train(root="data")
    test_images = test(root="data")
    print(f"Loaded {len(train_images)} train image tensors and {len(test_images)} test image tensors.")


if __name__ == "__main__":
    main()
