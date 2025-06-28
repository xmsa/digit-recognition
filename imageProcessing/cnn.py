from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST digit classification.

    Architecture:
        - Conv2d(1 -> 32, 3x3 kernel, padding=1) + ReLU + MaxPool(2x2)
        - Conv2d(32 -> 64, 3x3 kernel, padding=1) + ReLU + MaxPool(2x2)
        - Fully connected layer with 128 units + ReLU
        - Fully connected layer with 10 units (output classes)

    Methods:
        forward(x): Forward pass through the network.
        predict(x): Predicts class labels for input tensor or numpy array.
    """

    def __init__(self) -> None:
        """
        Initializes the MNIST_CNN model layers.
        """
        super(MNIST_CNN, self).__init__()

        # First convolutional layer: input 1 channel, output 32 channels
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )  # Output size: 28x28

        # Max pooling layer with kernel size 2x2 and stride 2
        self.pool: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Downsamples by factor 2

        # Second convolutional layer: input 32 channels, output 64 channels
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )  # Output size: 14x14

        # Fully connected layer 1: input flattened features, output 128 units
        self.fc1: nn.Linear = nn.Linear(in_features=64 * 7 * 7, out_features=128)

        # Fully connected layer 2: output layer with 10 units (number of classes)
        self.fc2: nn.Linear = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, 10).
        """
        # Apply first convolution + ReLU activation + max pooling
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [batch_size, 32, 14, 14]

        # Apply second convolution + ReLU activation + max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [batch_size, 64, 7, 7]

        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layer 1 with ReLU
        x = F.relu(self.fc1(x))

        # Fully connected layer 2 (output layer)
        x = self.fc2(x)

        return x

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predicts class labels for the input images.

        Args:
            x (Union[np.ndarray, torch.Tensor]):
                Input image batch as a NumPy array or PyTorch tensor.
                Expected shape: (batch_size, 1, 28, 28).

        Returns:
            np.ndarray: Predicted class indices with shape (batch_size,).
        """
        # Convert numpy array input to torch tensor if necessary
        if isinstance(x, np.ndarray):
            try:
                x = torch.tensor(x, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Failed to convert input array to torch tensor: {e}")

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a numpy array or a torch tensor.")

        # Forward pass to get raw logits
        logits = self.forward(x)

        # Get predicted class indices by taking argmax over output dimension 1
        preds = logits.argmax(dim=1).cpu().numpy()

        return preds
