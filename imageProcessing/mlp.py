from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model for MNIST digit classification.

    Architecture:
        - Fully connected layer with 512 units + BatchNorm + ReLU + Dropout
        - Fully connected layer with 256 units + BatchNorm + ReLU + Dropout
        - Output layer with 10 units (number of classes)

    Methods:
        forward(x): Defines the forward pass.
        predict(x): Predicts class labels for input tensor or numpy array.
    """

    def __init__(self) -> None:
        """
        Initializes the MNIST_MLP model layers.
        """
        super(MNIST_MLP, self).__init__()

        # First fully connected layer
        self.fc1: nn.Linear = nn.Linear(28 * 28, 512)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(512)
        self.drop1: nn.Dropout = nn.Dropout(0.3)

        # Second fully connected layer
        self.fc2: nn.Linear = nn.Linear(512, 256)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(256)
        self.drop2: nn.Dropout = nn.Dropout(0.3)

        # Output layer
        self.fc3: nn.Linear = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, 10).
        """
        # Flatten input images to vectors
        x = x.view(-1, 28 * 28)

        # Pass through first fully connected layer, batch norm, ReLU and dropout
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))

        # Pass through second fully connected layer, batch norm, ReLU and dropout
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        # Output layer without activation (logits)
        x = self.fc3(x)

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
