from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

data_dir = Path("./data")
sample_data_dir = data_dir / "Sample"

data_dir.mkdir(exist_ok=True)
sample_data_dir.mkdir(exist_ok=True)


def save_test_image() -> None:
    """
    Save one example MNIST test image per digit (0-9) as PNG files.

    Downloads the MNIST test dataset if not available, extracts one image
    for each digit label, converts it to a PIL grayscale image, and saves
    it under `./data/Sample/{label}.png`.
    """
    # Transform without normalization for saving raw pixel values
    transform_no_norm = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform_no_norm
    )

    saved_labels = set()
    for idx in range(len(mnist_dataset)):
        image_tensor, label = mnist_dataset[idx]

        if label not in saved_labels:
            saved_labels.add(label)
            # Convert tensor to uint8 numpy array scaled [0,255]
            image_np = (image_tensor.squeeze().numpy() * 255).astype("uint8")
            img = Image.fromarray(image_np)
            img.save(sample_data_dir / f"{label}.png")

        # Stop once all 10 digits are saved
        if len(saved_labels) == 10:
            break


def load_img(filepath: Path) -> np.ndarray:
    """
    Load an image file as a 28x28 grayscale numpy float32 array.

    Args:
        filepath (Path): Path to the image file.

    Returns:
        np.ndarray: Grayscale image resized to 28x28, dtype float32.
    """
    img = Image.open(filepath).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_np = np.array(img).astype(np.float32)
    return img_np
