from typing import Any, Dict, List, Tuple

import joblib
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier

from imageProcessing.cnn import MNIST_CNN
from imageProcessing.mlp import MNIST_MLP
from imageProcessing.utils import detect_characters, preocess_image


def load_model() -> Dict[str, Any]:
    """
    Load pretrained machine learning models from disk.

    Returns:
        Dict[str, Any]: A dictionary mapping model names to loaded model objects.
    """

    def __load_model(model_name):
        if model_name == "CNN":
            model = MNIST_CNN()
            model.load_state_dict(torch.load("models/mnist_cnn.pt", weights_only=True))
            model.eval()
        elif model_name == "MLP":
            model = MNIST_MLP()
            model.load_state_dict(torch.load("models/mnist_mlp.pt", weights_only=True))
            model.eval()
        return model

    clf = {
        "SVM": joblib.load("models/svm_model.joblib"),
        "MLP": __load_model(model_name="MLP"),
        "CNN": __load_model(model_name="CNN"),
    }
    return clf


def image_processing(
    clf: Any, image: Image.Image
) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """
    Detect characters in the input image and predict their labels using the given classifier.

    Args:
        clf (Any): A trained classifier supporting .predict().
        image (PIL.Image.Image): The input image to process.

    Returns:
        Tuple[List[str], List[Tuple[int, int, int, int]]]:
            - List of predicted character labels as strings.
            - List of bounding boxes corresponding to detected characters.
    """
    boxes = detect_characters(image)
    predictions = predict_characters(clf, image, boxes)
    return predictions, boxes


def predict_characters(
    clf: Any, image: Image.Image, boxes: List[Tuple[int, int, int, int]]
) -> List[str]:
    """
    Predict character labels for detected bounding boxes in the image using the classifier.

    Args:
        clf (Any): A trained classifier, e.g., sklearn's SGDClassifier.
        image (PIL.Image.Image): The input image.
        boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x, y, width, height).

    Returns:
        List[str]: Predicted character labels as strings.
    """
    predict_list = []

    # Determine if classifier expects flattened input
    flat = False
    if isinstance(clf, SGDClassifier):
        flat = True

    for box in boxes:
        # Preprocess image region inside box, flatten if needed
        process_imgs = preocess_image(image, box, flat=flat)
        y_pred = clf.predict(process_imgs)[0]
        predict_list.append(str(y_pred))

    return predict_list
