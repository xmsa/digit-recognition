import cv2
import numpy as np
import pandas as pd
from torchvision import transforms


def encode_image_to_bytes(image: np.ndarray) -> bytes:
    """
    Encode a NumPy image array into PNG byte format.

    Args:
        image (np.ndarray): Image array in BGR or grayscale format.

    Returns:
        bytes: PNG encoded image bytes.

    Raises:
        ValueError: If encoding fails.
    """
    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded_image.tobytes()


def read_grayscale_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes to a grayscale NumPy array using OpenCV.

    Args:
        image_bytes (bytes): Raw image bytes.

    Returns:
        np.ndarray: Grayscale image array.

    Raises:
        ValueError: If decoding fails or image is invalid.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image or failed to decode.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def detect_characters(image):
    """
    Detect characters in the input image.

    Note:
        In this version, character detection uses stored data from a CSV file
        located at "data/Sample/digit_lable.csv". The CSV data is loaded,
        indexed by 'label' and 'color', and the values returned.

        In future versions, this function will be replaced by actual code
        that detects digits directly from the image.

    Args:
        image: Input image (not used in this version).

    Returns:
        numpy.ndarray: Values from the CSV file representing character data.
    """
    info_img = pd.read_csv("data/Sample/digit_lable.csv")
    info_img.set_index(["label", "color"], inplace=True)
    return info_img.values


def draw_predictions(
    image: np.ndarray,
    boxes: list,
    predictions: list,
    color=(0, 0, 255),
    thickness=2,
    font_scale=0.5,
    box_format="xywh",
) -> np.ndarray:
    """
    Draw bounding boxes and prediction labels on an image.

    Args:
        image (np.ndarray): Input image array (BGR).
        boxes (list): List of bounding boxes. Each box is either
            (x, y, width, height) if box_format='xywh' or
            (x1, y1, x2, y2) if box_format='xyxy'.
        predictions (list): List of prediction strings corresponding to boxes.
        color (tuple): Box and text background color in BGR (default red).
        thickness (int): Thickness of box lines and text.
        font_scale (float): Font scale for text.
        box_format (str): Format of bounding boxes: 'xywh' or 'xyxy'.

    Returns:
        np.ndarray: Image with drawn boxes and labels.
    """
    img = image.copy()
    for box, pred in zip(boxes, predictions):
        if box_format == "xywh":
            x, y, w, h = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
        elif box_format == "xyxy":
            x1, y1, x2, y2 = map(int, box)
        else:
            raise ValueError("box_format must be either 'xyxy' or 'xywh'")

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Get text size for label background
        (text_width, text_height), baseline = cv2.getTextSize(
            pred, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Calculate coordinates for label background rectangle
        rect_x1 = x1
        rect_y1 = max(y1 - text_height - baseline - 6, 0)  # Padding above box
        rect_x2 = x1 + text_width + 6
        rect_y2 = y1

        # Draw filled rectangle for text background
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), color, cv2.FILLED)

        # Write prediction text in white color inside the rectangle
        text_pos = (rect_x1 + 3, rect_y2 - baseline - 3)
        cv2.putText(
            img,
            pred,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return img


def crop_image(image: np.ndarray, boxes: tuple) -> np.ndarray:
    """
    Crop a region from the image based on bounding box parameters.

    Args:
        image (np.ndarray): Input image array.
        boxes (tuple): Bounding box as (bbox_x, bbox_y, bbox_width, bbox_height).

    Returns:
        np.ndarray: Cropped image region.
    """
    bbox_x, bbox_y, bbox_width, bbox_height = boxes
    left = bbox_x
    upper = bbox_y
    right = bbox_x + bbox_width
    lower = bbox_y + bbox_height

    cropped_img = image[upper:lower, left:right]
    return cropped_img


def enhance_image_contrast_background_white(img: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast and invert colors to have white background.

    Args:
        img (np.ndarray): Grayscale input image.

    Returns:
        np.ndarray: Enhanced binary image with white background.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    cl1 = 255 - cl1  # invert colors

    _, thresh = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Return blank white image if no foreground found
        return 255 * np.ones((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y : y + h, x : x + w]

    return cropped


def make_square_with_padding(img: np.ndarray, pad_size=6) -> np.ndarray:
    """
    Pad image to make it square and add padding borders.

    Args:
        img (np.ndarray): Input grayscale image.
        pad_size (int): Number of pixels to pad on each side.

    Returns:
        np.ndarray: Square padded image.
    """
    h, w = img.shape
    size = max(h, w)
    square_img = np.zeros((size, size), dtype=img.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square_img[y_offset : y_offset + h, x_offset : x_offset + w] = img

    padded_img = cv2.copyMakeBorder(
        square_img,
        top=pad_size,
        bottom=pad_size,
        left=pad_size,
        right=pad_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return padded_img


def resize_image(squared: np.ndarray, final_size=28) -> np.ndarray:
    """
    Resize image to a fixed size.

    Args:
        squared (np.ndarray): Input image to resize.
        final_size (int): Desired output size (width and height).

    Returns:
        np.ndarray: Resized image.
    """
    resized = cv2.resize(
        squared, (final_size, final_size), interpolation=cv2.INTER_AREA
    )
    return resized


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def preocess_image(image: np.ndarray, box: tuple, flat=False) -> np.ndarray:
    """
    Crop, enhance, pad, resize, and transform an image region to tensor.

    Args:
        image (np.ndarray): Input grayscale image.
        box (tuple): Bounding box (bbox_x, bbox_y, bbox_width, bbox_height).
        flat (bool): If True, flatten the tensor to 1D.

    Returns:
        np.ndarray: Processed image tensor as numpy array.
    """
    img_crop = crop_image(image, box)
    process_img = enhance_image_contrast_background_white(img_crop)
    process_img = make_square_with_padding(process_img, pad_size=10)
    process_img = resize_image(process_img, final_size=28)

    if len(process_img.shape) == 2:
        process_img = np.expand_dims(process_img, axis=2)

    if process_img.dtype != np.uint8:
        process_img = (process_img * 255).astype(np.uint8)

    tensor_process_img = transform(process_img)
    if flat:
        tensor_process_img = tensor_process_img.view(tensor_process_img.size(0), -1)

    return tensor_process_img.numpy()
