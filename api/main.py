import base64
import io

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from imageProcessing.model import image_processing, load_model
from imageProcessing.utils import (
    draw_predictions,
    encode_image_to_bytes,
    read_grayscale_from_bytes,
)

app = FastAPI()
models = load_model()

# Mount static files directory for serving CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Render the homepage template.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        HTMLResponse: Rendered HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(
    file: UploadFile = File(...), model: str = Form("SVM"), request: Request = None
):
    """
    Endpoint to receive an uploaded image file and predict characters in it using the specified model.

    Args:
        file (UploadFile): The uploaded image file.
        model (str): The model name to use for prediction (default "SVM").
        request (Request, optional): The incoming HTTP request, used for headers.

    Returns:
        JSONResponse or StreamingResponse: Returns a JSON with predictions and base64 image string,
        or an image stream if requested by the client.
    """

    image_bytes = await file.read()
    try:
        # Decode image bytes to grayscale OpenCV image
        image = read_grayscale_from_bytes(image_bytes)
    except Exception as e:
        # Return error message in English
        return JSONResponse(
            content={"error": f"Error reading the image: {str(e)}"}, status_code=400
        )

    # Run image processing and prediction
    predictions, boxes = image_processing(models[model.upper()], image)

    # Draw prediction boxes and labels on a copy of the original image
    processed_image = draw_predictions(image.copy(), boxes, predictions)

    # Concatenate all predicted characters as a single string
    full_text = "".join(predictions)

    # Encode the processed image to PNG bytes
    image_bytes = encode_image_to_bytes(processed_image)
    buffered = io.BytesIO(image_bytes)
    buffered.seek(0)

    # Inspect headers to decide response type
    user_agent = request.headers.get("user-agent", "").lower() if request else ""
    accept = request.headers.get("accept", "").lower() if request else ""

    # If request from curl or client wants image, return image stream
    if "curl" in user_agent or "image" in accept:
        return StreamingResponse(buffered, media_type="image/png")

    # Otherwise, return JSON response with base64 image and predictions
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return JSONResponse(
        content={
            "processed_image": img_str,
            "predicted_characters": predictions,
            "full_prediction": full_text,
            "model_used": model,
        }
    )
