from typing import List

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    processed_image: str  # Base64 string
    predicted_characters: List[str]
    full_prediction: str
