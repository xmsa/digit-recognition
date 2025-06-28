#!/bin/bash

MODEL=${1:-CNN} 
MODEL_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')

FILE_PATH="data/Sample/digit.jpg"
OUTPUT="output.png"
URL="http://localhost:8001/predict"

echo "Using model: $MODEL_UPPER"

curl -X POST "$URL" \
  -F "file=@${FILE_PATH}" \
  -F "model=${MODEL_UPPER}" \
  -o "$OUTPUT"

echo "Response saved to $OUTPUT"
