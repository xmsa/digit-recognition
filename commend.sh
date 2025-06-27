curl -X POST http://localhost:8001/predict \
  -F "file=@data/Sample/digit.jpg" \
  -F "model=SVM" \
  -o output.png

