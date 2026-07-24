# Image Prediction API

A REST API that predicts handwritten digits (0-9) from an uploaded image, built with Flask and scikit-learn.

## How it works
- Trained an SVM classifier on the digits dataset (1,797 handwritten digit images)
- Achieves ~99% test accuracy
- Exposes a `/predict` endpoint that accepts an image and returns the predicted digit with confidence score

## Tech Stack
- Python, Flask
- scikit-learn (SVM classifier)
- Pillow (image preprocessing)
- Deployed on Railway

## API Usage

**Endpoint:** `POST /predict`

**Request:** multipart/form-data with key `image` and an image file

**Response:**
```json
{
  "prediction": 7,
  "confidence": 0.9842
}
```

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

## Author
Namani Shivani
