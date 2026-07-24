"""
Image Prediction API
Predicts handwritten digits (0-9) from an uploaded image.
"""
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import joblib
import io
import os

app = Flask(__name__)

# Load the trained model once at startup
model = joblib.load("model.pkl")


def preprocess_image(image_bytes):
    """Convert uploaded image into the 8x8 grayscale format the model expects."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((8, 8))
    arr = np.array(img).astype(float)
    arr = (arr / 255.0) * 16  # scale to match sklearn digits (0-16 range)
    arr = 16 - arr  # invert so dark pen strokes = high value, like training data
    return arr.reshape(1, -1)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Image Prediction API is running.",
        "usage": "POST an image file to /predict with form field name 'image'"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    file = request.files["image"]
    try:
        image_bytes = file.read()
        features = preprocess_image(image_bytes)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
