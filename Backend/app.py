
import io
import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "hemoscan_cnn.keras"
CLASSES_PATH = MODEL_DIR / "classes.json"

IMG_SIZE = 224
MAX_UPLOAD_MB = 10

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

CORS(app)

model = None
classes = []
model_error = None
model_has_internal_rescaling = False


@app.after_request
def attach_private_network_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Accept, X-Requested-With, ngrok-skip-browser-warning"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


def _detect_rescaling_layer(keras_model) -> bool:
    for layer in getattr(keras_model, "layers", []):
        if layer.__class__.__name__ == "Rescaling":
            return True
        if hasattr(layer, "layers") and _detect_rescaling_layer(layer):
            return True
    return False


def load_model():
    global model, classes, model_error, model_has_internal_rescaling

    try:
        import tensorflow as tf

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        if not CLASSES_PATH.exists():
            raise FileNotFoundError(
                f"Class file not found: {CLASSES_PATH}"
            )

        loaded_model = tf.keras.models.load_model(MODEL_PATH)

        loaded_classes = json.loads(
            CLASSES_PATH.read_text(encoding="utf-8")
        )

        expected_classes = [
            "A+", "A-", "AB+", "AB-",
            "B+", "B-", "O+", "O-"
        ]

        if loaded_classes != expected_classes:
            raise ValueError(
                "classes.json does not match the trained model class order."
            )

        if loaded_model.output_shape[-1] != len(loaded_classes):
            raise ValueError(
                "Model output count does not match classes.json."
            )

        if tuple(loaded_model.input_shape[1:]) != (224, 224, 3):
            raise ValueError(
                f"Unexpected model input shape: {loaded_model.input_shape}"
            )

        model = loaded_model
        classes = loaded_classes
        model_has_internal_rescaling = _detect_rescaling_layer(loaded_model)
        model_error = None

    except Exception as exc:
        model = None
        classes = []
        model_has_internal_rescaling = False
        model_error = str(exc)
        print("Model loading failed:", model_error)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "error": f"Image must be smaller than {MAX_UPLOAD_MB} MB."
    }), 413


@app.get("/")
def home():
    return jsonify({
        "name": "HemoScan AI API",
        "status": "running",
        "endpoints": ["/health", "/predict"],
        "notice": (
            "Experimental fingerprint-based research demo. "
            "Not a medically validated blood-group test."
        )
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_PATH.name if model is not None else None,
        "classes": classes,
        "error": model_error,
        "model_error": model_error
    })


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "code": "MODEL_NOT_CONFIGURED",
            "error": model_error or "Trained model is unavailable."
        }), 503

    uploaded_file = request.files.get("image") or request.files.get("file")

    if uploaded_file is None:
        return jsonify({
            "success": False,
            "error": "Upload an image using the 'file' or 'image' field."
        }), 400

    raw_bytes = uploaded_file.read(
        MAX_UPLOAD_MB * 1024 * 1024 + 1
    )

    if not raw_bytes:
        return jsonify({
            "success": False,
            "error": "Uploaded image is empty."
        }), 400

    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({
            "success": False,
            "error": "Image file is too large."
        }), 413

    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            source.verify()

        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = source.convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

        # hemoscan_cnn.keras includes a built-in Rescaling(1./255) layer;
        # only divide by 255.0 externally if the loaded model has no Rescaling layer.
        raw_pixels = np.asarray(image, dtype=np.float32)
        image_array = (
            raw_pixels if model_has_internal_rescaling else raw_pixels / 255.0
        )

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "success": False,
            "error": "Invalid or unsupported image."
        }), 400

    try:
        predictions = np.asarray(
            model.predict(image_array[None, ...], verbose=0)[0],
            dtype=np.float64
        )

        if (
            predictions.shape != (len(classes),)
            or not np.all(np.isfinite(predictions))
            or np.any(predictions < 0)
            or not np.isclose(predictions.sum(), 1.0, atol=1e-3)
        ):
            return jsonify({
                "success": False,
                "error": "Model returned invalid prediction scores."
            }), 500

        index = int(np.argmax(predictions))
        predicted_class = classes[index]

        return jsonify({
            "success": True,
            "experimental": True,
            "notice": (
                "Not medically validated. Confirm blood group "
                "using a conventional laboratory test."
            ),
            "blood_group": predicted_class,
            "predicted_class": predicted_class,
            "confidence": float(predictions[index]),
            "probabilities": {
                label: float(score)
                for label, score in zip(classes, predictions)
            },
            "model_version": MODEL_PATH.name
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "success": False,
            "error": "Model inference failed."
        }), 500


load_model()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
