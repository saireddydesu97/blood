import gc
import io
import json
import os
import threading
import time
from pathlib import Path

# Must be set BEFORE importing TensorFlow to minimize RAM and CPU threads on Render Free Tier (512 MB)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
ROOT_MODEL_DIR = BASE_DIR.parent / "models"
MODEL_DIR = (
    ROOT_MODEL_DIR
    if (ROOT_MODEL_DIR / "hemoscan_cnn.keras").exists()
    else BASE_DIR / "models"
)

MODEL_PATH = MODEL_DIR / "hemoscan_cnn.keras"
CLASSES_PATH = MODEL_DIR / "classes.json"

IMG_SIZE = 224
MAX_UPLOAD_MB = 10

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

CORS(app)

_model_lock = threading.Lock()
_model_load_attempted = False
model = None
classes = []
model_error = None
model_has_internal_rescaling = False


def _get_memory_mb() -> str:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024:.1f}MB"
    except OSError:
        pass
    return "n/a"


def _log(stage: str, message: str) -> None:
    print(
        f"[HemoScan][{stage}][pid={os.getpid()}][rss={_get_memory_mb()}] {message}",
        flush=True,
    )


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
    global model, classes, model_error, model_has_internal_rescaling, _model_load_attempted

    if model is not None:
        return

    with _model_lock:
        if model is not None or _model_load_attempted:
            return
        _model_load_attempted = True

        start_ts = time.perf_counter()
        _log("MODEL_LOAD", f"Starting single-instance model initialization from {MODEL_PATH}")

        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

            if not CLASSES_PATH.exists():
                raise FileNotFoundError(f"Class file not found: {CLASSES_PATH}")

            _log("MODEL_LOAD", "Importing TensorFlow (CPU-only, single-thread, oneDNN disabled)...")
            import tensorflow as tf

            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass

            try:
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass

            _log("MODEL_LOAD", "TensorFlow imported; calling tf.keras.models.load_model(..., compile=False)...")
            loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

            loaded_classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

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

            gc.collect()
            elapsed = time.perf_counter() - start_ts
            _log(
                "MODEL_LOAD",
                f"Model loaded successfully in {elapsed:.2f}s "
                f"(classes={len(classes)}, internal_rescaling={model_has_internal_rescaling})",
            )

        except Exception as exc:
            model = None
            classes = []
            model_has_internal_rescaling = False
            model_error = str(exc)
            _log("MODEL_LOAD", f"ERROR during model load: {model_error}")


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
    if model is None and not _model_load_attempted:
        load_model()

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
    req_start = time.perf_counter()
    _log("PREDICT", "Incoming POST /predict request")

    if model is None and not _model_load_attempted:
        load_model()

    if model is None:
        _log("PREDICT", f"Aborting: model unavailable ({model_error})")
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

    raw_bytes = uploaded_file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)

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

    _log(
        "PREDICT",
        f"Decoding image '{uploaded_file.filename or 'upload'}' ({len(raw_bytes)} bytes)...",
    )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            source.verify()

        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = source.convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

        del raw_bytes

        # hemoscan_cnn.keras includes a built-in Rescaling(1./255) layer;
        # only divide by 255.0 externally if the loaded model has no Rescaling layer.
        raw_pixels = np.asarray(image, dtype=np.float32)
        del image
        image_array = (
            raw_pixels if model_has_internal_rescaling else raw_pixels / 255.0
        )
        del raw_pixels

    except (UnidentifiedImageError, OSError, ValueError):
        _log("PREDICT", "Rejected invalid or unsupported image upload")
        return jsonify({
            "success": False,
            "error": "Invalid or unsupported image."
        }), 400

    try:
        batch = np.expand_dims(image_array, axis=0)
        del image_array

        infer_start = time.perf_counter()
        _log("PREDICT", "Running CPU forward pass model(batch, training=False)...")

        raw_output = model(batch, training=False)
        output_np = (
            raw_output.numpy()
            if hasattr(raw_output, "numpy")
            else np.asarray(raw_output)
        )
        predictions = np.asarray(output_np[0], dtype=np.float64)
        del batch, raw_output, output_np

        infer_elapsed = time.perf_counter() - infer_start

        if (
            predictions.shape != (len(classes),)
            or not np.all(np.isfinite(predictions))
            or np.any(predictions < 0)
            or not np.isclose(predictions.sum(), 1.0, atol=1e-3)
        ):
            _log("PREDICT", f"Invalid prediction vector returned: shape={predictions.shape}")
            return jsonify({
                "success": False,
                "error": "Model returned invalid prediction scores."
            }), 500

        index = int(np.argmax(predictions))
        predicted_class = classes[index]
        confidence = float(predictions[index])
        total_elapsed = time.perf_counter() - req_start

        _log(
            "PREDICT",
            f"Prediction succeeded in {infer_elapsed:.2f}s "
            f"(total={total_elapsed:.2f}s, class={predicted_class}, confidence={confidence:.4f})",
        )

        return jsonify({
            "success": True,
            "experimental": True,
            "notice": (
                "Not medically validated. Confirm blood group "
                "using a conventional laboratory test."
            ),
            "blood_group": predicted_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                label: float(score)
                for label, score in zip(classes, predictions)
            },
            "model_version": MODEL_PATH.name
        })

    except Exception:
        _log("PREDICT", "Unhandled exception during model inference")
        app.logger.exception("Prediction failed")
        return jsonify({
            "success": False,
            "error": "Model inference failed."
        }), 500


# Load once at worker startup unless explicitly deferred
if os.environ.get("LAZY_LOAD_MODEL", "0") != "1":
    load_model()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
