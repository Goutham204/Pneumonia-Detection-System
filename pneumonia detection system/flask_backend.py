from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import datetime

app = Flask(__name__)

MODEL_PATH = "pneumonia_model.keras"
model = load_model(MODEL_PATH)
MODEL_VERSION = "1.0"

IMAGE_SIZE = (128, 128)

def predict_xray(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    if pred >= 0.5:
        return "Pneumonia", float(pred)
    else:
        return "Normal", float(1 - pred)

@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("images")
    results = []

    for f in files:
        filename = f.filename
        os.makedirs("uploads", exist_ok=True)
        upload_path = os.path.join("uploads", filename)
        f.save(upload_path)

        label, confidence = predict_xray(upload_path)
        patient_id = str(uuid.uuid4()) 

        results.append({
            "patient_id": patient_id,
            "image_name": filename,
            "prediction": label,
            "confidence": round(confidence, 2),
            "model_version": MODEL_VERSION,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        os.remove(upload_path)

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
