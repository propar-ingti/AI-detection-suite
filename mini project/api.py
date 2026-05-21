from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
import numpy as np
import re
import torch
import torch.nn as nn

from scipy.sparse import hstack, csr_matrix

from PIL import Image
from torchvision import transforms, models

# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)
CORS(app)

# ==================================================
# LOAD TEXT MODELS
# ==================================================

text_model = joblib.load(
    "models/text_detector_v3.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_v3.pkl"
)

# ==================================================
# LOAD IMAGE MODEL
# ==================================================

class DeepDetectNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        return self.net(x)

image_model = DeepDetectNet()

image_model.load_state_dict(
    torch.load(
        "models/deepdetect_v1.pth",
        map_location=torch.device("cpu")
    )
)

image_model.eval()

# ==================================================
# IMAGE TRANSFORM
# ==================================================

image_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==================================================
# TEXT FEATURE EXTRACTION
# ==================================================

def get_numerical_features(text):

    text = str(text)

    sentences = re.split(
        r'[.!?]+',
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    words = text.split()

    # Burstiness
    if len(sentences) < 2:

        burstiness = 0.0

    else:

        lengths = [
            len(s.split())
            for s in sentences
        ]

        burstiness = (
            np.std(lengths)
            /
            (np.mean(lengths) + 1e-6)
        )

    # Average word length
    avg_word_len = (

        np.mean([
            len(w)
            for w in words
        ])

        if words else 0
    )

    # Sentence count
    sentence_count = len(sentences)

    # Unique word ratio
    unique_ratio = (

        len(set(words))
        /
        (len(words) + 1e-6)

        if words else 0
    )

    return [

        burstiness,

        avg_word_len,

        sentence_count,

        unique_ratio
    ]

# ==================================================
# TEXT PREDICTION ROUTE
# ==================================================

@app.route("/predict", methods=["POST"])
def predict_text():

    data = request.json

    text = data.get("text", "")

    # TF-IDF
    X_tfidf = vectorizer.transform([text])

    # Numerical Features
    X_numerical = np.array([
        get_numerical_features(text)
    ])

    X_numerical_sparse = csr_matrix(
        X_numerical
    )

    # Combine Features
    X_final = hstack([

        X_tfidf,

        X_numerical_sparse
    ])

    prediction = text_model.predict(
        X_final
    )[0]

    probabilities = text_model.predict_proba(
        X_final
    )[0]

    labels = {

        0: "Human",

        1: "AI",

        2: "Edited AI"
    }

    return jsonify({

        "prediction":
        labels[prediction],

        "confidence":
        float(max(probabilities) * 100)
    })

# ==================================================
# IMAGE PREDICTION ROUTE
# ==================================================

@app.route("/predict-image", methods=["POST"])
def predict_image():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["image"]

    image = Image.open(file).convert("RGB")

    image_tensor = image_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = image_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        fake_probability = (
            probabilities[0][1].item()
            * 100
        )

        real_probability = (
            probabilities[0][0].item()
            * 100
        )

    verdict = (

        "AI Generated"

        if fake_probability > real_probability

        else "Real Image"
    )

    return jsonify({

        "verdict":
        verdict,

        "fake_probability":
        round(fake_probability, 2),

        "real_probability":
        round(real_probability, 2)
    })

# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )