import joblib
import numpy as np
import re

def get_burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) < 2: return 0.0
    lengths = [len(s.split()) for s in sentences]
    return round(np.std(lengths) / np.mean(lengths), 4)

def predict_text(text):
    model = joblib.load("models/text_detector_v2.pkl")
    vec = joblib.load("models/tfidf_v2.pkl")
    probs = model.predict_proba(vec.transform([text]))[0]
    return {"Human": probs[0], "AI": probs[1], "Edited": probs[2]}