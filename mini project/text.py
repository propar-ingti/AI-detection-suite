import joblib
import numpy as np
import re
from scipy.sparse import hstack

def get_burstiness_features(text):
    """
    Extracts the same numerical features used during training.
    """
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # 1. Burstiness (Coefficient of Variation)
    if len(sentences) < 2:
        burstiness = 0.0
    else:
        lengths = [len(s.split()) for s in sentences]
        # Added epsilon (1e-6) to prevent division by zero
        burstiness = np.std(lengths) / (np.mean(lengths) + 1e-6)
    
    # 2. Average Word Length
    words = str(text).split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    
    # 3. Sentence Count
    sentence_count = len(sentences)
    
    return np.array([[burstiness, avg_word_len, sentence_count]])

def predict_text(text):
    # Load the updated models
    model = joblib.load("models/text_detector_v2.pkl")
    vec = joblib.load("models/tfidf_v2.pkl")
    
    # --- Step 1: Transform text with TF-IDF ---
    tfidf_vec = vec.transform([text])
    
    # --- Step 2: Extract numerical features ---
    extra_features = get_burstiness_features(text)
    
    # --- Step 3: Combine features ---
    # We must stack them in the EXACT same order as during training
    full_features = hstack([tfidf_vec, extra_features])
    
    # Get probabilities
    probs = model.predict_proba(full_features)[0]
    
    # Map to classes (assuming 0: Human, 1: AI, 2: Edited)
    return {
        "Human": round(float(probs[0]), 4),
        "AI": round(float(probs[1]), 4),
        "Edited": round(float(probs[2]), 4)
    }

# Example usage:
# result = predict_text("Insert your sample text here.")
# print(result)