import os
import sys
import joblib
import pandas as pd
import numpy as np
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Better for mixed features
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kagglehub

# --- 1. FORCE E SETTINGS ---
E_DRIVE = "E:/"
CACHE_DIR = os.path.join(E_DRIVE, "kaggle_cache")

# Set environment variable BEFORE importing kagglehub
os.environ['KAGGLEHUB_CACHE'] = CACHE_DIR

if not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR)
    except PermissionError:
        print(f"Error: No permission to write to {E_DRIVE}. Is the switch on your pendrive set to Read-Only?")
        sys.exit()

def get_numerical_features(text):
    text = str(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # Burstiness
    if len(sentences) < 2:
        burstiness = 0.0
    else:
        lengths = [len(s.split()) for s in sentences]
        burstiness = np.std(lengths) / (np.mean(lengths) + 1e-6)
    
    # Avg Word Length & Sentence Count
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    sentence_count = len(sentences)
    
    return [burstiness, avg_word_len, sentence_count]

# --- 3. TEXT MODEL TRAINING ---
print("--- Step 1: Training Text Model ---")

HF_CACHE = os.path.join(E_DRIVE, "huggingface_cache")

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE

# Load Hugging Face dataset
ds = load_dataset("artem9k/ai-text-detection-pile")

# Convert to pandas dataframe
df = ds["train"].to_pandas()

# Create labels
df['label'] = df['source'].map({
    'human': 0,
    'ai': 1
})

# Remove missing values
df = df.dropna(subset=['text', 'label'])

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_tfidf = vectorizer.fit_transform(df['text'])

# Calculate and stack numerical features
print("Calculating writing style features...")
X_numerical = np.array([get_numerical_features(t) for t in df['text']])
X_final = hstack([X_tfidf, X_numerical])

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

text_model = LogisticRegression(
    max_iter=2000
) 
text_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(text_model, "models/text_detector_v2.pkl")
joblib.dump(vectorizer, "models/tfidf_v2.pkl")

label_map = {
    "human": 0,
    "ai": 1
}

joblib.dump(label_map, "models/label_map.pkl")

accuracy = text_model.score(X_test, y_test)
predictions = text_model.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
print(f"Text Model Saved! Accuracy: {accuracy:.2%}")

# --- 4. IMAGE MODEL TRAINING (Robust Version) ---
print("\n--- Step 2: Accessing DeepDetect on E: drive---")
try:
    # download using a simpler method to avoid terminal "junk"
    img_path = kagglehub.dataset_download("ayushmandatta1/deepdetect-2025")
    train_dir = os.path.join(img_path, "train")
    
    # Check if directory actually exists
    if not os.path.exists(train_dir):
        # Handle cases where the path structure might be nested
        train_dir = os.path.join(img_path, "ddata", "train")

    print(f"Verified Training Path: {train_dir}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Custom Loader to skip corrupted/locked files on the pendrive
    def safe_loader(path):
        try:
            from PIL import Image
            return Image.open(path).convert('RGB')
        except Exception:
            return None # We will filter these out

    train_data = datasets.ImageFolder(train_dir, transform=transform, loader=safe_loader)
    
    
    from PIL import Image

    valid_samples = []

    for path, label in train_data.samples:
        try:
            img = Image.open(path)
            img.verify()
            valid_samples.append((path, label))
        except:
            print(f"Skipped corrupted file: {path}")

    train_data.samples = valid_samples

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    class DeepDetectNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 14 * 14, 128), nn.ReLU(),
                nn.Linear(128, 2)
            )
        def forward(self, x): return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDetectNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training Image Model...")
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Skip batch if image loading failed
        if images is None: continue 
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Batch {i}/100 processed...")
        if i == 100:
            print("Early stopping for testing purposes")
            break 

    os.makedirs("models", exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join("models", "deepdetect_v1.pth")
    )
    print("✅ Image Model Saved Successfully!")

except Exception as e:
    print(f" Error during processing: {e}")

print("\n--- ALL TRAINING COMPLETE ---")