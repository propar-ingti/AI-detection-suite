import os

# --- 1. FORCE PENDRIVE SETTINGS (MUST BE AT THE TOP) ---
# This ensures the 10GB+ of images go to your F: drive, not C:
PENDRIVE_DRIVE = "F:/"
CACHE_DIR = os.path.join(PENDRIVE_DRIVE, "kaggle_cache")

os.environ['KAGGLEHUB_CACHE'] = CACHE_DIR
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- 2. IMPORTS ---
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import kagglehub

# --- 3. TEXT MODEL TRAINING (Using your local CSV) ---
print("--- Step 1: Training Text Model ---")
# Use absolute path to avoid "File Not Found" errors
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'ai_human_detection_v1.csv')

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found in the project folder!")
else:
    df = pd.read_csv(csv_path)
    # Mapping 2026 labels: human=0, ai=1, post_edited_ai=2
    df['label'] = df['human_or_ai'].map({'human': 0, 'ai': 1, 'post_edited_ai': 2})
    df = df.dropna(subset=['text', 'label'])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # FIXED: Removed 'multi_class' argument for Scikit-Learn 2026 compatibility
    text_model = LogisticRegression(max_iter=1000) 
    text_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(text_model, "models/text_detector_v2.pkl")
    joblib.dump(vectorizer, "models/tfidf_v2.pkl")
    print(f"Text Model Saved! Accuracy: {text_model.score(X_test, y_test):.2%}")

# --- 4. IMAGE MODEL TRAINING (DeepDetect-2025) ---
print("\n--- Step 2: Downloading DeepDetect to Pendrive (F:) ---")
try:
    img_path = kagglehub.dataset_download("ayushmandatta1/deepdetect-2025")
    print(f"Images are located at: {img_path}")
    train_dir = os.path.join(img_path, "train")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
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

    print("Training Image Model (First 100 batches)...")
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i == 100: break 

    torch.save(model.state_dict(), "models/deepdetect_v1.pth")
    print("Image Model Saved Successfully!")

except Exception as e:
    print(f"An error occurred during image processing: {e}")

print("\n--- ALL TRAINING COMPLETE ---")