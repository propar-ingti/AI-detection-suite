import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

def predict_image(path):
    # 1. Setup Device (Handles case where you have a GPU or just CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    model = DeepDetectNet().to(device)
    # Use weights_only=True for security (standard in newer torch versions)
    model.load_state_dict(torch.load("models/deepdetect_v1.pth", map_location=device))
    model.eval()

    # 3. Process Image
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        return f"Error opening image: {e}"

    # IMPORTANT: These transforms MUST match your training script exactly
    t = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Added normalization
    ])
    
    img_t = t(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)[0]
    
    # Index 0: Fake, Index 1: Real
    fake_score = probs[0].item() * 100
    real_score = probs[1].item() * 100

    return {
        "Fake Probability": round(fake_score, 2),
        "Real Probability": round(real_score, 2),
        "Verdict": "AI/Fake" if fake_score > real_score else "Human/Real"
    }