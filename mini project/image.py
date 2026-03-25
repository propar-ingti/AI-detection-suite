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
    model = DeepDetectNet()
    model.load_state_dict(torch.load("models/deepdetect_v1.pth"))
    model.eval()
    
    img = Image.open(path).convert('RGB')
    t = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    img_t = t(img).unsqueeze(0)
    
    with torch.no_grad():
        output = torch.softmax(model(img_t), dim=1)
        # DeepDetect: Index 0 is Fake, Index 1 is Real
        return output[0][0].item() * 100