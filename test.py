'''
import os
import torch
from PIL import Image
import config  # Import dataset paths
from generator import Generator
from torchvision import transforms

# Load model
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth"))  
generator.eval()

# Load test image
test_image_path = os.path.join(config.IMAGE_DIR, "test_sample.jpg")  
test_image = Image.open(test_image_path).convert("RGB")
test_image = transforms.ToTensor()(test_image).unsqueeze(0).cuda()

# Run inference
output = generator(test_image)'''

#cnn 
import os
import torch
from PIL import Image
from torchvision import transforms
import config
from cnn import SimpleCNN  # Replace Generator with your CNN class
import torch.nn.functional as F
from torchvision.utils import save_image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, "cnn_segmentor_epoch10.pth"), map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((450, 600)),
    transforms.ToTensor(),
])

# Load test image
test_image_path = os.path.join(config.IMAGE_DIR, "ISIC_0024306.jpg")  # Replace with an actual test image name
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found: {test_image_path}")

test_image = Image.open(test_image_path).convert("RGB")
input_tensor = transform(test_image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    output = F.interpolate(output, size=(450, 600), mode="bilinear", align_corners=False)
    pred_mask = (output > 0.5).float()

# Save or display output
os.makedirs("outputs", exist_ok=True)
save_image(pred_mask, "outputs/predicted_mask.png")
print("✅ Prediction saved to outputs/predicted_mask.png")
