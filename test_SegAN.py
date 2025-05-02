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
output = generator(test_image)