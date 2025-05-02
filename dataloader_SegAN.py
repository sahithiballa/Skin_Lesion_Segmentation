from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import config  # Import centralized paths

# Validate dataset existence before running
if not os.path.exists(config.IMAGE_DIR):
    raise FileNotFoundError(f"Image directory not found: {config.IMAGE_DIR}")
if not os.path.exists(config.MASK_DIR):
    raise FileNotFoundError(f"Mask directory not found: {config.MASK_DIR}")

# Ensure all images have corresponding masks
image_files = sorted(os.listdir(config.IMAGE_DIR))
mask_files = sorted(os.listdir(config.MASK_DIR))

missing_masks = [
    img for img in image_files if img.replace(".jpg", "_segmentation.png").replace(" (2)", "") not in mask_files
]
if missing_masks:
    raise FileNotFoundError(f"Missing mask files for: {missing_masks[:10]}... (Showing first 10)")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((450, 600)),  # Ensure consistent size
    transforms.ToTensor()
])

class SkinDataset(Dataset):
    def __init__(self, image_dir=config.IMAGE_DIR, mask_dir=config.MASK_DIR, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_filename = self.image_filenames[idx].replace(".jpg", "_segmentation.png").replace(" (2)", "").replace(" (1)", "")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file missing: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# **Compile-time check ends here**
print("✅ All dataset files are verified!")

# Create DataLoader with the defined transformation
dataset = SkinDataset(transform=transform)
train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)