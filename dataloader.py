# cnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import config

# ✅ Validate dataset existence
if not os.path.exists(config.IMAGE_DIR):
    raise FileNotFoundError(f"Image directory not found: {config.IMAGE_DIR}")
if not os.path.exists(config.MASK_DIR):
    raise FileNotFoundError(f"Mask directory not found: {config.MASK_DIR}")

# ✅ Validate all images have masks
image_files = sorted(os.listdir(config.IMAGE_DIR))
mask_files = sorted(os.listdir(config.MASK_DIR))

missing_masks = [
    img for img in image_files 
    if img.replace(".jpg", "_segmentation.png").replace(" (2)", "").replace(" (1)", "") not in mask_files
]
if missing_masks:
    raise FileNotFoundError(f"Missing mask files for: {missing_masks[:10]}...")

print(f"✅ {len(image_files)} images and {len(mask_files)} masks verified.")

# ✅ Define image and mask transforms
transform_image = transforms.Compose([
    transforms.Resize((450, 600)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5]*3, [0.5]*3)  # Optional: if you use pretrained CNNs like ResNet
])
#optimised change
'''transform_image = transforms.Compose([
    transforms.Resize((448, 608)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])'''


transform_mask = transforms.Compose([
    transforms.Resize((450, 600)),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Lambda(lambda x: (x > 0.5).float())  # Binarize mask just in case
])

class SkinDataset(Dataset):
    def __init__(self, image_dir=config.IMAGE_DIR, mask_dir=config.MASK_DIR):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_file = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_file)

        # Clean the name and build mask filename
        base_name = img_file.replace(".jpg", "").replace(" (2)", "").replace(" (1)", "")
        mask_file = f"{base_name}_segmentation.png"
        mask_path = os.path.join(self.mask_dir, mask_file)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply separate transforms
        image = transform_image(image)
        mask = transform_mask(mask)

        return image, mask

# ✅ Create and export DataLoader
dataset = SkinDataset()
train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

print("✅ DataLoader is ready for training.")
