'''import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import train_loader  # Ensure this is correctly imported from dataloader.py
import config  # Import dataset paths and hyperparameters
from generator import Generator
from discriminator import Discriminator
from loss import adversarial_loss  # Import custom adversarial loss
from loss import multi_scale_l1_loss
import torch.nn.functional as F

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

# Define loss functions
criterion_L1 = nn.L1Loss()  # Multi-scale L1 loss
criterion_Adv = adversarial_loss  # Custom adversarial loss

# Training loop
for epoch in range(config.EPOCHS):
    print(f"Epoch [{epoch+1}/{config.EPOCHS}]")

    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        fake_masks = generator(images)  # Generate fake masks
        # Resize fake masks to match the ground truth masks
        fake_masks = F.interpolate(fake_masks, size=masks.shape[2:], mode="bilinear", align_corners=False)
        loss_L1 = criterion_L1(fake_masks, masks)  # L1 loss
        #loss_adv = criterion_Adv(discriminator(fake_masks), torch.ones_like(discriminator(fake_masks)))  # Adversarial loss
        loss_adv = criterion_Adv(discriminator(fake_masks, images), torch.ones_like(discriminator(fake_masks, images)))


        loss_G = loss_L1 + loss_adv  # Total generator loss
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        #real_preds = discriminator(masks)  # Real masks
        #fake_preds = discriminator(fake_masks.detach())  # Fake masks (detach to avoid affecting Generator)
        real_preds = discriminator(masks, images)  # Pass both masks and images
        fake_preds = discriminator(fake_masks.detach(), images)  # Pass both fake masks and images

        

        loss_real = criterion_Adv(real_preds, torch.ones_like(real_preds))
        loss_fake = criterion_Adv(fake_preds, torch.zeros_like(fake_preds))

        loss_D = (loss_real + loss_fake) / 2  # Total discriminator loss
        loss_D.backward()
        optimizer_D.step()

        # Print training progress every 100 batches
        if i % 100 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss_G: {loss_G.item():.4f} | Loss_D: {loss_D.item():.4f}")

    # Save models at the end of each epoch
    torch.save(generator.state_dict(), f"models/generator_epoch{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"models/discriminator_epoch{epoch+1}.pth")

print("Training Complete ✅")'''

#final segan
'''import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataloader import train_loader  # Ensure this is correctly imported from dataloader.py
import config  # Import dataset paths and hyperparameters
from generator import Generator
from discriminator import Discriminator
from loss import adversarial_loss, multi_scale_l1_loss  # Import custom loss functions
import torch.nn.functional as F

# **Compile-Time Checks**
if not os.path.exists(config.IMAGE_DIR) or not os.path.exists(config.MASK_DIR):
    raise FileNotFoundError(f"Dataset folders not found: {config.IMAGE_DIR} or {config.MASK_DIR}")

if not len(os.listdir(config.IMAGE_DIR)) or not len(os.listdir(config.MASK_DIR)):
    raise ValueError("Dataset is empty! Ensure images and masks exist in respective folders.")

print("✅ Dataset files verified. Proceeding to model initialization...")

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

# Define loss functions
criterion_L1 = nn.L1Loss()  # Multi-scale L1 loss
criterion_Adv = adversarial_loss  # Custom adversarial loss

# **Compile-Time Check for Model Shapes**
sample_images, sample_masks = next(iter(train_loader))
sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)

try:
    fake_masks = generator(sample_images)  # Test model output shape
    fake_masks = F.interpolate(fake_masks, size=sample_masks.shape[2:], mode="bilinear", align_corners=False)

    assert fake_masks.shape == sample_masks.shape, f"Shape mismatch: {fake_masks.shape} vs {sample_masks.shape}"

    real_preds = discriminator(sample_masks, sample_images)  # Test discriminator
    fake_preds = discriminator(fake_masks, sample_images)

    assert real_preds.shape == fake_preds.shape, "Discriminator output shape mismatch!"
except Exception as e:
    raise RuntimeError(f"Model shape mismatch: {e}")

print("✅ Model architecture verified. Proceeding to training...")
def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()  # Thresholding for binary segmentation
    intersection = torch.sum(y_pred * y_true)
    return (2. * intersection + smooth) / (torch.sum(y_pred) + torch.sum(y_true) + smooth)

def iou_score(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    return (intersection + smooth) / (union + smooth)

# Training loop
for epoch in range(config.EPOCHS):
    print(f"Epoch [{epoch+1}/{config.EPOCHS}]")

    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        fake_masks = generator(images)
        fake_masks = F.interpolate(fake_masks, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss_L1 = criterion_L1(fake_masks, masks)
        loss_adv = criterion_Adv(discriminator(fake_masks, images), torch.ones_like(discriminator(fake_masks, images)))
        loss_G = loss_L1 + loss_adv
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_preds = discriminator(masks, images)
        fake_preds = discriminator(fake_masks.detach(), images)

        loss_real = criterion_Adv(real_preds, torch.ones_like(real_preds))
        loss_fake = criterion_Adv(fake_preds, torch.zeros_like(fake_preds))
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # **Compute Dice & IoU for tracking performance**
        dice = dice_coefficient(fake_masks, masks)
        iou = iou_score(fake_masks, masks)

        # Print training progress every 100 batches
        if i % 100 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss_G: {loss_G.item():.4f} | Loss_D: {loss_D.item():.4f}")


    # Save models at the end of each epoch
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), f"models/generator_epoch{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"models/discriminator_epoch{epoch+1}.pth")

print("Training Complete ✅")'''

#cnn 
import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataloader import train_loader
import config
from cnn import SimpleCNN  # Your new CNN model
import torch.nn.functional as F
# optimized change 2 lines
from loss import bce_dice_loss
criterion = bce_dice_loss

# Compile-time dataset checks
if not os.path.exists(config.IMAGE_DIR) or not os.path.exists(config.MASK_DIR):
    raise FileNotFoundError(f"Dataset folders not found: {config.IMAGE_DIR} or {config.MASK_DIR}")

if not len(os.listdir(config.IMAGE_DIR)) or not len(os.listdir(config.MASK_DIR)):
    raise ValueError("Dataset is empty! Ensure images and masks exist.")

print("✅ Dataset files verified. Proceeding to model initialization...")

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize CNN model
model = SimpleCNN().to(device)
#optimized change
#from cnn import UNet
#model = UNet().to(device)


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    


# Loss
criterion = nn.BCELoss()  # For binary segmentation

# Verify model I/O shape
sample_images, sample_masks = next(iter(train_loader))
sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)
with torch.no_grad():
    outputs = model(sample_images)
    outputs = F.interpolate(outputs, size=sample_masks.shape[2:], mode="bilinear", align_corners=False)
    assert outputs.shape == sample_masks.shape, f"Shape mismatch: {outputs.shape} vs {sample_masks.shape}"

print("✅ Model architecture verified. Proceeding to training...")

# Dice and IoU metrics
def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_pred * y_true)
    return (2. * intersection + smooth) / (torch.sum(y_pred) + torch.sum(y_true) + smooth)

def iou_score(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    return (intersection + smooth) / (union + smooth)

# Training loop
for epoch in range(config.EPOCHS):
    print(f"Epoch [{epoch+1}/{config.EPOCHS}]")

    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        dice = dice_coefficient(outputs, masks)
        iou = iou_score(outputs, masks)

        if i % 100 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}")
        #scheduler.step(loss.item())  # loss from current batch or average


    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/cnn_segmentor_epoch{epoch+1}.pth")

print("Training Complete!")
