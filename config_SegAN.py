import os

# Define dataset paths
BASE_DIR = "D:/SAHITHI BALLA/projects/SegAN_Project/dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

# Training parameters
BATCH_SIZE = 32 #16,32,...
EPOCHS = 10
LEARNING_RATE = 0.001 #0.001 before 0.0002