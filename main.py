import os
from train import train

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
