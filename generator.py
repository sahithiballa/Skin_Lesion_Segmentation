# generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),# kernel size is not 4 keep it 3 or 7 5 3
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
