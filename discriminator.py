# discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        input = torch.cat((x, y), dim=1)
        return self.model(input)
