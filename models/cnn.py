import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (6, 1), (2, 1), 0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, (6, 1), (2, 1), 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 256, (6, 2), (2, 1), 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.flc = nn.Linear(10752, 6)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)
        return x