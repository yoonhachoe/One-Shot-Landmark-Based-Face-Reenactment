import torch.nn as nn


class LandmarkEncoder(nn.Module):
    def __init__(self, encoded_landmark_size=16 * 512):
        super(LandmarkEncoder, self).__init__()
        self.encoded_landmark_size = encoded_landmark_size
        self.landmark_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(8192, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),

            nn.Linear(768, self.encoded_landmark_size)
        )

    def forward(self, landmark):
        encoded_landmark = self.landmark_encoder(landmark)

        return encoded_landmark

class MappingNetwork(nn.Module):
    def __init__(self, LandmarkEncoder):
        super(MappingNetwork, self).__init__()
        self.landmark_encoder = LandmarkEncoder

    def forward(self, landmark_id, landmark):
        landmark_delta = landmark - landmark_id
        encoded_landmark = self.landmark_encoder(landmark_delta)

        return encoded_landmark