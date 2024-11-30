import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedAlexNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ModifiedAlexNet, self).__init__()

        firstLayerSize = 64
        secondLayerSize = 128
        thirdLayerSize = 164
        fourthLayerSize = 128
        fifthLayerSize = 64

        # Feature extractor with skip connections
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(6, firstLayerSize, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(firstLayerSize),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(firstLayerSize, secondLayerSize, kernel_size=5, padding=2),
            nn.BatchNorm2d(secondLayerSize),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(secondLayerSize, thirdLayerSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(thirdLayerSize),
            nn.ReLU(),
            nn.Conv2d(thirdLayerSize, fourthLayerSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(fourthLayerSize),
            nn.ReLU(),
            nn.Conv2d(fourthLayerSize, fifthLayerSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(fifthLayerSize),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Calculate feature size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, 227, 227)  # Batch size 1, 6 channels
            dummy_features = self.feature_extractor(dummy_input)
            self.feature_size = dummy_features.view(1, -1).size(1)

        # Fully connected layers for color data
        self.color_fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Combined classifier
        self.combined_fc = nn.Sequential(
            nn.Linear(self.feature_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 4)  # Output classes
        )

    def forward(self, images, colors):
        # Extract image features
        x = self.feature_extractor(images)
        x = x.reshape(-1, self.feature_size)

        # Extract color features
        color_features = self.color_fc(colors)

        # Combine image and color features
        combined_features = torch.cat((x, color_features), dim=1)

        # Output predictions
        outputs = self.combined_fc(combined_features)

        return outputs
