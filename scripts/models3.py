import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class ModifiedAlexNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ModifiedAlexNet, self).__init__()

        # Load AlexNet pretrained on ImageNet
        self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 6-channel input
        self.alexnet.features[0] = nn.Conv2d(
            in_channels=6,  # 6 channels for concatenated images
            out_channels=64,  # Same as original AlexNet
            kernel_size=11,
            stride=4,
            padding=2
        )

        # Calculate feature size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, 227, 227)  # Batch size 1, 6 channels
            dummy_features = self.alexnet.features(dummy_input)
            self.feature_size = dummy_features.view(1, -1).size(1)

        # Fully connected layers for color data
        self.color_fc = nn.Sequential(
            nn.Linear(3, 32),  # Input: 3 RGB channels
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # Replace AlexNet's classifier to output combined features
        self.combined_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_size + 32, 256),  # Adjusted size
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 4)  # 4 output classes
        )

    def forward(self, images, colors):
        # Pass images through AlexNet feature extractor
        x = self.alexnet.features(images)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view to handle non-contiguous tensors

        # Process color data
        color_features = self.color_fc(colors)

        # Combine features from images and color inputs
        combined_features = torch.cat((x, color_features), dim=1)

        # Predict control classes
        outputs = self.combined_fc(combined_features)

        return outputs
