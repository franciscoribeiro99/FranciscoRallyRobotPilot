import torch
import torch.nn as nn
import torch

class SimpleImageColorNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleImageColorNet, self).__init__()

        # Robust CNN for image processing
        self.image_net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),  # Input: 6 channels
            nn.BatchNorm2d(32),  # Improved stability
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Image fully connected layers
        self.image_fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),  # Adjust based on input image size (224x224)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Regularization
        )

        # Fully connected layers for color data
        self.color_fc = nn.Sequential(
            nn.Linear(3, 64),  # Input: 3 (R, G, B values)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Regularization
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 128),  # Combine image and color features
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(128, num_classes),  # Final classification
        )

    def forward(self, images, colors):
        # Ensure image data is in the correct shape
        if images.dim() != 4:
            raise ValueError("Expected image input with 4 dimensions: [batch_size, channels, height, width]")
        
        # Process image data
        x = self.image_net(images)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]
        image_features = self.image_fc(x)

        # Ensure color data is in the correct shape
        if colors.dim() != 2 or colors.size(1) != 3:
            raise ValueError("Expected color input with shape [batch_size, 3]")
        
        color_features = self.color_fc(colors)

        # Concatenate features and classify
        combined_features = torch.cat((image_features, color_features), dim=1)
        outputs = self.classifier(combined_features)

        return outputs