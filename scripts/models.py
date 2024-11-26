import torch
import torch.nn as nn
from torchvision import models

# Pretrained CNN model (ResNet as feature extractor)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Load a pretrained ResNet18 model
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze the pretrained layers to prevent updates
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Add a custom fully connected layer for your task
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Pass through the ResNet feature extractor
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Fully connected layer
        return x

# CNN for color processing
class CNNColor(nn.Module):
    def __init__(self, num_classes):
        super(CNNColor, self).__init__()
        self.fc = nn.Linear(3, num_classes)  # Input is RGB (3 channels)

    def forward(self, x):
        return self.fc(x)

# Combined model
class ConcatModel(nn.Module):
    def __init__(self, cnn, cnn_color):
        super(ConcatModel, self).__init__()
        self.cnn = cnn
        self.cnn_color = cnn_color
        # Combine the output features of both networks
        self.fc = nn.Linear(cnn.fc.out_features + cnn_color.fc.out_features, 4)  # 4 output classes (commands)

    def forward(self, x, y):
        x = self.cnn(x)  # Image features from pretrained CNN
        y = self.cnn_color(y)  # Color features
        z = torch.cat((x, y), 1)  # Concatenate features
        z = torch.relu(z)
        z = self.fc(z)  # Final prediction
        return z
