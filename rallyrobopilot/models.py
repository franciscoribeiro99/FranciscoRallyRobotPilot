import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten_size = 32 * 128 * 160  # Update this based on actual output after layer2
        self.fc = nn.Linear(self.flatten_size, num_classes)  # Use calculated flatten size
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out

# Define the second CNN for colors
class CNNColor(nn.Module):
    def __init__(self, num_classes):
        super(CNNColor, self).__init__()
        self.fc = nn.Linear(3, num_classes)
        
    def forward(self, x):
        out = self.fc(x)
        return out

class ConcatModel(nn.Module):
    def __init__(self, cnn, cnn_color):
        super(ConcatModel, self).__init__()
        self.cnn = cnn
        self.cnn_color = cnn_color
        # Update output dimension of the final layer to match the labels' dimension (4)
        self.fc = nn.Linear(cnn.fc.out_features + cnn_color.fc.out_features, 4)
        
    def forward(self, x, y):
        x = self.cnn(x)  # Output from image CNN
        y = self.cnn_color(y)  # Output from color CNN
        z = torch.cat((x, y), 1)  # Concatenate along the feature dimension
        z = self.fc(z)  # Final output with 4 dimensions
        return z