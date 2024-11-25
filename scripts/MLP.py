import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )

        

    def forward(self, x):
        x = x.view(x.size(0), -1)

        return self.layers(x)

    def normalize(self, data): 
        max_speed = 50.0
        min_speed = -50.0
        max_ray_length = 100.0
        min_ray_length = 0.0

        norm_data = []

        for recording in data:
            ray = recording[:15]
            speed = recording[15]

            norm_ray = []
            for distance in ray:
                norm_ray.append((distance - min_ray_length) / (max_ray_length - min_ray_length))

            norm_speed = (speed - min_speed) / (max_speed - min_speed)

            norm_data.append(norm_ray + [norm_speed])

        return norm_data