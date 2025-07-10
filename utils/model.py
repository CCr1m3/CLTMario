import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPolicy(nn.Module):
    def __init__(self, num_actions):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)   # Input: (C, H, W) = (3, 120, 128)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute conv output size: use dummy forward pass or compute manually
        self.flattened_size = self._get_conv_output_shape()

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)  # outputs Q-values or logits per action

    def _get_conv_output_shape(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 120, 128)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        x = x / 255.0  # normalize to [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values/logits (no softmax)