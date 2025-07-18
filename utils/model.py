import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPolicy(nn.Module):
    def __init__(self, num_actions, frame_stack=4):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.extra_features = 2 + 2 + num_actions
        self.flattened_size = self._get_conv_output_shape(frame_stack)
        self.fc1 = nn.Linear(self.flattened_size + self.extra_features, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output_shape(self, frame_stack):
        with torch.no_grad():
            dummy_input = torch.zeros(1, frame_stack, 60, 64)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x, extra=None):
        x = x / 255.0  # normalize to [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        if extra is not None:
            x = torch.cat([x, extra], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values/logits (no softmax)