import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(4, 16, kernel_size=8, padding=3,
                           stride=4)  # output = 28 x 28
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4,
                           padding=0, stride=2)  # output = 13 x 13
    self.fc1 = nn.Linear(2592, 256)
    self.fc2 = nn.Linear(256, 6)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = out.view(-1, 2592)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out
