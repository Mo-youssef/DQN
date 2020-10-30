import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pfrl


class Net(nn.Module):
  def __init__(self, actions=6, dueling=0, mode='train'):
    super().__init__()
    self.dueling = dueling
    self.mode = mode
    self.conv1 = nn.Conv2d(4, 32, kernel_size=8, padding=0, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=0, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1)
    self.fc1 = nn.Linear(64*7*7, 512)

    self.value_stream = nn.Linear(64*7*7, 512)
    self.value_out = nn.Linear(512, 1)
    self.adv_stream = nn.Linear(64*7*7, 512)

    self.output_layer = nn.Linear(512, actions)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))
    feature_map = out.view(x.size(0), -1)
    if self.dueling:
        if self.mode == 'train' and feature_map.requires_grad:
          feature_map.register_hook(lambda x: x / np.sqrt(2))
        value_input = F.relu(self.value_stream(feature_map))
        value = self.value_out(value_input)
        adv_input = F.relu(self.adv_stream(feature_map))
        adv = self.output_layer(adv_input)
        out = value + adv - adv.mean(1).unsqueeze(1)
    else:
        out = F.relu(self.fc1(feature_map))
        out = self.output_layer(out)
    return pfrl.action_value.DiscreteActionValue(out)


class RNDNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.bn1 = nn.BatchNorm2d(4, affine=False)
    self.conv1 = nn.Conv2d(4, 32, kernel_size=8, padding=0, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=0, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1)
    self.output_layer = nn.Linear(64*7*7, 128)

  def forward(self, x):
    x = self.bn1(x)
    x = x.clamp_(-5, 5)
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))
    feature_map = out.view(x.size(0), -1)
    out = self.output_layer(feature_map)
    return out
