import torch
import torch.nn as nn
import torch.nn.functional as F

class Region_Proposal(nn.Module):
    def __init__(self):
        super(Region_Proposal, self).__init__()
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_reg = nn.Conv2d(128, 9*4, kernel_size=1, stride=1, padding=0)
        self.conv_cls = nn.Conv2d(128, 9*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        out1 = self.conv_reg(x)
        out2 = self.conv_cls(x)
        return out1, out2

