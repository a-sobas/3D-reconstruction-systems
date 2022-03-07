from torch import nn
from torch.nn import functional as F


class VoxelsModule(nn.Module):
    def __init__(self, input_channels):
        super(VoxelsModule, self).__init__()

        self.voxel_size = 32

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=self.voxel_size,
            kernel_size=1,
            stride=1,
            padding=0
        )


    def forward(self, x):
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv2(x))
        x = self.conv4(x)

        return x