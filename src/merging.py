import torch
import torch.nn as nn

class TestConv3D(nn.Module):

    def __init__(self):
        super().__init__()
        num_channels = 128
        self.conv3d0 = nn.Conv3d(num_channels * 2, num_channels, kernel_size= 3, padding = 1)
        self.conv3d1 = nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1)

        self.activation = nn.ReLU()

    def forward(self, fea):

        """

        Args:
            fea: A dictionary containing encoded features/latent codes. Its components are:
              - ['latent_1']: First/even index scan
              - ['latent_2']: Second/odd index scan
              - ['latent_12']: Combination of first and second scans

        Returns: Predicted merged latent code

        """
        c = {}
        # Extract latent code
        z1, z2 = fea['latent_1'], fea['latent_2']

        # concat subsequent scans
        z = torch.cat([z1, z2], dim=1)

        z = self.conv3d0(z)
        z = self.activation(z)
        z = self.conv3d1(z)

        c['latent_merge'] = z
        c['latent_12'] = fea['latent_12']
        c['unet3d'] = fea['unet3d']
        return c