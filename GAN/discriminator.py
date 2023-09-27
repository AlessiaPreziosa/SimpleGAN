import torch
import torch.nn as nn
from GAN.blocks import Encoder


class Discriminator(nn.Module):

    def __init__(self):

        """
        This class creates the discriminator of the network: PatchGAN, it uses only a patch of the image
        to discriminate if it is real or fake.
        It is faster and motivates the discriminator to work in a high-frequency structure
        """

        super(Discriminator, self).__init__()

        # input size (N, C, H, W): (N, 3, 128, 128) + (N, 3, 128, 128) -> (N, 6, 128, 128)

        ch = 64
        self.Encoder = nn.ModuleList([
            Encoder(6, ch, normalization=False),  # 1. (N, 64, 64, 64)
            Encoder(ch, ch*2),  # 2. (N, 128, 32, 32)
            Encoder(ch*2, ch*4),  # 3. (N, 256, 16, 16)
            Encoder(ch*4, ch*8, kernel_size=3, stride=1),  # 4. (N, 512, 16, 16)
            nn.Conv2d(ch*8, 1, kernel_size=4, stride=1, padding=1)  # 5. (N, 1, 15, 15)
        ])

    def forward(self, X, Y):
        """
        :param X: input image
        :param Y: target image or generated image
        :return: patch that classifies the input image
        """

        X = torch.cat((X, Y), 1)

        for encoder in self.Encoder:
            X = encoder(X)

        return X

