import torch
from GAN.blocks import *


class Generator(nn.Module):

    def __init__(self):

        """
        This class creates the generator of the network (sort of U-Net),
        starting from the Encoder and Decoder blocks
        """

        super(Generator, self).__init__()

        # input size (N, C, H, W): (N, 3, 128, 128)
        ch = 64

        # an encoder ModuleList is created to ease the access to weights
        self.Encoder = nn.ModuleList([
            Encoder(3, ch, normalization=False),  # 1. (N, 64, 64, 64)
            Encoder(ch, ch*2),  # 2. (N, 128, 32, 32)
            Encoder(ch*2, ch*4),  # 3. (N, 256, 16, 16)
            Encoder(ch*4, ch*8),  # 4. (N, 512, 8, 8)
            Encoder(ch*8, ch*8)  # 5. (N, 512, 4, 4)
        ])

        # input size: (N, 512, 4, 4)

        # a decoder ModuleList is created to ease the access to parameters
        self.Decoder = nn.ModuleList([
            Decoder(ch*8, ch*8, dropout=True),  # 1. (N, 512, 8, 8) + (4E) -> (N, 1024, 8, 8)
            Decoder(ch*16, ch*4, dropout=True),  # 2. (N, 256, 16, 16) + (3E) -> (N, 512, 16, 16)
            Decoder(ch*8, ch*2),  # 3. (N, 128, 32, 32) + (2E) -> (N, 256, 32, 32)
            Decoder(ch*4, ch),  # 4. (N, 64, 64, 64) + (1E) -> (N, 128, 64, 64)
            nn.ConvTranspose2d(ch*2, 3, kernel_size=4, stride=2, padding=1)  # 5. (N, 3, 128, 128)
        ])

    def forward(self, X):
        """
        :param X: input image
        :return: generated image
        """

        skip_cons = []

        # encoding from 128x128x3 to 4x4x512: downsample
        for encoder in self.Encoder:
            X = encoder(X)
            skip_cons.append(X)

        skip_cons = list(reversed(skip_cons[:-1]))

        # decoding from 4x4x512 to 64x64x128: upsample
        for decoder, skip in zip(self.Decoder[:-1], skip_cons):
            X = decoder(X)
            X = torch.cat((X, skip), 1)

        # last layer: De-Convolution + Tanh
        # size from 64x64x128 to 128x128x3
        X = F.tanh(self.Decoder[-1](X))

        return X
