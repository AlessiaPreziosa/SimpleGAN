import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    """
    This class is the encoder block used in the U-Net, Generator of the GAN, and in the Discriminator.
    It comprehends a Convolutional operation, an eventual Normalization operation (BatchNorm)
    and an activation layer (LeakyReLU with α=0.2). This block down-samples the input by a factor of 2

    :param in_channels: Number of input channels for Convolution,
    :param out_channels: number of output channels for Convolution,
    :param stride: stride for convolution,
    :param normalization: boolean
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, normalization=True):

        super(Encoder, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=1,
                                bias=False)
        self.normalization = normalization
        if normalization:
            self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        X = self.conv2d(X)
        if self.normalization:
            X = self.batchNorm(X)
        X = F.leaky_relu(X, 0.2)
        return X


class Decoder(nn.Module):

    """
    This class is the decoder block of the U-Net used as Generator of the GAN.
    It comprehends a De-Convolutional operation, a Normalization operation (BatchNorm), an eventual Dropout (p=0.5),
    and an activation layer (ReLU). This block up-samples the input by a factor of 2

    :param in_channels: Number of input channels for Transposed Convolution,
    :param out_channels: number of output channels for Transposed Convolution,
    :param dropout: boolean
    """

    def __init__(self, in_channels, out_channels, dropout=False):

        super(Decoder, self).__init__()
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.dropout = dropout
        if dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, X):
        X = self.batchNorm(self.conv2d_trans(X))
        if self.dropout:
            X = self.drop(X)
        X = F.relu(X)
        return X
