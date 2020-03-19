import torch
import torch.nn as nn


# Functions

def define_G():
    """
    Create a Discriminator
    Returns: Generator Network

    """


def define_D():
    """
    Create a Discriminator
    Returns: Discriminator network

    """


def init_weights():
    """
    Initialize network weights
    Returns:

    """


# Classes

class GANLoss(nn.Module):
    """
    Define different GAN objective
    """


class ConvBlock(nn.Module):
    """
    Define a Convolutional building block (Excluding input and output layer)
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)


class TransposeConvBlock(nn.Module):
    """
    Define a Transpose Convolutional building block (Excluding input and output layer)
    """

    def __init__(self, in_channels, out_channels):
        """

        Args:
            in_channels: Number of channels entering the transpose conv
            out_channels: Number of channels after the transpose conv
        """
        super(TransposeConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.main(input)
