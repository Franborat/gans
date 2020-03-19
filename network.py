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


def init_weights(net):
    """
    Initialize network weights
    Returns: Convolutional and BatchNorm layers initialized as in DCGAN paper

    """
    classname = net.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, mean=1.0, std=.02)
        nn.init.constant_(net.bias.data, 0)


# Classes

class GANLoss(nn.Module):
    """
    Define different GAN objective
    """
    def __init__(self, gan_mode):
        """

        Args:
            gan_mode: Adversarial Loss type. Choose between vanilla (log loss) or LSGAN loss (sigmoid + BCE)
        Note:
            Do not use sigmoid in the last layer of the discriminator for the vanilla GAN
            BCEWithLogitsLoss already handles that.

        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)


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
