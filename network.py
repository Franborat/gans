import torch
import torch.nn as nn


# Functions
from torch.optim import lr_scheduler


def define_G(input_nc, ngf, output_nc):
    """

    Args:
        input_nc: Number of channels of input noise
        ngf: Number of filters in the last conv layer
        output_nc: Number of channels of output image

    Returns: Generator network with initialized weights

    """
    generator_net = nn.Sequential(
        # Input layer
        nn.ConvTranspose2d(in_channels=input_nc, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=ngf * 8),
        nn.ReLU(True),
        # Trans Conv blocks
        TransposeConvBlock(in_channels=ngf * 8, out_channels=ngf * 4),
        TransposeConvBlock(in_channels=ngf * 4, out_channels=ngf * 2),
        TransposeConvBlock(in_channels=ngf * 2, out_channels=ngf),
        # Output layer
        nn.ConvTranspose2d(in_channels=ngf, out_channels=output_nc, kernel_size=4, stride=2, padding=1, bias=False)
    )
    # Initialize network weights
    init_weights(generator_net)
    return generator_net


def define_D(input_nc, ndf, output_nc):
    """

    Args:
        input_nc: Number of channels of input image
        ndf: Number of filters in the first conv layer
        output_nc: Number of channels of output image

    Returns: Discriminator network with initialized weights

    """
    discriminator_net = nn.Sequential(
        # Input layer
        nn.Conv2d(in_channels=input_nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
        # Conv blocks
        ConvBlock(in_channels=ndf, out_channels=ndf * 2),
        ConvBlock(in_channels=ndf * 2, out_channels=ndf * 4),
        ConvBlock(in_channels=ndf * 4, out_channels=ndf * 8),
        # Output layer
        nn.Conv2d(in_channels=ndf * 8, out_channels=output_nc, kernel_size=4, stride=1, padding=0, bias=False)
    )
    init_weights(discriminator_net)
    return discriminator_net


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


def get_scheduler(optimizer, opt):
    """

    Args:
        optimizer: Optimizer of the network
        opt: opt.lr_policy stores the name of the learning rate policy

    Returns:

    """
    # Keep the same lr for the first opt.n_epochs and then decay for the next opt.n_epochs_decay
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # Reduce learning rate when a metric has stopped improving
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


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
