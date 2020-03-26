import torch
from .base_model import BaseModel
from . import network


class DCGANModel(BaseModel):
    """
    This class implements the DCGAN model

    dcgan paper: https://arxiv.org/pdf/1511.06434.pdf
    """

    def __init__(self, opt):
        """
        Initialize the dcgan class
        Args:
            opt: (Option class) --- stores all the experiment flags
        """

        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # Define networks

        self.netG = network.define_G(opt.input_nc, opt.ngf, opt.output_nc)

        if self.isTrain:
            self.netD = network.define_D(opt.input_nc, opt.ndf, opt.output_nc)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers (schedulers will be automatically created by function <BaseModel.setup>
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        pass

    def forward(self):
        pass

    def optimize_parameters(self):
        pass