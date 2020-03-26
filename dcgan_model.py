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

        self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        self.visual_names = ['noise', 'fake', 'real']
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
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real = input['real'].to(self.device)
        self.noise = input['noise'].to(self.device)
        self.image_paths = input['paths']

    def forward(self):
        """Run forward pass (Only for the G) ; called by both functions <optimize_parameters> and <test>"""
        self.fake = self.netG(self.noise)  # G(noise)
        pass

    def backward_D(self):
        """Calculate GAN Loss for the D by:
            1st: Run a forward pass through D
            2nd: Calculate loss
            3rd: Backpropagate loss
        """
        # Train with all-fake batch
        pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Train with all-real batch
        pred_real = self.netD(self.real)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combine loss and calculate gradients
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """ Calculate GAN Loss for G by:
            1st: Calculate predictions of D when passing fake images
            2nd: Calculate loss (for G is that fake images are predicted as True)
            3rd: Backpropagate loss

        """
        # Train with all-fake batch
        pred_fake = self.netD(self.fake.detach())
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN.backward()


    def optimize_parameters(self):
        self.forward()
        # Update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to 0
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # Update G
        self.set_requires_grad(self.netD, False)  # D needs no gradients when training G
        self.optimizer_G.zero_grad()  # set G's gradients to 0
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update D's weights

        pass