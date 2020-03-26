import os
from abc import ABC, abstractmethod
import torch
import network


class BaseModel(ABC):
    """
    Abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five methods:
    -- <__init__>: Class initializer
    -- <set_input>: Unpack data from dataset and apply preprocessing
    -- <forward>: Produce intermediate results
    -- <optimize_parameters>: Calculate losses, gradients, and upgrade network weights
    -- <modify_commandline_options>: (optionally) add model-specific options and set default options
    """

    def __add__(self, opt):
        """ Initialize the BaseModel class

        Args:
            opt: (Option class) -- Stores all the experiment flags

        When creating a custom class:
            1st: call <BaseModel.__init__(self, opt)>
            2nd: define four lists:
                self.loss_names (str list): training losses to display and save
                self.model_names (str lists): networks used in our training
                self.visual_names (str lists): images to display and save
                self.optimizers (optimizer list): define and initialize optimizers

        Returns:

        """

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids \
            else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True

        # To be defined when creating a custom class
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0 # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add model-specific options, and rewrite default values for existing options
        Args:
            parser: original option parser
            is_train: whether training/test phase. Use flag to add training or test-specific options

        Returns:
            The modified parser
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps
        Args:
            input: (dict): includes the data itself and its metadata information

        Returns:

        """
        pass

    @abstractmethod
    def forward(self):
        """ Run forward pass; called by both functions <optimize_parameters> and <test> """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """ Calculate losses, gradients, and update network weights; called every training iteration """
    pass

    def setup(self, opt):
        """
        Load and print networks; print schedulers
        Args:
            opt: (option class)
        """
        if self.isTrain:
            self.schedulers = [network.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
