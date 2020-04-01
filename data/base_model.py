"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

from abc import ABC, abstractmethod

from torch.utils import data


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """
        Initialize the class; save the options in the class
        Args:
            opt: (Option class) -- Stores all the experiment flags
        """
        self.opt = opt

    @abstractmethod
    def __len__(self):
        """
        Returns: The total number of images in the dataset
        """

    @abstractmethod
    def __getitem__(self, index):
        """
        Return a data point and its metadata information
        Args:
            index: a random integer for data indexing

        Returns: A dictionary of data with their names; It usually contains the data itself and its metadata information
        """
        pass

    @staticmethod
    def modify_commandline_options(parser, isTrain):
        """
        Add dataset-specific options, and rewrite default values for existing options.
        Args:
            parser: Original option parser
            isTrain: Whether training/test phase. Use flag to add training or test-specific options

        Returns: The modified parser
        """
        return parser









