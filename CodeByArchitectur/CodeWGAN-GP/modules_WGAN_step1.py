
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import wandb
import scipy.linalg
from torch import autograd



def initializeWandB(
        project_name: str,
        entity: str) -> object:
    """This method initializes Weights&Biases.

    Args:
        project_name (str): Name of project to which runs will be added.
        entity (_type_): username

    Returns:
        object: wandb object
    """

    wandb.login()
    run = wandb.init(project=project_name, entity=entity)
    return wandb, run


class Flatten(
    nn.Module
):
    """Modified version of torchs flatten layer.
    """

    def __init__(
        self
    ):
        super().__init__()

    def forward(
            self,
            inp: torch.tensor) -> torch.tensor:
        """Reshapes a multidimensional tensor to a vector.

        Args:
            inp (torch.tensor): Tensor to transform.

        Returns:
            torch.tensor: Flattened Tensor.
        """
        return torch.reshape(inp, (inp.shape[0], torch.prod(torch.as_tensor(inp.shape[1:]))))


class Generator(
    nn.Module
):
    """This class creates a deep convolutional generator using TransposedConvolutions.
    """

    def __init__(
        self,
        latent_dim: int
    ):
        """Initialisierung des Generators:

        Args:
            latent_dim (int): Gibt die Vektorlaenge der latenten Repraesentation an.
        """

        # Aufruf des Konstruktors der Superklasse:
        super().__init__()

        # Initialisierung:
        self.generate = nn.Sequential(
            # Transpose Convolution Output Size = output = [(input-1)*stride]+kernel_size-2*padding_of_output
            # H_out​=⌊(H_in​+2×padding−dilation×(kernel_size−1)−1​)/stride+1⌋
            # (latent_dimx1x1)
            nn.Upsample(scale_factor=4, mode = 'bilinear'),
            # (latent_dimx4x4)
            nn.Conv2d(in_channels=latent_dim,
                               out_channels=128, stride=1, kernel_size=5, padding=2),
            # (128x4x4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            # (128x8x8)
            nn.Conv2d(
                in_channels=128, out_channels=64, stride=1, kernel_size=5, padding=2),
            # (64x8x8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            # (64x16x16)
            nn.Conv2d(
                in_channels=64, out_channels=32, stride=1, kernel_size=5, padding=2),
            # (32x16x16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            # (32x32x32)
            nn.Conv2d(
                in_channels=32, out_channels=16, stride=1, kernel_size=5, padding=2),
            # (16x32x32)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
             # (16x64x64)
            nn.Conv2d(
                in_channels=16, out_channels=8, stride=1, kernel_size=5, padding=2),
            # (8x64x64)
            nn.Conv2d(
                in_channels=8, out_channels=3, stride=1, kernel_size=3, padding=1),
            # (3x64x64)
            # WICHTIG: Verwendung von tanh, um die Pixelwerte wieder in den Normierungsbereich [-1,1] zu bringen.
            nn.Tanh(),
        )

    def forward(
        self,
        input: torch.tensor
    ) -> torch.tensor:
        """Forward propagation.

        Args:
            input (torch.tensor): Zu propagierender Tensor.

        Returns:
            torch.tensor: Resultierender Tensor.
        """
        # Rufe Methode generate auf:
        return self.generate(input)



class Discriminator(
    nn.Module
):
    """This class creates a deep convolutional discriminator using convolutional layer.
    """

    def __init__(
        self
    ):

        # Aufruf des Konstruktors der Superklasse:
        super().__init__()

        # Initialisierung:
        self.dicriminate = nn.Sequential(
            # Bias not needed due to Batch Normalization!
            # H_out​=⌊(H_in​+2×padding−dilation×(kernel_size−1)−1​)/stride+1⌋
            # (3x64x64)
            nn.Conv2d(in_channels=3, out_channels=128,
                      stride=2, kernel_size=4, padding=1),
            # (128x32x32)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128,
                      stride=1, kernel_size=3, padding=1),
            # (128x32x32)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256,
                      stride=2, kernel_size=4, padding=1),
            # (256x16x16)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256,
                      stride=1, kernel_size=3, padding=1),
            # (256x16x16)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512,
                      stride=2, kernel_size=4, padding=1),
            # (512x8x8)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=512,
                      stride=1, kernel_size=3, padding=1),
            # (512x8x8)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024,
                      stride=2, kernel_size=4, padding=1),
            # (1024x4x4)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      stride=1, kernel_size=3, padding=1),
            # (1024x4x4)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      stride=1, kernel_size=4, padding=0),
            # (1024x1x1)
            nn.Flatten(),
            # (1024x1)
            nn.Linear(in_features = 1024, out_features = 1)  # WICHTIG: Critic kann auf R abbilden!
        )

    def forward(
        self,
        input: torch.tensor
    ) -> torch.tensor:
        """Forward propagation.

        Args:
            input (torch.tensor): Tensor to propagate.

        Returns:
            torch.tensor: Resulting probability.
        """
        return self.dicriminate(input)


def weights_init(
    m
):
    """Method to initialize discriminators and generators weights by normally distributed random values.

    Args:
        m: Torch layer added to sequential model.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CustomDataset(
    torch.utils.data.Dataset
):
    """Custom Dataset class to be able to read in images without necessity to specify labels.
    """

    def __init__(
        self,
        image_paths: list,
        transform:  torchvision.transforms = None
    ):
        """Initialization.

        Args:
            image_paths (list): List of image paths.
            transform (torchvision.transforms, optional): Transformation operations specified. Defaults to None.
        """
        self.image_paths = image_paths
        self.transform = transform

    def get_class_label(
        self,
        image_name: str
    ) -> str:
        """Overwrite method to return an empty string as label.

        Args:
            image_name (str): File name of image.

        Returns:
            str: empty string because labels are not needed in case of GANs.
        """
        return ""

    def __getitem__(
        self,
        index
    ):
        """Method which reads in an image, converts it to RGB and applies transformations stored in transform.

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Read out correct path to image:
        image_path = self.image_paths[index]
        # Import it via PIL:
        x = Image.open(image_path).convert('RGB')
        # Apply all transformations:
        if self.transform is not None:
            x = self.transform(x)
        # Return result:
        return x

    def __len__(self):
        """Method to receive number of images stored in dataset.
        """
        return len(self.image_paths)



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



##############################
### Define Wasserstein loss function:
### IMPORTANT: Pytorch minimizes --> (-1)*function
##############################
def wlossGrad(dx_hat_grad):
    
    return 10*((dx_hat_grad.norm(2, dim=1) - 1) ** 2).mean()

