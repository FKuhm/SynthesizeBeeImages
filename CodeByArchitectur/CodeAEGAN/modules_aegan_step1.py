
from turtle import forward
import torch
from torch import nn, tanh
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import wandb
import scipy.linalg
from torch import autograd


#########################################
### In AEGAN Code on Github:
### netRS: Encoder
### netG2: Decoder
### --> Both nets contain ResNet blocks!
#########################################

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
            
            
            ############# Block 0:
            # (latent_dimx1x1)
            nn.Upsample(scale_factor=4, mode = 'bilinear'),
            # (latent_dimx4x4)
            nn.Conv2d(in_channels=latent_dim,
                               out_channels=256, stride=1, kernel_size=3, padding=1),
            # (256x4x4)
            nn.LeakyReLU(0.1),
            ResNetBlock(input_channel_dim=256, batch_norm=False),
            nn.LeakyReLU(0.1),


            ############# Block 1:
            # (256x4x4)
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            # (256x8x8)
            nn.Conv2d(in_channels=256,
                               out_channels=128, stride=1, kernel_size=3, padding=1),
            # (128x8x8)
            nn.LeakyReLU(0.1),
            ResNetBlock(input_channel_dim=128, batch_norm=False),
            nn.LeakyReLU(0.1),


            ############# Block 2:
            # (128x8x8)
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            # (128x16x16)
            nn.Conv2d(in_channels=128,
                               out_channels=64, stride=1, kernel_size=3, padding=1),
            # (64x16x16)
            nn.LeakyReLU(0.1),
            ResNetBlock(input_channel_dim=64, batch_norm=False),
            nn.LeakyReLU(0.1),


            ############# Block 3:
            # (64x16x16)
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            # (64x32x32)
            nn.Conv2d(in_channels=64,
                               out_channels=64, stride=1, kernel_size=3, padding=1),
            # (64x32x32)
            nn.LeakyReLU(0.1),
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
            
            
            ############ Block 0:
            # (64x32x32)
            nn.Conv2d(in_channels=64, out_channels=128,
                      stride=1, kernel_size=3, padding=1),
            # (128x32x32)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128,
                      stride=2, kernel_size=4, padding=1),
            # (128x16x16)
            nn.LeakyReLU(negative_slope=0.1),

            ############ Block 1:
            # (128x16x16)
            nn.Conv2d(in_channels=128, out_channels=256,
                      stride=1, kernel_size=3, padding=1),
            # (256x16x16)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256,
                      stride=2, kernel_size=4, padding=1),
            # (256x8x8)
            nn.LeakyReLU(negative_slope=0.1),

            ############ Block 2:
            # (256x8x8)
            nn.Conv2d(in_channels=256, out_channels=512,
                      stride=1, kernel_size=3, padding=1),
            # (512x8x8)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=512,
                      stride=2, kernel_size=4, padding=1),
            # (512x4x4)
            nn.LeakyReLU(negative_slope=0.1),

            ############ Block 4:
            # (512x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024,
                      stride=1, kernel_size=3, padding=1),
            # (1024x4x4)
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      stride=1, kernel_size=4, padding=0),
            # (1024x1x1)
            nn.LeakyReLU(negative_slope=0.1),

            
            ############ Block 5:
            # (1024x1x1)
            nn.Conv2d(in_channels=1024, out_channels=1,
                      stride=1, kernel_size=1, padding=0),
            nn.Flatten(),
            # WICHTIG: Critic kann auf R abbilden!
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






#############################################################################


class Decoder(
    nn.Module
):
    """This class creates a deep convolutional generator using TransposedConvolutions.
    """

    def __init__(
        self,
    ):
        """Initialisierung des Generators:

        Args:
            latent_dim (int): Gibt die Vektorlaenge der latenten Repraesentation an.
        """

        # Aufruf des Konstruktors der Superklasse:
        super().__init__()

        # Initialisierung:
        self.decode = nn.Sequential(
            # Transpose Convolution Output Size = output = [(input-1)*stride]+kernel_size-2*padding_of_output
            
            ############# Block 0:
            # (64x32x32)
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            # (64x64x64)
            nn.Conv2d(in_channels=64,
                        out_channels=64, stride=1, kernel_size=3, padding=1, bias = False),
            # (64x64x64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            ResNetBlock(input_channel_dim = 64),

            ############# Block 1:
            # (64x64x64)
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            # (64x128x128)
            nn.Conv2d(in_channels=64,
                        out_channels=64, stride=1, kernel_size=3, padding=1, bias = False),
            # (64x128x128)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            ResNetBlock(input_channel_dim = 64),

            ############# Block 2:
            # (64x128x128)
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            nn.Conv2d(in_channels=64,
                        out_channels=64, stride=1, kernel_size=3, padding=1, bias = False),
            # (64x256x256)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            ResNetBlock(input_channel_dim = 64),

            ############# Block 3:
            # (64x256x256)
            nn.Conv2d(in_channels=64,
                        out_channels=3, stride=1, kernel_size=3, padding=1, bias = False),
            # (3x256x256)
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
        return self.decode(input)



### netRS uses ReflectPadding of (1,1,1,1) in ResNet Block!
### ResBlock can contain BatchNorm2d because Encoder is not trained with Wasserstein-Loss!

class Encoder(
    nn.Module
):
    """This class creates an encoder using convolutional layers.
    """

    def __init__(
        self
    ):

        # Aufruf des Konstruktors der Superklasse:
        super().__init__()

        # Initialisierung:
        self.encode = nn.Sequential(
            # H_out​=⌊(H_in​+2×padding−dilation×(kernel_size−1)−1​)/stride+1⌋

            ############# Block 0:
            # (3x256x256)
            nn.Conv2d(in_channels=3, out_channels=64,
                      stride=2, kernel_size=4, padding=1),
            # (64x128x128)
            nn.LeakyReLU(negative_slope=0.2),
            ResNetBlock(64),

            ############# Block 1:
            # (64x128x128)
            nn.Conv2d(in_channels=64, out_channels=64,
                      stride=2, kernel_size=4, padding=1),
            # (64x64x64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            ResNetBlock(64),


            ############# Block 2:
            # (64x64x64)
            nn.Conv2d(in_channels=64, out_channels=64,
                      stride=2, kernel_size=4, padding=1),
            # (64x32x32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            # (64x32x32)
            nn.Conv2d(in_channels=64, out_channels=64,
                      stride=1, kernel_size=3, padding=1),
            
        )

    def forward(
        self,
        input: torch.tensor
    ) -> torch.tensor:
        """Forward propagation.

        Args:
            input (torch.tensor): Tensor to propagate.

        Returns:
            torch.tensor: output of propagation.
        """
        return self.encode(input)



class ResNetBlock(nn.Module):

    def __init__(self, input_channel_dim, batch_norm = True) -> None:
        super().__init__()

        
        layers = []

        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(input_channel_dim, input_channel_dim, kernel_size=3, stride = 1, padding=0, bias = False))

        if batch_norm:
            layers.append(nn.BatchNorm2d(input_channel_dim))

        layers.append(nn.ReLU())
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(input_channel_dim, input_channel_dim, kernel_size=3, stride = 1, padding=0, bias = False))

        if batch_norm:
            layers.append(nn.BatchNorm2d(input_channel_dim))
    
        self.propagate = nn.Sequential(*layers)

    def forward(self, input):

        return input + self.propagate(input)






