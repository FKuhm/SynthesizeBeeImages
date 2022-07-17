# Implementierung des Wasserstein-Losses in Anlehnung an https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py



# Import all required packages:
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as funct
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from modules_WGAN_step1 import *
from datetime import datetime
from glob import glob
import scipy
from scipy import linalg
import torchvision.utils as vutils
from torchvision.utils import save_image
from cleanfid import fid




# Search for an available GPU:
device = "cuda" if torch.cuda.is_available() else "cpu"


def trainGAN(
    lr_gen: float,
    lr_disc: float,
    latent_dim: int,
    number_epochs: int,
    batch_size: int,
    wandb: object,
    path_to_input_dir: str,
    calculate_fid: bool,
    n_critic: int,
    image_output_dir: str,
):
    """Training function. Creates generator and discriminator,
     conducts training and saves trained models.

    Args:
        lr_gen (float): Learning rate of generator.
        lr_disc (float): Learning rate of discriminator.
        latent_dim (int): Size of latent dimension.
        number_epochs (int): Number of epochs to train.
        batch_size (int): Batch size to use during training.
        wandb (object): wandb object.
    """





    # Config wandb run:
    wandb.config = {
        "learning_rate_gen": lr_gen,
        "learning_rate_disc": lr_disc,
        "latent_dim": latent_dim,
        "epochs": number_epochs,
        "batch_size": batch_size
    }


    # Create noise vector which is used throught the run to produce fake images:
    noise_vecs = torch.randn(size=(25, latent_dim, 1, 1), device=device)


    ####################################################################
    ####################################################################
    ####################################################################


    # Write all images stored in image_list to a directory:
    if not os.path.exists(image_output_dir):
        os.mkdir(image_output_dir)
    
    # Create new image dir in image_output_dir:
    os.mkdir(os.path.join(image_output_dir, wandb.run.name))


    # Define image transformations:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.CenterCrop(size = (350,350)),
         transforms.Resize(size = (64,64)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Standardize all channels

    # Create path to dataset directory:
    file_list = []
    image_list = []

    for f in glob(os.path.join(path_to_input_dir, "**", "*.png"), recursive=True):
        file_list.append(f)
    
    traindat = CustomDataset(file_list, transform)
    trainloader = torch.utils.data.DataLoader(
        traindat, batch_size=batch_size, shuffle=True, drop_last = True)

    # Initialize model (with normally distributed weights as suggested in paper):
    generator = Generator(latent_dim).to(device).apply(weights_init)
    discriminator = Discriminator().to(device).apply(weights_init)

    # Specify optimizers:
    optimizer_gen = torch.optim.Adam(params=generator.parameters(), lr=lr_gen, betas=(0.5, 0.9))
    optimizer_disc = torch.optim.Adam(params=discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.9))

    # Store errors in lists:
    gen_error = []
    disc_error = []


    ####################################################################
    ####################################################################
    ####################################################################
    # Calculate FID score at the beginning of the run:
    # If not already done in earlier runs, take 10000 real images, crop them and save them in another directory
    # for a fair comparison of FID score:
    if not( os.path.exists(os.path.join(image_output_dir, "temp_images_real")) and len(glob(os.path.join(image_output_dir, "temp_images_real", "*.png")))>0 ):

        # Create directory:
        if not os.path.exists(os.path.join(image_output_dir, "temp_images_real")):

            os.mkdir(os.path.join(image_output_dir, "temp_images_real"))

        else:
            # Delete all images contained:
            for old_image in glob(os.path.join(image_output_dir, "temp_images_real", "*.png")):
                os.remove(old_image)

        # Create second dataloader and save transformed images to folder:
        transform_fid = transforms.Compose(
            [transforms.ToTensor(),
            transforms.CenterCrop(size = (350,350)),
            # transforms.Resize(size = (64,64)), DO NOT RESIZE, otherwise the effect of the package is destroyed!
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Standardize all channels
        evaldat_fid = CustomDataset(file_list, transform_fid)
        trainloader_fid = torch.utils.data.DataLoader(
                        evaldat_fid, batch_size=batch_size, shuffle=True, drop_last = True)
        
        for u in range(0, int(np.ceil(10000/batch_size))):
            data = next(iter(trainloader_fid)).to(device)

            for i in range(0,batch_size):
                save_image(data[i], os.path.join(image_output_dir, "temp_images_real", "image_" + str(u*batch_size + i) + ".png"))



    # Create a new temporary folder if not already existing:
    if not os.path.exists(os.path.join(image_output_dir, "temp_images")):

        os.mkdir(os.path.join(image_output_dir, "temp_images"))

    # Generate 10000 images an save them in the newly created folder (delete all old ones if existent):
    for old_image in glob(os.path.join(image_output_dir, "temp_images", "*.png")):

        os.remove(old_image)

    for u in range(0,int(np.ceil(10000/batch_size))):

        images = generator(torch.randn(size = (batch_size, latent_dim, 1, 1), device = device))
        # Write all images to file:
        for i in range(0,batch_size):
            save_image(images[i], os.path.join(image_output_dir, "temp_images", "image_" + str(u*batch_size + i) + ".png"))

    # Now calculate FID score:
    fid_res = fid.compute_fid(fdir1 = os.path.join(image_output_dir, "temp_images_real"), fdir2 = os.path.join(image_output_dir, "temp_images"))

    wandb.log({"fid_score": fid_res}, step = -1)


    ### Conduct a graphical evaluation:
    with torch.no_grad():
        # Forward prop of noise vector:
        generated_images = generator(noise_vecs)

        # Plot images:
        fig = plt.figure()
        plt.imshow(np.transpose(vutils.make_grid(generated_images[:25], padding=4, normalize=True, nrow = 5).cpu(),(1,2,0)))
        plt.axis('off')

        # Save image in folder:
        fig.savefig(os.path.join(image_output_dir, wandb.run.name, "image_epoch_"+str(-1) + ".png"))
        images = wandb.Image(fig, caption="Epoch %i" % -1)

        wandb.log({"image_results": images}, step = -1, commit = True) # Commit directly because no info in epoch -1 is added anyway.

        # Delete image:
        plt.close()



    #######################
    # Iterate till number of epochs reached:
    #######################
    for epoch in range(0, number_epochs):

        # For each batch:
        for i, data in enumerate(trainloader, 0):

            # Push to GPU:
            data = autograd.Variable(data.to(device))

            #######################
            # Train discriminator
            #######################

            # Reset gradient od discriminator:
            discriminator.zero_grad()
            generator.zero_grad()

            data = torch.autograd.Variable(data)

            # Generator produces images out of noise vector:
            x_tilde = torch.normal(0,1, size=(data.shape[0], latent_dim, 1, 1), device = device)
            #x_tilde.requires_grad_(True)
            x_tilde = autograd.Variable(x_tilde, volatile = True) # volatile: Kennzeichnet inference mode & schaltet damit Gradientenberechnung für Generator aus!
            x_tilde = autograd.Variable(generator(x_tilde))

            # Use interpolation to generate an image:
            epsilon = torch.rand(size = (x_tilde.size()[0], 3, data.size()[2],data.size()[3]), device = device)
            epsilon = epsilon.expand(data.size())
            x_hat = (x_tilde.mul(epsilon) + data.mul(1-epsilon))
            x_hat = autograd.Variable(x_hat, requires_grad=True)

                
            # Compute gradient of loss function and add it to buffer:
            loss = wlossGrad(
                autograd.grad(inputs = x_hat, outputs = discriminator(x_hat), grad_outputs = torch.ones(size = (data.shape[0],1), device = device),
                create_graph=True, retain_graph=True, only_inputs=True)[0])
            
            loss.backward()
            loss_fake = discriminator(x_tilde)
            loss_fake = loss_fake.mean()
            loss_fake.backward(torch.tensor(1, device = device, dtype=torch.float))
            loss_real = discriminator(data).mean()
            loss_real.backward((-1)*torch.tensor(1, device = device, dtype=torch.float))

            loss = -loss_real + loss_fake
            

            # After each batch adjust parameters of critic:
            optimizer_disc.step()
                
            # After n_critic batches also adjust generators parameters:
            if i % n_critic == 0:

                for p in discriminator.parameters():
                    p.requires_grad = False
                
                generator.zero_grad()
                
                # Generate normally distributed noise:
                z = autograd.Variable(torch.normal(0,1, size=(batch_size, latent_dim, 1, 1), device = device))
                # Generator loss function
                loss_gen = discriminator(generator(z)).mean()
                # Compute gradient and update:
                loss_gen.backward((-1)*torch.tensor(1, device = device, dtype=torch.float))
                optimizer_gen.step()
                loss_gen = (-1)*loss_gen
            
                for p in discriminator.parameters():
                    p.requires_grad = True


            #######################
            # Evaluation: Save errors of both nets in a list:
            #######################
            gen_error.append(loss_gen.item())
            disc_error.append(loss.item())


        #######################
        # Epoch finished

        # Graphical evaluation:
        with torch.no_grad():
            # Forward prop of noise vector:
            generated_images = generator(noise_vecs)

            # Plot images:
            fig = plt.figure()
            plt.imshow(np.transpose(vutils.make_grid(generated_images[:25], padding=3, normalize=True, nrow = 5).cpu(),(1,2,0)))
            plt.axis('off')

            # Save image in folder:
            fig.savefig(os.path.join(image_output_dir, wandb.run.name, "image_epoch_"+str(epoch) + ".png"))
            images = wandb.Image(fig, caption="Epoch %i" % epoch)

            # Delete image:
            plt.close()
            



        # After each epoch log losses in W&B:
        if epoch % 1 == 0:

            wandb.log({"loss_gen": loss_gen.item(),
                    "loss_disc": loss.item(),
                    "image_results": images}, step = epoch)
            


            print("############################")
            print("Epoch %i completed" % (epoch))
            print("Loss of Generator: %f; loss of discriminator: %f" %
                (loss_gen.item(), loss.item()))
            print("############################")

        ########################################################################################################
        ########################################################################################################
        ########################################################################################################

        if (epoch < 10 and epoch%1 == 0) or (epoch >= 10 and epoch%10 == 0):

            # Generate 10000 images an save them in the newly created folder (delete all old ones if existent):
            for old_image in glob(os.path.join(image_output_dir, "temp_images", "*.png")):

                os.remove(old_image)

            for u in range(0,int(np.ceil(10000/batch_size))):

                images = generator(torch.randn(size = (batch_size, latent_dim, 1, 1), device = device))
                # Write all images to file:
                for i in range(0,batch_size):
                    save_image(images[i], os.path.join(image_output_dir, "temp_images", "image_" + str(u*batch_size + i) + ".png"))

            # Now calculate FID score:
            fid_res = fid.compute_fid(fdir1 = os.path.join(image_output_dir, "temp_images_real"), fdir2 = os.path.join(image_output_dir, "temp_images"))

            wandb.log({"fid_score": fid_res}, step = epoch)
            
            # Save a copy of current models to a file:
            torch.save(generator.state_dict(), os.path.join(image_output_dir, wandb.run.name, "generator_WGAN_030622_run_" + wandb.run.name + "_epoch_" + str(epoch) + ".pth"))
            torch.save(discriminator.state_dict(), os.path.join(image_output_dir, wandb.run.name, "discriminator_WGAN_030622_run_" + wandb.run.name + "_epoch_" + str(epoch) + ".pth"))

        # Log model at the end of run:
        if epoch == number_epochs-1:
            wandb.save(os.getcwd() + "/generator_WGAN_030622_run_" + wandb.run.name + "_epoch_" + str(epoch) + ".pth", policy = 'now')
            wandb.save(os.getcwd() + "/discriminator_WGAN_030622_run_" + wandb.run.name + "_epoch_" + str(epoch) + ".pth", policy = 'now')
        

    #######################
    # All epochs finished




def trainGAN_sweep(config, wandb, dir_path):
    # One Run:
    # lr_gen, lr_disc, latent_dim, number_epochs, batch_size, wandb, path_to_input_dir, calculate_fid, n_critic
    print("\n Start of Training\n")
    print(config)
    trainGAN(config.lr_gen, config.lr_disc, config.latent_dim, config.epochs, config.batch_size, wandb, dir_path, True, config.n_critic, "/home/ws/uwhbq/Bee_AEGAN/generated_bee_images")
    # Finish wandb run:
    # Not needed in a sweep run:
    #run.finish()


# Initialize Weights&Biases:
# Not needed in case of sweeps:
"""
wandb, run = initializeWandB(
    project_name="GAN_GymEnv_Sweep_v" + str(datetime.now())[0:10], entity="uwhbq")
"""


# Path to image directory:
dir_path = "/home/ws/uwhbq/Code/bee_GAN/bee_images/Users/Florian-Kuhm/Documents/bee_images/"


# Set up your default hyperparameters
hyperparameter_defaults = dict(
    lr_gen= [0.0001],
    lr_disc=[0.0001],
    latent_dim = [256],
    epochs=100,
    batch_size=[64],
    n_critic = [10],
)


# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
# Access all hyperparameter values through wandb.config
config = wandb.config

# Train models:
model = trainGAN_sweep(config, wandb, dir_path)








