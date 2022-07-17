# Content of this repository

In this repository, all codes and images used in my seminar paper can be found.

The folder **Preprocessing** contains one script which was used to crop bee images from videos to create a data set on which to train GANs.

The folder **CodeByArchitecture** contains the scripts corresponding to the respective architecture. The FastGAN code has been forked from [this repository](https://github.com/odegeasslbc/FastGAN-pytorch) and the creation of the dataloader has been changed due to the special folder structure of the bee images dataset. The scripts regarding the AEGAN and WGAN-GP architectures require the availability of a Weights&Biases account, as the images and losses are logged through it.

The folder **ImageCreationScriptsAndFigures** contains some images shown in the seminar work and partly the scripts necessary for their creation. In most cases, the image manipulation program Gimp was used. For three images OmniGraffle has been chosen. Images characterizing the training progress of a model are not included in the folder, but must be downloaded from the cloud analogous to the pth files (see below) due to their large number and corresponding size.

Because of the size of the pth files of the trained models, they have been uploaded to the cloud *GigaMove* of the RWTH Aachen University[^1]. On this platform, they are still available for download until 2022-07-31.

All FID scores has been calculated via [this package](https://github.com/GaParmar/clean-fid).

[^1]: Zipped file can be downloaded from [this link](https://gigamove.rwth-aachen.de/de/download/2d8e5e00e8acf21ac99cfb37fa0e909f) (approx. 66.06 GB).