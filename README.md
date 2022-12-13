# BEV-MAE: Bird's Eye View Masked Autoencoders for Outdoor Point Cloud Pre-training

This is the official implementation of BEV-MAE. The training code is coming soon.



## Model

We release the pre-training weights of VoxelNet on Waymo dataset.

| pre-trained 3D backbone |      Dataset      |                           Weights                            |
| :---------------------: | :---------------: | :----------------------------------------------------------: |
|        VoxelNet         | Waymo (20% data)  | [Google_drive](https://drive.google.com/file/d/1S2a2uhmRPqWQ6LGcFHfw-Cdch1jdgY6U/view?usp=share_link) |
|        VoxelNet         | Waymo (full data) | [Google_drive](https://drive.google.com/file/d/1d8CXTSjFASXOo9UZ2fhmIyUObClEJ6od/view?usp=share_link) |

Our code is base on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). To use our pre-trained weights, please refer to [INSTALL.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for installation and follow the instructions in [GETTING_STARTED.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to train the model.



## Contact Us

If you have any problem about this work, please feel free to reach us out at `zwlin@pku.edu.cn`.
