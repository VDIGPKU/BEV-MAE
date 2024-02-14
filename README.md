# BEV-MAE: Bird's Eye View Masked Autoencoders for Point Cloud Pre-training in Autonomous Driving Scenarios

This is the official implementation of [BEV-MAE](https://arxiv.org/abs/2212.05758).



## Model

We release the pre-training weights of VoxelNet on Waymo dataset.

| pre-trained 3D backbone |      Dataset      |                           Weights                            |
| :---------------------: | :---------------: | :----------------------------------------------------------: |
|        VoxelNet         | Waymo (20% data)  | [Google_drive](https://drive.google.com/file/d/1S2a2uhmRPqWQ6LGcFHfw-Cdch1jdgY6U/view?usp=share_link) |
|        VoxelNet         | Waymo (full data) | [Google_drive](https://drive.google.com/file/d/1d8CXTSjFASXOo9UZ2fhmIyUObClEJ6od/view?usp=share_link) |

Our code is base on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (0.5 version). To use our pre-trained weights, please refer to [INSTALL.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for installation and follow the instructions in [GETTING_STARTED.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to train the model.



## Training

See the scripts in `tools/run.sh`



## Acknowledgements

BEV-MAE is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). It is also greatly inspired by the open-source code [Occupancy-MAE](https://github.com/chaytonmin/Occupancy-MAE).




## Citation

If BEV-MAE is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@inproceedings{lin2024bevmae,
  title={BEV-MAE: Bird's Eye View Masked Autoencoders for Point Cloud Pre-training in Autonomous Driving Scenarios},
  author={Lin, Zhiwei and Wang, Yongtao and Qi, Shengxiang and Dong, Nan and Yang, Ming-Hsuan},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  year={2024}
}
```



## Contact Us

If you have any problem about this work, please feel free to reach us out at `zwlin@pku.edu.cn`.

The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact `wyt@pku.edu.cn`.
