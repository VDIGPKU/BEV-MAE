

# pretraining
bash ./scripts/dist_train_bevmae.sh 4 --cfg_file cfgs/waymo_models/bevmae_waymo_voxel_20ep.yaml --extra_tag $NAME$ --ckpt_save_interval 1

# finetuning
bash ./scripts/dist_train.sh 4 --cfg_file cfgs/waymo_models/centerpoint_2xlr.yaml --pretrained_model $Pretrain_weights$ --ckpt_save_interval 1

