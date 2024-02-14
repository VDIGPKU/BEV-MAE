from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from ...utils import loss_utils
from .spconv_backbone import post_act_block

import matplotlib.pyplot as plt
import cv2


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        # bias = norm_fn is not None
        # bias = False
        bias = True
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class BEV_MAE_res_dbscan(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        
        self.num_point_features = 16                

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv = nn.Conv2d(256, 3*20, 1)
        self.num_conv = nn.Conv2d(256, 1, 1)

        down_factor = 8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)
        # self.vx = voxel_size[0] * down_factor
        # self.vy = voxel_size[1] * down_factor
        # self.vz = voxel_size[2] * down_factor
        voxel_size = model_cfg.VOXEL_SIZE
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0] 
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1,5))

        self.forward_re_dict = {}
        
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # pred = self.forward_re_dict['pred']
        # target = self.forward_re_dict['target']
        pred_coor = self.forward_re_dict['pred_coor']
        gt_coor = self.forward_re_dict['gt_coor'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask'].detach()

        pred_num = self.forward_re_dict['pred_num']
        gt_num = self.forward_re_dict['gt_num'].detach()


        gt_mask = self.forward_re_dict['gt_mask'].detach()
        # loss = self.criterion(pred, target)
        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        loss = loss_num + loss_coor

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
        }

        return loss, tb_dict
    
    def get_num_loss(self, pred, target, mask):
        bs = pred.shape[0]
        loss = self.num_loss(pred, target).squeeze()
        if bs == 1:
            loss = loss.unsqueeze(dim=0)

        assert loss.size() == mask.size()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def get_coor_loss(self, pred, target, mask, chamfer_mask):
        
        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1)
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]


        pred = pred.reshape(-1, 3, 20).permute(0, 2, 1)
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss
    
    def decode_feat(self, feats, mask=None):
        # feats = feats[mask]
        if mask is not None:
            bs, c, h, w = feats.shape
            # print(mask.shape)
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder(feats)
        bs, c, h, w = x.shape
        # x = feats
        coor = self.coor_conv(x)
        num = self.num_conv(x)
        # x = x.reshape(bs, )
        return coor, num


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        # self.draw_point(voxel_features.cpu().numpy(), './imgs/ori.jpg')
        # print(voxel_features.size())
        coor_down_sample = coors.int().detach().clone()
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:]//(self.down_factor * self.grid)
        coor_down_sample[:, 1] = coor_down_sample[:, 1]//(coor_down_sample[:, 1].max()*2)

        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0)

        select_ratio = 1 - self.mask_ratio # ratio for select voxel
        nums = unique_coor_down_sample.shape[0]
        
        len_keep = int(nums * select_ratio)

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)

        keep = ids_shuffle[:len_keep]

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach()
        unique_keep_bool[keep] = 1
        # unique_mask_bool = unique_mask_bool.bool()
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index)
        ids_keep = ids_keep.bool()

        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask,:], coors[ids_mask,:]

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0],1).to(voxel_features_mask.device).detach()
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()

        pts_mask = pts_mask.detach()
        point_mask = pts_mask.clone()

        pts_mask = self.unshuffle(pts_mask)
        # print(pts_mask.shape)
        bev_mask = pts_mask.squeeze().max(dim=1)[0]
        self.forward_re_dict['gt_mask'] = bev_mask
        
        #### gt num
        pts_gt_num = spconv.SparseConvTensor(
            num_points.view(-1, 1).detach(),
            coors.int(),
            self.sparse_shape,
            batch_size
        ).dense()
        bs, _, d, h, w = pts_gt_num.shape
        # print('num shape 1', pts_gt_num.shape)
        pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w))
        pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor**2
        # pts_gt_num = pts_gt_num.mean(dim=1, keepdim=True)
        # print(pts_gt_num[pts_gt_num>0].float().mean())
        # print('num shape 2', pts_gt_num.shape)
        pts_gt_num = pts_gt_num.detach()
        self.forward_re_dict['gt_num'] = pts_gt_num

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep,:], coors[ids_keep,:]
        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1)

        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        feats = out.dense()
        bs, c, d, h, w = feats.shape
        feats = feats.reshape(bs, -1, h, w)

        pred_coor, pred_num = self.decode_feat(feats)
        self.forward_re_dict['pred_coor'] = pred_coor
        self.forward_re_dict['pred_num'] = pred_num

        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], batch_dict['voxel_coords_bev'], 
        
        f_center = torch.zeros_like(voxels_large[:, :, :3])

        f_center[:, :, 0] = (voxels_large[:, :, 0] - (coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz

        voxel_count = f_center.shape[1]
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0)
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center)
        f_center *= mask_num

        sparse_shape = [1, self.sparse_shape[1]//self.down_factor, self.sparse_shape[2]//self.down_factor,]

        chamfer_mask = spconv.SparseConvTensor(
            mask_num.squeeze().detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()

        self.forward_re_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense() # 

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w)
        self.forward_re_dict['gt_coor'] = pts_gt_coor

        voxels_large_bev_one = torch.ones(voxels_large.shape[0],1).to(voxels_large.device).detach()
        pts_mask_bev = spconv.SparseConvTensor(
            voxels_large_bev_one,
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()

        pts_mask_bev = pts_mask_bev.detach()
        # print(pts_mask_bev.shape)
        pts_mask_bev = pts_mask_bev[:, 0, 0,...]
        # print(pts_mask_bev.shape)
        self.forward_re_dict['gt_mask'] *= pts_mask_bev
        # exit(0)

        return batch_dict
    
    def draw_point(self, points, path, size=20, s=2):
        plt.figure(figsize=(size, size))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        ax.axis('off')
        # points = points[(points[:, 0]>-40) & (points[:, 0]<40) &(points[:, 0]>-40) &(points[:, 1]<40)]
        if points.shape[1]<3:
            # ax.scatter(points[:, 1], points[:, 0], s=0.5, c='b', alpha=0.5)
            ax.scatter(points[:, 1], points[:, 0], s=s, c='b', alpha=0.5)
        else:
            # ax.scatter(points[:, 1], points[:, 0], s=0.5, c=points[:, 2], alpha=0.5)
            ax.scatter(points[:, 1], points[:, 0], s=s, c=points[:, 2], alpha=0.5)
        print(path)
        plt.savefig(path)

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator