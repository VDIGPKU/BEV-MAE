import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from ....utils import common_utils
from ....utils import loss_utils
from ....utils.spconv_utils import replace_feature, spconv

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFEMAE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.mask_ratio = model_cfg.MASKED_RATIO

        self.sparse_shape = self.model_cfg.GRID_SIZE[::-1]
        self.decoder = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.coor_conv = nn.Conv2d(384, 3*20, 1)
        self.num_conv = nn.Conv2d(384, 1, 1)

        down_factor = 1
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)
        self.coor_loss = loss_utils.MaskChamferDistance()
        self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1,11))
        self.forward_re_dict = {}

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # print(features.size()) # 72865, 20, 11

        ################### get mask index #####################

        coor_down_sample = coords.int().detach().clone()
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:]//self.down_factor
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
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index)
        ids_keep = ids_keep.bool()

        ids_mask = ~ids_keep

        ###########################################################
        batch_size = batch_dict['batch_size']
        ####################### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask,...], coords[ids_mask,...]

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
        bev_mask = pts_mask.max(dim=1)[0].max(dim=1)[0]
        self.forward_re_dict['gt_mask'] = bev_mask
        
        ####################### gt num
        pts_gt_num = spconv.SparseConvTensor(
            voxel_num_points.view(-1, 1).detach(),
            coords.int(),
            self.sparse_shape,
            batch_size
        ).dense()
        bs, _, d, h, w = pts_gt_num.shape

        pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w))
        pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor**2

        pts_gt_num = pts_gt_num.detach()
        self.forward_re_dict['gt_num'] = pts_gt_num

        #################### gt coord

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

        #################################

        voxel_features_partial, voxel_coords_partial = features[ids_keep,:], coords[ids_keep,:]

        average_features = self.mask_token.repeat(voxel_features_mask.size(0), voxel_features_mask.size(1), 1)

        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

    def decode_feat(self, feats):

        x = self.decoder(feats)

        coor = self.coor_conv(x)
        num = self.num_conv(x)

        return coor, num
    

    def get_loss(self, batch_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        feats = batch_dict['spatial_features_2d']
        pred_coor, pred_num = self.decode_feat(feats)

        gt_coor = self.forward_re_dict['gt_coor'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask'].detach()

        gt_num = self.forward_re_dict['gt_num'].detach()

        gt_mask = self.forward_re_dict['gt_mask'].detach()

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