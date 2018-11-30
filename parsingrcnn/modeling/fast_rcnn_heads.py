import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils


class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    loss_cls = F.cross_entropy(cls_score, rois_label)

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)

    return loss_cls, loss_bbox, accuracy_cls


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


class roi_2mlp_head_adp(nn.Module):
    """Add a ReLU MLP with two hidden layers and adp."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.num_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        self.fc1 = nn.ModuleList()
        for i in range(self.num_levels):
            self.fc1.append(nn.Sequential(
                nn.Linear(dim_in * roi_size ** 2, hidden_dim),
                nn.ReLU(inplace=True)
            ))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(inplace=True))

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)

        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_levels):
            mapping_to_detectron.update({
                'fc1.{}.0.weight'.format(i): 'fc6_{}_w'.format(i),
                'fc1.{}.0.bias'.format(i): 'fc6_{}_b'.format(i)
            })

        mapping_to_detectron.update({
            'fc2.0.weight': 'fc7_w',
            'fc2.0.bias': 'fc7_b',
        })
        return mapping_to_detectron, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            panet=True
        )
        batch_size = x[0].size(0)
        for i in range(len(x)):
            x[i] = self.fc1[i](x[i].view(batch_size, -1))
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        x = self.fc2(x)

        return x


class roi_2mlp_head_gn(nn.Module):
    """Add a ReLU MLP with two hidden layers and GroupNorm."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Sequential(nn.Linear(dim_in * roi_size ** 2, hidden_dim),
                                 nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                                              eps=cfg.GROUP_NORM.EPSILON))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                                              eps=cfg.GROUP_NORM.EPSILON))

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)

        self.apply(_init)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.0.weight': 'fc6_w',
            'fc1.0.bias': 'fc6_b',
            'fc1.1.weight': 'fc6_gn_s',
            'fc1.1.bias': 'fc6_gn_b',
            'fc2.0.weight': 'fc7_w',
            'fc2.0.bias': 'fc7_b',
            'fc2.1.weight': 'fc7_gn_s',
            'fc2.1.bias': 'fc7_gn_b',
        }
        return detectron_weight_mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


class roi_2mlp_head_gn_adp(nn.Module):
    """Add a ReLU MLP with two hidden layers, adp, and GroupNorm."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.num_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        self.fc1 = nn.ModuleList()
        for i in range(self.num_levels):
            self.fc1.append(nn.Sequential(
                nn.Linear(dim_in * roi_size ** 2, hidden_dim),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                            eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                                              eps=cfg.GROUP_NORM.EPSILON),
                                 nn.ReLU(inplace=True))

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)

        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_levels):
            mapping_to_detectron.update({
                'fc1.{}.0.weight'.format(i): 'fc6_{}_w'.format(i),
                'fc1.{}.0.bias'.format(i): 'fc6_{}_b'.format(i),
                'fc1.{}.1.weight'.format(i): 'fc6_{}_gn_s'.format(i),
                'fc1.{}.1.bias'.format(i): 'fc6_{}_gn_b'.format(i)
            })

        mapping_to_detectron.update({
            'fc2.0.weight': 'fc7_w',
            'fc2.0.bias': 'fc7_b',
            'fc2.1.weight': 'fc7_gn_s',
            'fc2.1.bias': 'fc7_gn_b',
        })
        return mapping_to_detectron, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            panet=True
        )
        batch_size = x[0].size(0)
        for i in range(len(x)):
            x[i] = self.fc1[i](x[i].view(batch_size, -1))
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        x = self.fc2(x)

        return x

    
class roi_mask_fusion_2mlp_head(nn.Module):
    """RoIMaskAlign + RoIAlign by adding a ReLU MLP with two hidden layers."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1_1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.fc1_2 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1_1.weight)
        init.constant_(self.fc1_1.bias, 0)
        mynn.init.XavierFill(self.fc1_2.weight)
        init.constant_(self.fc1_2.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1_1.weight': 'fc6_1_w',
            'fc1_1.bias': 'fc6_1_b',
            'fc1_2.weight': 'fc6_2_w',
            'fc1_2.bias': 'fc6_2_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x1 = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method='RoIAlign',
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        
        x2 = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,  # RoIMaskAlign
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        
        batch_size = x1.size(0)
        x1 = self.fc1_1(x1.view(batch_size, -1))
        x2 = self.fc1_2(x2.view(batch_size, -1))
        
        x = F.relu(x1 + x2, inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x
    
    
class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)

        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i * 2): 'head_conv%d_w' % (i + 1),
                'convs.%d.bias' % (i * 2): 'head_conv%d_b' % (i + 1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)

        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i * 3): 'head_conv%d_w' % (i + 1),
                'convs.%d.weight' % (i * 3 + 1): 'head_conv%d_gn_s' % (i + 1),
                'convs.%d.bias' % (i * 3 + 1): 'head_conv%d_gn_b' % (i + 1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head_adp(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm and adp"""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS - 1):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)
        self.num_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        self.conv1_head = nn.ModuleList()
        for i in range(self.num_levels):
            self.conv1_head.append(nn.Sequential(
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ))

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)

        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(self.num_levels):
            mapping.update({
                'conv1_head.{}.0.weight'.format(i): 'adp{}_head_conv_w'.format(i + 1),
                'conv1_head.{}.1.weight'.format(i): 'adp{}_head_conv_gn_w'.format(i + 1),
                'conv1_head.{}.1.bias'.format(i): 'adp{}_head_conv_gn_s'.format(i + 1)
            })

        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS - 1):
            mapping.update({
                'convs.%d.weight' % (i * 3): 'head_conv%d_w' % (i + 1),
                'convs.%d.weight' % (i * 3 + 1): 'head_conv%d_gn_s' % (i + 1),
                'convs.%d.bias' % (i * 3 + 1): 'head_conv%d_gn_b' % (i + 1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            panet=True
        )
        for i in range(len(x)):
            x[i] = self.conv1_head[i](x[i])
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        batch_size = x.size(0)

        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x
