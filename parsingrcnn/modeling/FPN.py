import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from parsingrcnn.core.config import cfg
import parsingrcnn.utils.net as net_utils
import parsingrcnn.modeling.ResNet as ResNet
import parsingrcnn.modeling.backbone.Air as Air
import parsingrcnn.modeling.backbone.Attir as Attir
import parsingrcnn.modeling.backbone.CrossNet as CrossNet
import parsingrcnn.modeling.backbone.DetNet as DetNet
import parsingrcnn.modeling.backbone.DetAir as DetAir
import parsingrcnn.modeling.backbone.MnasNet as MnasNet
import parsingrcnn.modeling.backbone.Inception_v3 as Inception_v3
import parsingrcnn.modeling.backbone.MobileNet_v1 as MobileNet_v1
import parsingrcnn.modeling.backbone.MobileNet_v2 as MobileNet_v2
import parsingrcnn.modeling.backbone.ResNet_priv as ResNet_priv
import parsingrcnn.modeling.backbone.VGG as VGG
from parsingrcnn.modeling.generate_anchors import generate_anchors
from parsingrcnn.modeling.generate_proposals import GenerateProposalsOp
from parsingrcnn.modeling.collect_and_distribute_fpn_rpn_proposals import CollectAndDistributeFpnRpnProposalsOp
import parsingrcnn.nn as mynn


# Lowest and highest pyramid levels in the backbone network. For FPN, we assume
# that all networks have 5 spatial reductions, each by a factor of 2. Level 1
# would correspond to the input image, hence it does not make sense to use it.
# LOWEST_BACKBONE_LVL = 2  # E.g., "conv2"-like level
# HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level


# ---------------------------------------------------------------------------- #
# FPN with ResNet
# ---------------------------------------------------------------------------- #


def fpn_ResNet50_conv5_body():
    return fpn(
        ResNet.ResNet50_conv5_body, fpn_level_info_ResNet50_conv5()
    )


def fpn_ResNet50_conv5_body_bup():
    return fpn(
        ResNet.ResNet50_conv5_body, fpn_level_info_ResNet50_conv5(),
        panet_buttomup=True
    )


def fpn_ResNet50_conv5_P2only_body():
    return fpn(
        ResNet.ResNet50_conv5_body,
        fpn_level_info_ResNet50_conv5(),
        P2only=True
    )


def fpn_ResNet101_conv5_body():
    return fpn(
        ResNet.ResNet101_conv5_body, fpn_level_info_ResNet101_conv5()
    )


def fpn_ResNet101_conv5_P2only_body():
    return fpn(
        ResNet.ResNet101_conv5_body,
        fpn_level_info_ResNet101_conv5(),
        P2only=True
    )


def fpn_ResNet152_conv5_body():
    return fpn(
        ResNet.ResNet152_conv5_body, fpn_level_info_ResNet152_conv5()
    )


def fpn_ResNet152_conv5_P2only_body():
    return fpn(
        ResNet.ResNet152_conv5_body,
        fpn_level_info_ResNet152_conv5(),
        P2only=True
    )


# ---------------------------------------------------------------------------- #
# FPN with ResNet_priv
# ---------------------------------------------------------------------------- #

def fpn_ResNet26_priv_conv5_body():
    return fpn(
        ResNet_priv.ResNet26_conv5_body, fpn_level_info_ResNet26_priv_conv5()
    )


def fpn_ResNet26_priv_conv5_P2only_body():
    return fpn(
        ResNet_priv.ResNet26_conv5_body,
        fpn_level_info_ResNet26_priv_conv5(),
        P2only=True
    )


def fpn_ResNet50_priv_conv5_body():
    return fpn(
        ResNet_priv.ResNet50_conv5_body, fpn_level_info_ResNet50_priv_conv5()
    )


def fpn_ResNet50_priv_conv5_P2only_body():
    return fpn(
        ResNet_priv.ResNet50_conv5_body,
        fpn_level_info_ResNet50_priv_conv5(),
        P2only=True
    )


def fpn_ResNet101_priv_conv5_body():
    return fpn(
        ResNet_priv.ResNet101_conv5_body, fpn_level_info_ResNet101_priv_conv5()
    )


def fpn_ResNet101_priv_conv5_P2only_body():
    return fpn(
        ResNet_priv.ResNet101_conv5_body,
        fpn_level_info_ResNet101_priv_conv5(),
        P2only=True
    )


def fpn_ResNet152_priv_conv5_body():
    return fpn(
        ResNet_priv.ResNet152_conv5_body, fpn_level_info_ResNet152_priv_conv5()
    )


# ---------------------------------------------------------------------------- #
# FPN with Air
# ---------------------------------------------------------------------------- #

def fpn_Air26_conv5_body():
    return fpn(
        Air.Air26_conv5_body, fpn_level_info_ResNet26_priv_conv5(), net_type='air'
    )


def fpn_Air26_conv5_P2only_body():
    return fpn(
        Air.Air26_conv5_body,
        fpn_level_info_ResNet26_priv_conv5(),
        P2only=True,
        net_type='air'
    )


def fpn_Air50_conv5_body():
    return fpn(
        Air.Air50_conv5_body, fpn_level_info_Air50_conv5(), net_type='air'
    )


def fpn_Air50_conv5_P2only_body():
    return fpn(
        Air.Air50_conv5_body,
        fpn_level_info_Air50_conv5(),
        P2only=True,
        net_type='air'
    )


def fpn_Air101_conv5_body():
    return fpn(
        Air.Air101_conv5_body, fpn_level_info_Air101_conv5(), net_type='air'
    )


def fpn_Air101_conv5_P2only_body():
    return fpn(
        Air.Air101_conv5_body,
        fpn_level_info_Air101_conv5(),
        P2only=True,
        net_type='air'
    )


def fpn_Air152_conv5_body():
    return fpn(
        Air.Air152_conv5_body, fpn_level_info_Air152_conv5(), net_type='air'
    )


# ---------------------------------------------------------------------------- #
# FPN with Attir
# ---------------------------------------------------------------------------- #

def fpn_Attir50_conv5_body():
    return fpn(
        Attir.Attir50_conv5_body, fpn_level_info_Attir50_conv5(), net_type='attir'
    )


def fpn_Attir101_conv5_body():
    return fpn(
        Attir.Attir101_conv5_body, fpn_level_info_Attir101_conv5(), net_type='attir'
    )


# ---------------------------------------------------------------------------- #
# FPN with DetNet
# ---------------------------------------------------------------------------- #

def fpn_DetNet59_conv6_body():
    return fpn(
        DetNet.DetNet59_conv6_body, fpn_level_info_DetNet59_conv6(), net_type='detnet'
    )


def fpn_DetNet110_conv6_body():
    return fpn(
        DetNet.DetNet110_conv6_body, fpn_level_info_DetNet110_conv6(), net_type='detnet'
    )


# ---------------------------------------------------------------------------- #
# FPN with DetAir
# ---------------------------------------------------------------------------- #

def fpn_DetAir32_conv6_body():
    return fpn(
        DetAir.DetAir32_conv6_body, fpn_level_info_DetAir32_conv6(), net_type='detair'
    )


def fpn_DetAir59_conv6_body():
    return fpn(
        DetAir.DetAir59_conv6_body, fpn_level_info_DetAir59_conv6(), net_type='detair'
    )


def fpn_DetAir110_conv6_body():
    return fpn(
        DetAir.DetAir110_conv6_body, fpn_level_info_DetAir110_conv6(), net_type='detair'
    )


def fpn_DetAir161_conv6_body():
    return fpn(
        DetAir.DetAir161_conv6_body, fpn_level_info_DetAir161_conv6(), net_type='detair'
    )


# ---------------------------------------------------------------------------- #
# FPN with CrossNet
# ---------------------------------------------------------------------------- #

def fpn_CrossNet_C5_body():
    return fpn(
        CrossNet.CrossNet_C5_body, fpn_level_info_CrossNet_C5(), net_type='cross'
    )


# ---------------------------------------------------------------------------- #
# FPN with MobileNet_v1
# ---------------------------------------------------------------------------- #

def fpn_MobileNet_v1_C5_body():
    return fpn(
        MobileNet_v1.MobileNet_v1_C5_body, fpn_level_info_MobileNet_v1_C5(), net_type='mobv1_'
    )


# ---------------------------------------------------------------------------- #
# FPN with MobileNet_v2
# ---------------------------------------------------------------------------- #

def fpn_MobileNet_v2_C5_body():
    return fpn(
        MobileNet_v2.MobileNet_v2_C5_body, fpn_level_info_MobileNet_v2_C5(), net_type='mobv2_'
    )


# ---------------------------------------------------------------------------- #
# FPN with MnasNet
# ---------------------------------------------------------------------------- #

def fpn_MnasNet_C5_body():
    return fpn(
        MnasNet.MnasNet_C5_body, fpn_level_info_MnasNet_C5(), net_type='mnas_'
    )


# ---------------------------------------------------------------------------- #
# FPN with VGG
# ---------------------------------------------------------------------------- #

def fpn_VGG13_C5_body():
    return fpn(
        VGG.VGG13_conv5_body, fpn_level_info_VGG13_C5(), net_type='vgg_'
    )


def fpn_VGG16_C5_body():
    return fpn(
        VGG.VGG16_conv5_body, fpn_level_info_VGG16_C5(), net_type='vgg_'
    )


# ---------------------------------------------------------------------------- #
# FPN with Inception_v3
# ---------------------------------------------------------------------------- #

def fpn_Inception_v3_C5_body():
    return fpn(
        Inception_v3.Inception_v3_C5_body, fpn_level_info_Inception_v3_C5(), net_type='incepv3_'
    )


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
class fpn(nn.Module):
    """Add FPN connections based on the model described in the FPN paper.

    fpn_output_blobs is in reversed order: e.g [fpn5, fpn4, fpn3, fpn2]
    similarly for fpn_level_info.dims: e.g [2048, 1024, 512, 256]
    similarly for spatial_scale: e.g [1/32, 1/16, 1/8, 1/4]
    """

    def __init__(self, conv_body_func, fpn_level_info, P2only=False, panet_buttomup=False, net_type='res'):
        super().__init__()
        self.fpn_level_info = fpn_level_info
        self.P2only = P2only
        self.panet_buttomup = panet_buttomup
        self.net_type = net_type

        self.dim_out = fpn_dim = cfg.FPN.DIM
        min_level, max_level = get_min_max_levels()
        self.num_backbone_stages = len(fpn_level_info.blobs) - (min_level - cfg.FPN.LOWEST_BACKBONE_LVL)
        fpn_dim_lateral = fpn_level_info.dims
        self.spatial_scale = []  # a list of scales for FPN outputs

        #
        # Step 1: recursively build down starting from the coarsest backbone level
        #
        # For the coarest backbone level: 1x1 conv only seeds recursion
        self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)
        if cfg.FPN.USE_GN:
            self.conv_top = nn.Sequential(
                nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                             eps=cfg.GROUP_NORM.EPSILON)
            )
        elif cfg.FPN.USE_SN:
            self.conv_top = nn.Sequential(
                nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0, bias=False),
                mynn.SwitchNorm(fpn_dim, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.FPN.SN.USE_BN)
            )
        else:
            self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)
        self.topdown_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()

        # For other levels add top-down and lateral connections
        for i in range(self.num_backbone_stages - 1):
            self.topdown_lateral_modules.append(
                topdown_lateral_module(fpn_dim, fpn_dim_lateral[i + 1])
            )

        # Post-hoc scale-specific 3x3 convs
        for i in range(self.num_backbone_stages):
            if cfg.FPN.USE_GN:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                                 eps=cfg.GROUP_NORM.EPSILON)
                ))
            elif cfg.FPN.USE_SN:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    mynn.SwitchNorm(fpn_dim, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                    using_bn=cfg.FPN.SN.USE_BN)
                ))
            else:
                self.posthoc_modules.append(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
                )

            self.spatial_scale.append(fpn_level_info.spatial_scales[i])

        # add for panet buttom-up path
        if self.panet_buttomup:
            self.panet_buttomup_conv1_modules = nn.ModuleList()
            self.panet_buttomup_conv2_modules = nn.ModuleList()
            for i in range(self.num_backbone_stages - 1):
                if cfg.FPN.USE_GN:
                    self.panet_buttomup_conv1_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1, bias=False),
                        nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                                     eps=cfg.GROUP_NORM.EPSILON),
                        nn.ReLU(inplace=True)
                    ))
                    self.panet_buttomup_conv2_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                        nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                                     eps=cfg.GROUP_NORM.EPSILON),
                        nn.ReLU(inplace=True)
                    ))
                elif cfg.FPN.USE_SN:
                    self.panet_buttomup_conv1_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1, bias=False),
                        mynn.SwitchNorm(fpn_dim, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                        using_bn=cfg.FPN.SN.USE_BN),
                        nn.ReLU(inplace=True)
                    ))
                    self.panet_buttomup_conv2_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 21, 1, bias=False),
                        mynn.SwitchNorm(fpn_dim, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                        using_bn=cfg.FPN.SN.USE_BN),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.panet_buttomup_conv1_modules.append(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1)
                    )
                    self.panet_buttomup_conv2_modules.append(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
                    )

        #
        # Step 2: build up starting from the coarsest backbone level
        #
        # Check if we need the P6 feature map
        if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == cfg.FPN.HIGHEST_BACKBONE_LVL + 1:
            # Original FPN P6 level implementation from our CVPR'17 FPN paper
            # Use max pooling to simulate stride 2 subsampling
            self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        # Coarser FPN levels introduced for RetinaNet
        if cfg.FPN.EXTRA_CONV_LEVELS and max_level > cfg.FPN.HIGHEST_BACKBONE_LVL:
            self.extra_pyramid_modules = nn.ModuleList()
            dim_in = fpn_level_info.dims[0]
            for i in range(cfg.FPN.HIGHEST_BACKBONE_LVL + 1, max_level + 1):
                self.extra_pyramid_modules.append(
                    nn.Conv2d(dim_in, fpn_dim, 3, 2, 1)
                )
                dim_in = fpn_dim
                self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        if self.P2only:
            # use only the finest level
            self.spatial_scale = self.spatial_scale[-1]

        self._init_weights()

        # Deliberately add conv_body after _init_weights.
        # conv_body has its own _init_weights function
        self.conv_body = conv_body_func()  # e.g resnet

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for child_m in self.children():
            if (not isinstance(child_m, nn.ModuleList) or
                    not isinstance(child_m[0], topdown_lateral_module)):
                # topdown_lateral_module has its own init method
                child_m.apply(init_func)

    def detectron_weight_mapping(self):
        conv_body_mapping, orphan_in_detectron = self.conv_body.detectron_weight_mapping()
        mapping_to_detectron = {}
        for key, value in conv_body_mapping.items():
            mapping_to_detectron['conv_body.' + key] = value

        d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[0]
        if cfg.FPN.USE_GN:
            mapping_to_detectron['conv_top.0.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.1.weight'] = d_prefix + '_gn_s'
            mapping_to_detectron['conv_top.1.bias'] = d_prefix + '_gn_b'
        elif cfg.FPN.USE_SN:
            mapping_to_detectron['conv_top.0.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.1.weight'] = d_prefix + '_sn_s'
            mapping_to_detectron['conv_top.1.bias'] = d_prefix + '_sn_b'
            mapping_to_detectron['conv_top.1.mean_weight'] = d_prefix + '_sn_mean_weight'
            mapping_to_detectron['conv_top.1.var_weight'] = d_prefix + '_sn_var_weight'
            if cfg.FPN.SN.USE_BN:
                mapping_to_detectron['conv_top.1.running_mean'] = d_prefix + '_sn_rm'
                mapping_to_detectron['conv_top.1.running_var'] = d_prefix + '_sn_riv'
        else:
            mapping_to_detectron['conv_top.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.bias'] = d_prefix + '_b'
        for i in range(self.num_backbone_stages - 1):
            p_prefix = 'topdown_lateral_modules.%d.conv_lateral' % i
            d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[i + 1] + '_lateral'
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight': d_prefix + '_w',
                    p_prefix + '.1.weight': d_prefix + '_gn_s',
                    p_prefix + '.1.bias': d_prefix + '_gn_b'
                })
            elif cfg.FPN.USE_SN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight': d_prefix + '_w',
                    p_prefix + '.1.weight': d_prefix + '_sn_s',
                    p_prefix + '.1.bias': d_prefix + '_sn_b',
                    p_prefix + '.1.mean_weight': d_prefix + '_sn_mean_weight',
                    p_prefix + '.1.var_weight': d_prefix + '_sn_var_weight',
                })
                if cfg.FPN.SN.USE_BN:
                    mapping_to_detectron.update({
                        p_prefix + '.1.running_mean': d_prefix + '_sn_rm',
                        p_prefix + '.1.running_var': d_prefix + '_sn_riv',
                    })
            else:
                mapping_to_detectron.update({
                    p_prefix + '.weight': d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        for i in range(self.num_backbone_stages):
            p_prefix = 'posthoc_modules.%d' % i
            d_prefix = 'fpn_' + self.fpn_level_info.blobs[i]
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight': d_prefix + '_w',
                    p_prefix + '.1.weight': d_prefix + '_gn_s',
                    p_prefix + '.1.bias': d_prefix + '_gn_b'
                })
            elif cfg.FPN.USE_SN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight': d_prefix + '_w',
                    p_prefix + '.1.weight': d_prefix + '_sn_s',
                    p_prefix + '.1.bias': d_prefix + '_sn_b',
                    p_prefix + '.1.mean_weight': d_prefix + '_sn_mean_weight',
                    p_prefix + '.1.var_weight': d_prefix + '_sn_var_weight',
                })
                if cfg.FPN.SN.USE_BN:
                    mapping_to_detectron.update({
                        p_prefix + '.1.running_mean': d_prefix + '_sn_rm',
                        p_prefix + '.1.running_var': d_prefix + '_sn_riv',
                    })
            else:
                mapping_to_detectron.update({
                    p_prefix + '.weight': d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        if self.panet_buttomup:
            for i in range(self.num_backbone_stages - 1):
                p1_prefix = 'panet_buttomup_conv1_modules.%d' % i
                d1_prefix = 'fpn_inner_bup_' + self.fpn_level_info.blobs[i + 1] + '_downsample'
                p2_prefix = 'panet_buttomup_conv2_modules.%d' % i
                d2_prefix = 'fpn_inner_bup_' + self.fpn_level_info.blobs[i + 1] + '_lateral'
                if cfg.FPN.USE_GN:
                    mapping_to_detectron.update({
                        p1_prefix + '.0.weight': d1_prefix + '_w',
                        p1_prefix + '.0.bias': d1_prefix + '_b',
                        p1_prefix + '.1.weight': d1_prefix + '_gn_s',
                        p1_prefix + '.1.bias': d1_prefix + '_gn_b',
                        p2_prefix + '.0.weight': d2_prefix + '_w',
                        p2_prefix + '.0.bias': d2_prefix + '_b',
                        p2_prefix + '.1.weight': d2_prefix + '_gn_s',
                        p2_prefix + '.1.bias': d2_prefix + '_gn_b'
                    })
                elif cfg.FPN.USE_SN:
                    mapping_to_detectron.update({
                        p1_prefix + '.0.weight': d1_prefix + '_w',
                        p1_prefix + '.1.weight': d1_prefix + '_sn_s',
                        p1_prefix + '.1.bias': d1_prefix + '_sn_b',
                        p1_prefix + '.1.mean_weight': d1_prefix + '_sn_mean_weight',
                        p1_prefix + '.1.var_weight': d1_prefix + '_sn_var_weight',
                        p2_prefix + '.0.weight': d2_prefix + '_w',
                        p2_prefix + '.1.weight': d2_prefix + '_sn_s',
                        p2_prefix + '.1.bias': d2_prefix + '_sn_b',
                        p2_prefix + '.1.mean_weight': d2_prefix + '_sn_mean_weight',
                        p2_prefix + '.1.var_weight': d2_prefix + '_sn_var_weight',
                    })
                    if cfg.FPN.SN.USE_BN:
                        mapping_to_detectron.update({
                            p1_prefix + '.1.running_mean': d1_prefix + '_sn_rm',
                            p1_prefix + '.1.running_var': d1_prefix + '_sn_riv',
                            p2_prefix + '.1.running_mean': d2_prefix + '_sn_rm',
                            p2_prefix + '.1.running_var': d2_prefix + '_sn_riv',
                        })
                else:
                    mapping_to_detectron.update({
                        p1_prefix + '.weight': d1_prefix + '_w',
                        p1_prefix + '.bias': d1_prefix + '_b',
                        p2_prefix + '.weight': d2_prefix + '_w',
                        p2_prefix + '.bias': d2_prefix + '_b'
                    })

        if hasattr(self, 'extra_pyramid_modules'):
            for i in range(len(self.extra_pyramid_modules)):
                p_prefix = 'extra_pyramid_modules.%d' % i
                d_prefix = 'fpn_%d' % (cfg.FPN.HIGHEST_BACKBONE_LVL + 1 + i)
                mapping_to_detectron.update({
                    p_prefix + '.weight': d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        conv_body_mapping, orphan_in_pytorch = self.conv_body.pytorch_weight_mapping()
        mapping_to_pytorch = {}
        for key, value in conv_body_mapping.items():
            mapping_to_pytorch['conv_body.' + key] = value

        return mapping_to_pytorch, orphan_in_pytorch

    def forward(self, x):
        # conv_body_blobs = [self.conv_body.res1(x)]
        conv_body_blobs = [getattr(self.conv_body, self.net_type + '1')(x)]
        for i in range(1, self.conv_body.convX):
            conv_body_blobs.append(
                getattr(self.conv_body, self.net_type + '%d' % (i + 1))(conv_body_blobs[-1])
            )
        fpn_inner_blobs = [self.conv_top(conv_body_blobs[-1])]
        for i in range(self.num_backbone_stages - 1):
            fpn_inner_blobs.append(
                self.topdown_lateral_modules[i](fpn_inner_blobs[-1], conv_body_blobs[-(i + 2)])
            )
        fpn_output_blobs = []
        if self.panet_buttomup:
            fpn_middle_blobs = []
        for i in range(self.num_backbone_stages):
            if not self.panet_buttomup:
                fpn_output_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
            else:
                fpn_middle_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
        if self.panet_buttomup:
            fpn_output_blobs.append(fpn_middle_blobs[-1])
            for i in range(2, self.num_backbone_stages + 1):
                fpn_tmp = self.panet_buttomup_conv1_modules[i - 2](fpn_output_blobs[0])
                # print(fpn_middle_blobs[self.num_backbone_stages - i].size())
                fpn_tmp = fpn_tmp + fpn_middle_blobs[self.num_backbone_stages - i]
                fpn_tmp = self.panet_buttomup_conv2_modules[i - 2](fpn_tmp)
                fpn_output_blobs.insert(0, fpn_tmp)

        if hasattr(self, 'maxpool_p6'):
            fpn_output_blobs.insert(0, self.maxpool_p6(fpn_output_blobs[0]))
            
        if hasattr(self, 'extra_pyramid_modules'):
            blob_in = conv_body_blobs[-1]
            fpn_output_blobs.insert(0, self.extra_pyramid_modules[0](blob_in))
            for module in self.extra_pyramid_modules[1:]:
                fpn_output_blobs.insert(0, module(F.relu(fpn_output_blobs[0])))

        if self.P2only:
            # use only the finest level
            return fpn_output_blobs[-1]
        else:
            # use all levels
            return fpn_output_blobs


class topdown_lateral_module(nn.Module):
    """Add a top-down lateral module."""

    def __init__(self, dim_in_top, dim_in_lateral):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        if cfg.FPN.USE_GN:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(self.dim_out), self.dim_out,
                             eps=cfg.GROUP_NORM.EPSILON)
            )
        elif cfg.FPN.USE_SN:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0, bias=False),
                mynn.SwitchNorm(self.dim_out, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.FPN.SN.USE_BN)
            )
        else:
            self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        if cfg.FPN.USE_GN:
            conv = self.conv_lateral[0]
        elif cfg.FPN.USE_SN:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral

        if cfg.FPN.ZERO_INIT_LATERAL:
            init.constant_(conv.weight, 0)
        else:
            mynn.init.XavierFill(conv.weight)
        if conv.bias is not None:
            init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        # Lateral 1x1 conv
        lat = self.conv_lateral(lateral_blob)
        # Top-down 2x upsampling
        # td = F.upsample(top_blob, size=lat.size()[2:], mode='bilinear')
        if top_blob.size()[2:] == lat.size()[2:]:
            td = top_blob
        else:
            td = F.upsample(top_blob, scale_factor=2, mode='nearest')
        # Sum lateral and top-down
        return lat + td


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = cfg.FPN.LOWEST_BACKBONE_LVL
    max_level = cfg.FPN.HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level


# ---------------------------------------------------------------------------- #
# RPN with an FPN backbone
# ---------------------------------------------------------------------------- #

class fpn_rpn_outputs(nn.Module):
    """Add RPN on FPN specific outputs."""

    def __init__(self, dim_in, spatial_scales):
        super().__init__()
        self.dim_in = dim_in
        self.spatial_scales = spatial_scales
        self.dim_out = self.dim_in
        num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)

        # Create conv ops shared by all FPN levels
        self.FPN_RPN_conv = nn.Conv2d(dim_in, self.dim_out, 3, 1, 1)
        dim_score = num_anchors * 2 if cfg.RPN.CLS_ACTIVATION == 'softmax' \
            else num_anchors
        self.FPN_RPN_cls_score = nn.Conv2d(self.dim_out, dim_score, 1, 1, 0)
        self.FPN_RPN_bbox_pred = nn.Conv2d(self.dim_out, 4 * num_anchors, 1, 1, 0)

        self.GenerateProposals_modules = nn.ModuleList()
        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid: 6
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid: 2
        for lvl in range(k_min, k_max + 1):  # lvl = 2, 3, 4, 5, 6
            sc = self.spatial_scales[k_max - lvl]  # in reversed order
            lvl_anchors = generate_anchors(
                stride=min(2. ** lvl, cfg.FPN.BACKBONE_STRIDE),
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2. ** (lvl - k_min),),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS
            )
            self.GenerateProposals_modules.append(GenerateProposalsOp(lvl_anchors, sc))

        self.CollectAndDistributeFpnRpnProposals = CollectAndDistributeFpnRpnProposalsOp()

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.FPN_RPN_conv.weight, std=0.01)
        init.constant_(self.FPN_RPN_conv.bias, 0)
        init.normal_(self.FPN_RPN_cls_score.weight, std=0.01)
        init.constant_(self.FPN_RPN_cls_score.bias, 0)
        init.normal_(self.FPN_RPN_bbox_pred.weight, std=0.01)
        init.constant_(self.FPN_RPN_bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        k_min = cfg.FPN.RPN_MIN_LEVEL
        mapping_to_detectron = {
            'FPN_RPN_conv.weight': 'conv_rpn_fpn%d_w' % k_min,
            'FPN_RPN_conv.bias': 'conv_rpn_fpn%d_b' % k_min,
            'FPN_RPN_cls_score.weight': 'rpn_cls_logits_fpn%d_w' % k_min,
            'FPN_RPN_cls_score.bias': 'rpn_cls_logits_fpn%d_b' % k_min,
            'FPN_RPN_bbox_pred.weight': 'rpn_bbox_pred_fpn%d_w' % k_min,
            'FPN_RPN_bbox_pred.bias': 'rpn_bbox_pred_fpn%d_b' % k_min
        }
        return mapping_to_detectron, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, blobs_in, im_info, roidb=None):
        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
        assert len(blobs_in) == k_max - k_min + 1
        return_dict = {}
        rois_blobs = []
        score_blobs = []
        for lvl in range(k_min, k_max + 1):
            slvl = str(lvl)
            bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order

            fpn_rpn_conv = F.relu(self.FPN_RPN_conv(bl_in), inplace=True)
            fpn_rpn_cls_score = self.FPN_RPN_cls_score(fpn_rpn_conv)
            fpn_rpn_bbox_pred = self.FPN_RPN_bbox_pred(fpn_rpn_conv)
            return_dict['rpn_cls_logits_fpn' + slvl] = fpn_rpn_cls_score
            return_dict['rpn_bbox_pred_fpn' + slvl] = fpn_rpn_bbox_pred

            if not self.training or cfg.MODEL.FASTER_RCNN:
                # Proposals are needed during:
                #  1) inference (== not model.train) for RPN only and Faster R-CNN
                #  OR
                #  2) training for Faster R-CNN
                # Otherwise (== training for RPN only), proposals are not needed
                if cfg.RPN.CLS_ACTIVATION == 'softmax':
                    B, C, H, W = fpn_rpn_cls_score.size()
                    fpn_rpn_cls_probs = F.softmax(
                        fpn_rpn_cls_score.view(B, 2, C // 2, H, W), dim=1)
                    fpn_rpn_cls_probs = fpn_rpn_cls_probs[:, 1].squeeze(dim=1)
                else:  # sigmoid
                    fpn_rpn_cls_probs = F.sigmoid(fpn_rpn_cls_score)

                fpn_rpn_rois, fpn_rpn_roi_probs = self.GenerateProposals_modules[lvl - k_min](
                    fpn_rpn_cls_probs, fpn_rpn_bbox_pred, im_info)
                rois_blobs.append(fpn_rpn_rois)
                score_blobs.append(fpn_rpn_roi_probs)
                return_dict['rpn_rois_fpn' + slvl] = fpn_rpn_rois
                return_dict['rpn_rois_prob_fpn' + slvl] = fpn_rpn_roi_probs

        if cfg.MODEL.FASTER_RCNN:
            # CollectAndDistributeFpnRpnProposals also labels proposals when in training mode
            blobs_out = self.CollectAndDistributeFpnRpnProposals(rois_blobs + score_blobs, roidb, im_info)
            return_dict.update(blobs_out)

        return return_dict


def fpn_rpn_losses(**kwargs):
    """Add RPN on FPN specific losses."""
    losses_cls = []
    losses_bbox = []
    for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
        slvl = str(lvl)
        # Spatially narrow the full-sized RPN label arrays to match the feature map shape
        b, c, h, w = kwargs['rpn_cls_logits_fpn' + slvl].shape
        rpn_labels_int32_fpn = kwargs['rpn_labels_int32_wide_fpn' + slvl][:, :, :h, :w]
        h, w = kwargs['rpn_bbox_pred_fpn' + slvl].shape[2:]
        rpn_bbox_targets_fpn = kwargs['rpn_bbox_targets_wide_fpn' + slvl][:, :, :h, :w]
        rpn_bbox_inside_weights_fpn = kwargs[
                                          'rpn_bbox_inside_weights_wide_fpn' + slvl][:, :, :h, :w]
        rpn_bbox_outside_weights_fpn = kwargs[
                                           'rpn_bbox_outside_weights_wide_fpn' + slvl][:, :, :h, :w]

        if cfg.RPN.CLS_ACTIVATION == 'softmax':
            rpn_cls_logits_fpn = kwargs['rpn_cls_logits_fpn' + slvl].view(
                b, 2, c // 2, h, w).permute(0, 2, 3, 4, 1).contiguous().view(-1, 2)
            rpn_labels_int32_fpn = rpn_labels_int32_fpn.contiguous().view(-1).long()
            # the loss is averaged over non-ignored targets
            loss_rpn_cls_fpn = F.cross_entropy(
                rpn_cls_logits_fpn, rpn_labels_int32_fpn, ignore_index=-1)
        else:  # sigmoid
            weight = (rpn_labels_int32_fpn >= 0).float()
            loss_rpn_cls_fpn = F.binary_cross_entropy_with_logits(
                kwargs['rpn_cls_logits_fpn' + slvl], rpn_labels_int32_fpn.float(), weight,
                size_average=False)
            loss_rpn_cls_fpn /= cfg.TRAIN.RPN_BATCH_SIZE_PER_IM * cfg.TRAIN.IMS_PER_BATCH

        # Normalization by (1) RPN_BATCH_SIZE_PER_IM and (2) IMS_PER_BATCH is
        # handled by (1) setting bbox outside weights and (2) SmoothL1Loss
        # normalizes by IMS_PER_BATCH
        loss_rpn_bbox_fpn = net_utils.smooth_l1_loss(
            kwargs['rpn_bbox_pred_fpn' + slvl], rpn_bbox_targets_fpn,
            rpn_bbox_inside_weights_fpn, rpn_bbox_outside_weights_fpn,
            beta=1 / 9)

        losses_cls.append(loss_rpn_cls_fpn)
        losses_bbox.append(loss_rpn_bbox_fpn)

    return losses_cls, losses_bbox


# ---------------------------------------------------------------------------- #
# FPN level info for stages 5, 4, 3, 2 for select models (more can be added)
# ---------------------------------------------------------------------------- #

FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)


# ---------------------------------------------------------------------------- #
# FPN for ResNet
# ---------------------------------------------------------------------------- #

def fpn_level_info_ResNet50_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for ResNet_priv
# ---------------------------------------------------------------------------- #

ResNetPrivStageDims = (cfg.AIRS.WIDTH_OUTPLANE * 8, cfg.AIRS.WIDTH_OUTPLANE * 4,
                       cfg.AIRS.WIDTH_OUTPLANE * 2, cfg.AIRS.WIDTH_OUTPLANE)


def fpn_level_info_ResNet26_priv_conv5():
    return FpnLevelInfo(
        blobs=('res8', 'res6', 'res4', 'res2'),
        dims=ResNetPrivStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet50_priv_conv5():
    return FpnLevelInfo(
        blobs=('res16', 'res13', 'res7', 'res3'),
        dims=ResNetPrivStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet101_priv_conv5():
    return FpnLevelInfo(
        blobs=('res33', 'res30', 'res7', 'res3'),
        dims=ResNetPrivStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_priv_conv5():
    return FpnLevelInfo(
        blobs=('res50', 'res47', 'res11', 'res3'),
        dims=ResNetPrivStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for Air
# ---------------------------------------------------------------------------- #

AirStageDims = (cfg.AIRS.WIDTH_OUTPLANE * 8, cfg.AIRS.WIDTH_OUTPLANE * 4,
                cfg.AIRS.WIDTH_OUTPLANE * 2, cfg.AIRS.WIDTH_OUTPLANE)


def fpn_level_info_Air26_conv5():
    return FpnLevelInfo(
        blobs=('air8', 'air6', 'air4', 'air2'),
        dims=AirStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_Air50_conv5():
    return FpnLevelInfo(
        blobs=('air16', 'air13', 'air7', 'air3'),
        dims=AirStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_Air101_conv5():
    return FpnLevelInfo(
        blobs=('air33', 'air30', 'air7', 'air3'),
        dims=AirStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_Air152_conv5():
    return FpnLevelInfo(
        blobs=('air50', 'air47', 'air11', 'air3'),
        dims=AirStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for Aittir
# ---------------------------------------------------------------------------- #

AttirStageDims = (cfg.ATTIRS.WIDTH_OUTPLANE * 8, cfg.ATTIRS.WIDTH_OUTPLANE * 4,
                  cfg.ATTIRS.WIDTH_OUTPLANE * 2, cfg.ATTIRS.WIDTH_OUTPLANE)


def fpn_level_info_Attir50_conv5():
    return FpnLevelInfo(
        blobs=('attir16', 'attir13', 'attir7', 'attir3'),
        dims=AttirStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_Attir101_conv5():
    return FpnLevelInfo(
        blobs=('attir33', 'attir30', 'attir7', 'attir3'),
        dims=AttirStageDims,
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for DetNet
# ---------------------------------------------------------------------------- #

DetNetStageDims = (cfg.DETNETS.WIDTH_OUTPLANE * 4, cfg.DETNETS.WIDTH_OUTPLANE * 4, cfg.DETNETS.WIDTH_OUTPLANE * 4,
                   cfg.DETNETS.WIDTH_OUTPLANE * 2, cfg.DETNETS.WIDTH_OUTPLANE)


def fpn_level_info_DetNet59_conv6():
    return FpnLevelInfo(
        blobs=('detnet19', 'detnet16', 'detnet13', 'detnet7', 'detnet3'),
        dims=DetNetStageDims,
        spatial_scales=(1. / 16., 1. / 16, 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_DetNet110_conv6():
    return FpnLevelInfo(
        blobs=('detnet36', 'detnet33', 'detnet30', 'detnet7', 'detnet3'),
        dims=DetNetStageDims,
        spatial_scales=(1. / 16., 1. / 16, 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for DetAir
# ---------------------------------------------------------------------------- #

DetAirStageDims = (cfg.DETAIRS.WIDTH_OUTPLANE * 4, cfg.DETAIRS.WIDTH_OUTPLANE * 4, cfg.DETAIRS.WIDTH_OUTPLANE * 4,
                   cfg.DETAIRS.WIDTH_OUTPLANE * 2, cfg.DETAIRS.WIDTH_OUTPLANE)


def fpn_level_info_DetAir32_conv6():
    return FpnLevelInfo(
        blobs=('detair10', 'detair8', 'detair6', 'detair4', 'detair2'),
        dims=DetAirStageDims,
        spatial_scales=(1. / 16., 1. / 16, 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_DetAir59_conv6():
    return FpnLevelInfo(
        blobs=('detair19', 'detair16', 'detair13', 'detair7', 'detair3'),
        dims=DetAirStageDims,
        spatial_scales=(1. / 16., 1. / 16, 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_DetAir110_conv6():
    return FpnLevelInfo(
        blobs=('detair36', 'detair33', 'detair30', 'detair7', 'detair3'),
        dims=DetAirStageDims,
        spatial_scales=(1. / 16., 1. / 16, 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_DetAir161_conv6():
    return FpnLevelInfo(
        blobs=('detair53', 'detair50', 'detair47', 'detair11', 'detair3'),
        dims=DetAirStageDims,
        spatial_scales=(1. / 16., 1. / 16, 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for CrossNet
# ---------------------------------------------------------------------------- #

def fpn_level_info_CrossNet_C5():
    CrossWidth = int(cfg.CROSSNET.BASE_WIDTH * cfg.CROSSNET.EXPANSION)

    _blob1 = 'pod{}crs{}'.format(sum(cfg.CROSSNET.BLOCK_COUNTS[:1]), cfg.CROSSNET.POD_DEPTH)
    _blob2 = 'pod{}crs{}'.format(sum(cfg.CROSSNET.BLOCK_COUNTS[:2]), cfg.CROSSNET.POD_DEPTH)
    _blob3 = 'pod{}crs{}'.format(sum(cfg.CROSSNET.BLOCK_COUNTS[:3]), cfg.CROSSNET.POD_DEPTH)
    _blob4 = 'pod{}crs{}'.format(sum(cfg.CROSSNET.BLOCK_COUNTS[:4]), cfg.CROSSNET.POD_DEPTH)

    return FpnLevelInfo(
        blobs=(_blob4, _blob3, _blob2, _blob1),
        dims=(CrossWidth * 8, CrossWidth * 4, CrossWidth * 2, CrossWidth),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for MobileNet_v1
# ---------------------------------------------------------------------------- #

def fpn_level_info_MobileNet_v1_C5():
    _outplanes = MobileNet_v1.level_outplanes()

    return FpnLevelInfo(
        blobs=('mobv1_12', 'mobv2_10', 'mobv2_4', 'mobv2_2'),
        dims=(_outplanes[3], _outplanes[2], _outplanes[1], _outplanes[0]),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for MobileNet_v2
# ---------------------------------------------------------------------------- #

def fpn_level_info_MobileNet_v2_C5():
    _outplanes = MobileNet_v2.level_outplanes()

    return FpnLevelInfo(
        blobs=('mobv2_16_conv3_bn', 'mobv2_12', 'mobv2_5', 'mobv2_2'),
        dims=(_outplanes[3], _outplanes[2], _outplanes[1], _outplanes[0]),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for MnasNet
# ---------------------------------------------------------------------------- #

def fpn_level_info_MnasNet_C5():
    _outplanes = MnasNet.level_outplanes()

    return FpnLevelInfo(
        blobs=('mnas_16_conv3_bn', 'mnas_12', 'mnas_5', 'mnas_2'),
        dims=(_outplanes[3], _outplanes[2], _outplanes[1], _outplanes[0]),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for VGG
# ---------------------------------------------------------------------------- #

def fpn_level_info_VGG13_C5():
    return FpnLevelInfo(
        blobs=('conv5_2', 'conv4_2', 'conv3_2', 'conv2_2'),
        dims=(512, 512, 256, 128),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_VGG16_C5():
    return FpnLevelInfo(
        blobs=('conv5_3', 'conv4_3', 'conv3_3', 'conv2_2'),
        dims=(512, 512, 256, 128),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


# ---------------------------------------------------------------------------- #
# FPN for Inception_v3
# ---------------------------------------------------------------------------- #

def fpn_level_info_Inception_v3_C5():
    return FpnLevelInfo(
        blobs=('Mixed_7c_concat', 'Mixed_6e_concat', 'Mixed_5d_concat', 'Conv2d_4a_3x3.bn'),
        dims=(2048, 768, 288, 192),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )
