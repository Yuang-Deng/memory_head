# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .memory_bbox_head import MMBBoxHead
import torch.nn.functional as F


@HEADS.register_module()
class MMConvFCBBoxHead(MMBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 memory_k=32768,
                 *args,
                 **kwargs):
        super(MMConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # self.memory_k = memory_k
        # self.init_mem_bank(memory_k=memory_k)
        
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        
        self.mid_cls_convs, self.mid_cls_fcs, self.mid_cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        self.mid_det_convs, self.mid_det_fcs, self.mid_det_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.fc_mid_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=self.num_classes)
            self.fc_mid_det = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]
    
    # def init_mem_bank(self, memory_k, dim=1024):
    #     self.register_buffer("queue_vector", torch.randn(dim, memory_k)) 
    #     self.queue_vector = F.normalize(self.queue_vector, dim=0)
    #     self.register_buffer("queue_lable", torch.ones(size=[memory_k]))

    #     self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward_mem(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_mid_cls = x
        x_mid_det = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.mid_cls_convs:
            x_mid_cls = conv(x_mid_cls)
        if x_mid_cls.dim() > 2:
            if self.with_avg_pool:
                x_mid_cls = self.avg_pool(x_mid_cls)
            x_mid_cls = x_mid_cls.flatten(1)
        for fc in self.mid_cls_fcs:
            x_mid_cls = self.relu(fc(x_mid_cls))

        for conv in self.mid_det_convs:
            x_mid_det = conv(x_mid_det)
        if x_mid_det.dim() > 2:
            if self.with_avg_pool:
                x_mid_det = self.avg_pool(x_mid_det)
            x_mid_det = x_mid_det.flatten(1)
        for fc in self.mid_det_fcs:
            x_mid_det = self.relu(fc(x_mid_det))

        cls_score, mid_cls_score, mid_det_score = self.fc_cls(x_cls), self.fc_mid_cls(x_mid_cls), self.fc_mid_det(x_mid_det) if self.with_cls else None
        return cls_score
    
    # @torch.no_grad()
    # def concat_all_gather(tensor):
    #     """
    #     Performs all_gather operation on the provided tensors.
    #     *** Warning ***: torch.distributed.all_gather has no gradient.
    #     """
    #     tensors_gather = [torch.ones_like(tensor)
    #         for _ in range(torch.distributed.get_world_size())]
    #     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    #     output = torch.cat(tensors_gather, dim=0)
    #     return output

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, features, labels):
    #     # gather keys before updating queue
    #     features = self.concat_all_gather(features)
    #     labels = self.concat_all_gather(labels)

    #     batch_size = features.shape[0]

    #     ptr = int(self.queue_ptr)

    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue_vector[:, ptr:ptr + batch_size] = features
    #     self.queue_lable[:, ptr:ptr + batch_size] = labels
    #     ptr = (ptr + batch_size) % self.K  # move pointer

    #     self.queue_ptr[0] = ptr

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_mid_cls = x
        x_mid_det = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.mid_cls_convs:
            x_mid_cls = conv(x_mid_cls)
        if x_mid_cls.dim() > 2:
            if self.with_avg_pool:
                x_mid_cls = self.avg_pool(x_mid_cls)
            x_mid_cls = x_mid_cls.flatten(1)
        for fc in self.mid_cls_fcs:
            x_mid_cls = self.relu(fc(x_mid_cls))

        for conv in self.mid_det_convs:
            x_mid_det = conv(x_mid_det)
        if x_mid_det.dim() > 2:
            if self.with_avg_pool:
                x_mid_det = self.avg_pool(x_mid_det)
            x_mid_det = x_mid_det.flatten(1)
        for fc in self.mid_det_fcs:
            x_mid_det = self.relu(fc(x_mid_det))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score, mid_cls_score, mid_det_score = self.fc_cls(x_cls), self.fc_mid_cls(x_mid_cls), self.fc_mid_det(x_mid_det) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, mid_cls_score, mid_det_score, bbox_pred


@HEADS.register_module()
class MMShared2FCBBoxHead(MMConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(MMShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class MMShared4Conv1FCBBoxHead(MMConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(MMShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
