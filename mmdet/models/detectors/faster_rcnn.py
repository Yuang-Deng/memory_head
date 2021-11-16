# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

@DETECTORS.register_module()
class EMAFasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(EMAFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.ema = train_cfg.ema
        
        self.ema_backbone = build_backbone(backbone)

        if neck is not None:
            self.ema_neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.ema_rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.ema_roi_head = build_head(roi_head)
        
        self._init_and_freeze('ema_backbone', 'backbone')
        self._init_and_freeze('ema_neck', 'neck')
        self._init_and_freeze('ema_rpn_head', 'rpn_head')
        self._init_and_freeze('ema_roi_head', 'roi_head')

        # self.roi_head.ema_roi_head = self.ema_roi_head

    def _init_and_freeze(self, model_ref:str, model_base:str):
        model_q = getattr(self, model_ref)
        model_k = getattr(self, model_base)
        model_q.eval()
        for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
            param_q.data.copy_(param_k.data)
            param_q.requires_grad = False

    def extract_feat_ema(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.ema_backbone(img)
        if self.with_neck:
            x = self.ema_neck(x)
        return x
    
    def ema_update(self):
        for param_q, param_k in zip(self.ema_backbone.parameters(), self.backbone.parameters()):
            param_q.data = param_q.data * self.ema + param_k.data * (1. - self.ema)

        for param_q, param_k in zip(self.ema_neck.parameters(), self.neck.parameters()):
            param_q.data = param_q.data * self.ema + param_k.data * (1. - self.ema)

        for param_q, param_k in zip(self.ema_rpn_head.parameters(), self.rpn_head.parameters()):
            param_q.data = param_q.data * self.ema + param_k.data * (1. - self.ema)

        for param_q, param_k in zip(self.ema_roi_head.parameters(), self.roi_head.parameters()):
            param_q.data = param_q.data * self.ema + param_k.data * (1. - self.ema)