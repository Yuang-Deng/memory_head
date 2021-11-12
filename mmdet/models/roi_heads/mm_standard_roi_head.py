# Copyright (c) OpenMMLab. All rights reserved.
from re import S
import torch
from torch._C import wait
from torch.functional import split

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .mm_base_roi_head import MMBaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn.functional as F
from numpy import random


@HEADS.register_module()
class MMStandardRoIHead(MMBaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_mem_bank(self):

        self.mem_fc = torch.nn.Sequential(torch.nn.Linear(12544, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 128))
        self.fwd_fc = torch.nn.Sequential(torch.nn.Linear(12544, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 128))

        self.register_buffer("queue_vector", torch.randn(self.memory_k, 128)) 
        self.queue_vector = F.normalize(self.queue_vector, dim=1)
        self.register_buffer("queue_label", torch.ones(self.memory_k).long() * 0)
        self.register_buffer("queue_boxes", torch.ones(self.memory_k, 4).long())

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    @torch.no_grad()
    def concat_all_gather(self, features, labels, boxes=None):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        device = features.device
        assert labels.size(0) != 0 and features.size(0) == labels.size(0)
        local_batch = torch.tensor(labels.size(0)).to(device)
        batch_size_gather = [torch.ones((1)).to(device)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(batch_size_gather, local_batch.float(), async_op=False)
        
        batch_size_gather = [int(bs.item()) for bs in batch_size_gather]

        max_batch = max(batch_size_gather)
        size = (max_batch, features.size(1))
        temp_features = torch.zeros(max_batch - local_batch, features.size(1)).to(device)
        features = torch.cat([features, temp_features])

        # size = (int(tensors_gather[0].item()), features.size(1))
        # (int(tensors_gather[i].item()), features.size(1))
        features_gather = [torch.ones(size).to(device)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(features_gather, features, async_op=False)

        features_gather = [f[:bs, :] for bs, f in zip(batch_size_gather, features_gather)]

        features = torch.cat(features_gather, dim=0)

        temp_labels = torch.zeros(max_batch - local_batch).to(device)
        labels = torch.cat([labels, temp_labels])

        labels_gather = [torch.ones(max_batch).to(device)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(labels_gather, labels, async_op=False)

        labels_gather = [l[:bs] for bs, l in zip(batch_size_gather, labels_gather)]

        labels = torch.cat(labels_gather, dim=0)

        if boxes is not None:
            temp_boxes = torch.zeros(max_batch - local_batch, 4).to(device)
            boxes = torch.cat([boxes, temp_boxes])

            boxes_gather = [torch.ones(max_batch, 4).to(device)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(boxes_gather, boxes, async_op=False)

            boxes_gather = [l[:bs] for bs, l in zip(batch_size_gather, boxes_gather)]

            boxes = torch.cat(boxes_gather, dim=0)

            return features, labels, boxes

        return features, labels, None

    @torch.no_grad()
    def _dequeue_and_enqueue(self, features, labels, boxes=None):
        # gather keys before updating queue
        features, labels, boxes = self.concat_all_gather(features, labels, boxes)

        batch_size = features.size(0)

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size >= self.memory_k:
            redundant = ptr + batch_size - self.memory_k
            self.queue_vector[ptr:self.memory_k, :] = features.view(batch_size, -1)[:batch_size - redundant]
            self.queue_vector[:redundant, :] = features.view(batch_size, -1)[batch_size - redundant:]
            self.queue_label[ptr:self.memory_k] = labels[:batch_size - redundant]
            self.queue_label[:redundant] = labels[batch_size - redundant:]
            if boxes is not None:
                self.queue_boxes[ptr:self.memory_k] = boxes[:batch_size - redundant]
                self.queue_boxes[:redundant] = boxes[batch_size - redundant:]
        else:
            self.queue_vector[ptr:ptr + batch_size, :] = features.view(batch_size, -1)
            self.queue_label[ptr:ptr + batch_size] = labels
            if boxes is not None:
                self.queue_boxes[ptr:ptr + batch_size] = boxes
        ptr = (ptr + batch_size) % self.memory_k  # move pointer

        self.queue_ptr[0] = ptr
    
    def mem_forward(self,
                      x,
                      gt_bboxes,
                      gt_labels,):
        rois = bbox2roi([res for res in gt_bboxes])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        mem_gt_label = torch.cat(gt_labels)
        mem_gt_boxes = torch.cat(gt_bboxes)
        mem_bbox_feats = bbox_feats.view(bbox_feats.size(0), -1)
        with torch.no_grad():
            mem_bbox_feats = self.fwd_fc(mem_bbox_feats)
            mem_bbox_feats = F.normalize(mem_bbox_feats, dim=1)
        self._dequeue_and_enqueue(mem_bbox_feats.detach(), mem_gt_label.detach(), mem_gt_boxes.detach())

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_tags=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            sampling_results_ctr = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

                assign_result_ctr = self.bbox_assigner.assign(
                    kwargs['aug_proposal_list'][i], kwargs['aug_gt_bboxes'][i], gt_bboxes_ignore[i],
                    kwargs['aug_gt_labels'][i])
                sampling_result_ctr = self.bbox_sampler.sample(
                    assign_result_ctr,
                    kwargs['aug_proposal_list'][i],
                    kwargs['aug_gt_bboxes'][i],
                    kwargs['aug_gt_labels'][i],
                    feats=[lvl_feat[i][None] for lvl_feat in kwargs['x_saug']])
                sampling_results_ctr.append(sampling_result_ctr)
            kwargs['sampling_results_ctr'] = sampling_results_ctr

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._unlabel_bbox_forward_train(x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas, gt_tags, **kwargs)

            # if 1 == img_metas[0]['label_type'] and self.cur_epoch >= self.warm_epoch:
            #     bbox_results = self._unalbel_bbox_forward_train(x, sampling_results,
            #                                             gt_bboxes, gt_labels,
            #                                             img_metas, gt_tags, **kwargs)
            # else:
            #     bbox_results = self._bbox_forward_train(x, sampling_results,
            #                                             gt_bboxes, gt_labels,
            #                                             img_metas, gt_tags, **kwargs)

            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, mid_cls_score, mid_det_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            mid_cls_score=mid_cls_score,
            mid_det_score=mid_det_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats)
        return bbox_results
    
    def _unlabel_interp_bbox_forward(self, x, rois, labels):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        pos_labels = labels[pos_inds]
        pos_bbox_feats = bbox_feats[pos_inds].view(pos_labels.size(0), -1)

        neg_bbox_feats = bbox_feats[~pos_inds].view(labels.size(0) - pos_labels.size(0), -1)

        dot_sum = torch.einsum('ik,jk->ij', [pos_bbox_feats, self.queue_vector])
        vector_norm = torch.norm(pos_bbox_feats, dim=-1, p=2)
        memory_norm = torch.norm(self.queue_vector, dim=-1, p=2)
        norm_mul = torch.einsum('i,j->ij', [vector_norm, memory_norm])
        cos_sim = dot_sum / norm_mul

        add_sim = torch.zeros(0, self.top_k).to(pos_labels.device)
        add_label = torch.zeros(0).to(pos_labels.device).long()
        add_boxes = torch.zeros(0, 4).to(pos_labels.device).long()
        add_feat = torch.zeros(0, self.queue_vector.size(1)).to(pos_labels.device)
        
        for i in range(pos_labels.size(0)):
            neg_cls_inds = (self.queue_label != pos_labels[i]) & (self.queue_label != 30)
            _, neg_inds = torch.topk(cos_sim[i][neg_cls_inds], self.top_k)
            add_sim = torch.cat([add_sim, cos_sim[i][neg_cls_inds][neg_inds][None,:]])
            cur_interp_feat = self.interpolation_feature_augment(neg_bbox_feats[random.randint(0, neg_bbox_feats.size(0), neg_inds.size(0))], self.queue_vector[neg_inds])
            add_feat = torch.cat([add_feat, cur_interp_feat], dim=0)
            add_label = torch.cat([add_label, self.queue_label[neg_inds]])
            add_boxes = torch.cat([add_boxes, self.queue_boxes[neg_inds]])

        cls_score, mid_cls_score, mid_det_score, bbox_pred = self.bbox_head(bbox_feats)

        add_cls_score, _, _, add_bbox_pred = self.bbox_head(add_feat.view(add_feat.size(0), bbox_feats.size(1), bbox_feats.size(2), bbox_feats.size(3)))

        bbox_results = dict(
            cls_score=cls_score,
            mid_cls_score=mid_cls_score,
            mid_det_score=mid_det_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
            fg_anchor=pos_bbox_feats,
            add_feat=add_feat,
            add_sim=add_sim,
            add_cls_score=add_cls_score,
            add_bbox_pred=add_bbox_pred,
            add_label=add_label,
            add_boxes=add_boxes)
        return bbox_results
    
    def _contrast_bbox_forward(self, x, rois, labels, x_saug, saug_bboxes, saug_labels, sampling_results, sampling_results_ctr):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        # pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)

        device = rois.device
        batch = sampling_results[0].bboxes.size(0)
        pos_inds = torch.zeros([0]).to(device).long()
        pos_gt_map = torch.zeros([0]).to(device).long()
        pos_gt_map_ctr = torch.zeros([0]).to(device).long()
        pos_labels = torch.zeros([0]).to(device).long()
        for i, (res, res_ctr) in enumerate(zip(sampling_results, sampling_results_ctr)):
            pos_inds = torch.cat([pos_inds, (torch.arange(0, res.pos_inds.size(0)).to(device).long() + (i * batch)).view(-1)])
            pos_gt_map = torch.cat([pos_gt_map, (res.pos_assigned_gt_inds+ (i * batch)).view(-1)])
            pos_gt_map_ctr = torch.cat([pos_gt_map_ctr, (res_ctr.pos_assigned_gt_inds+ (i * batch)).view(-1)])
            pos_labels = torch.cat([pos_labels, res.pos_gt_labels])


        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        saug_rois = bbox2roi([res for res in saug_bboxes])
        all_saug_labels = torch.cat(saug_labels)
        saug_rois_ctr = bbox2roi([res.pos_bboxes for res in sampling_results_ctr])
        all_saug_labels_ctr = torch.cat([res.pos_gt_labels for res in sampling_results_ctr])
        with torch.no_grad():
            contrast_bbox_feats = self.bbox_roi_extractor(
                x_saug[:self.bbox_roi_extractor.num_inputs], saug_rois)
            contrast_bbox_feats = self.mem_fc(contrast_bbox_feats.view(contrast_bbox_feats.size(0), -1))
            contrast_bbox_feats = F.normalize(contrast_bbox_feats, dim=1)

            contrast_bbox_feats_ctr = self.bbox_roi_extractor(
                x_saug[:self.bbox_roi_extractor.num_inputs], saug_rois_ctr)
            contrast_bbox_feats_ctr = self.mem_fc(contrast_bbox_feats_ctr.view(contrast_bbox_feats_ctr.size(0), -1))
            contrast_bbox_feats_ctr = F.normalize(contrast_bbox_feats_ctr, dim=1)

            pos_bbox_feats_ctr = bbox_feats[pos_inds].view(pos_labels.size(0), -1)
            pos_bbox_feats_ctr = self.mem_fc(pos_bbox_feats_ctr)
            pos_bbox_feats_ctr = F.normalize(pos_bbox_feats_ctr, dim=1)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        pos_bbox_feats = bbox_feats[pos_inds].view(pos_labels.size(0), -1)
        pos_bbox_feats = self.fwd_fc(pos_bbox_feats)
        pos_bbox_feats = F.normalize(pos_bbox_feats, dim=1)

        all_ori_pos_logit_pseudo = []
        for _ in range(self.ori_pos_k):
            all_ori_pos_logit_pseudo.append(torch.zeros(0, 128).to(pos_labels.device))

        all_pos_logit_pseudo = []
        for _ in range(self.pos_k):
            all_pos_logit_pseudo.append(torch.zeros(0, 128).to(pos_labels.device))
        
        for i in range(pos_labels.size(0)):
            pos_inds = pos_gt_map_ctr == pos_gt_map[i]
            pos_logits = contrast_bbox_feats_ctr[pos_inds, :]
            if pos_logits.size(0) == 0:
                pos_inds = pos_gt_map == pos_gt_map[i]
                pos_logits = pos_bbox_feats_ctr[pos_inds, :]
            for j in range(self.ori_pos_k):
                rand_index = torch.randint(low=0, high=pos_logits.size(0), size=(1,))
                pos_logit = pos_logits[rand_index, :]
                all_ori_pos_logit_pseudo[j] = torch.cat([all_ori_pos_logit_pseudo[j], pos_logit], dim=0)

            pos_inds = all_saug_labels == pos_labels[i]
            pos_logits = contrast_bbox_feats[pos_inds, :]
            for j in range(self.pos_k):
                rand_index = torch.randint(low=0, high=pos_logits.size(0), size=(1,))
                pos_logit = pos_logits[rand_index, :]
                all_pos_logit_pseudo[j] = torch.cat([all_pos_logit_pseudo[j], pos_logit], dim=0)

        re_logits = []
        neg_logits = torch.einsum('nc,kc->nk', [pos_bbox_feats, self.queue_vector.clone().detach()])
        for i in range(self.pos_k):
            pos_logits = torch.einsum('nc,nc->n', [pos_bbox_feats, all_pos_logit_pseudo[i]])
            logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
            logits /= self.T
            re_logits.append(logits)
        
        re_ori_logits = []
        for i in range(self.ori_pos_k):
            pos_logits = torch.einsum('nc,nc->n', [pos_bbox_feats, all_ori_pos_logit_pseudo[i]])
            logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
            logits /= self.T
            re_ori_logits.append(logits)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(contrast_bbox_feats, all_saug_labels)

        cls_score, mid_cls_score, mid_det_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            mid_cls_score=mid_cls_score,
            mid_det_score=mid_det_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
            logits=re_logits,
            ori_logits=re_ori_logits,
            labels=labels,)
        return bbox_results

    def _unlabel_bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_tags, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        x_saug = kwargs['x_saug']
        bbox_results = self._contrast_bbox_forward(x, rois, bbox_targets[0], x_saug, kwargs['aug_gt_bboxes'], kwargs['aug_gt_labels'], sampling_results, kwargs['sampling_results_ctr'])
        split_list = [sr.bboxes.size(0) for sr in sampling_results]
        
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'],
                                            bbox_results['mid_cls_score'],
                                            bbox_results['mid_det_score'],
                                            split_list,
                                            rois,
                                            *bbox_targets,
                                            gt_tags, **kwargs)

        loss_contrastive = []
        for i in range(self.pos_k):
            loss_contrastive.append(F.cross_entropy(bbox_results['logits'][i], bbox_results['labels']) * (1 / self.pos_k))
        loss_contrastive = sum(loss_contrastive) * self.contrastive_lambda

        loss_contrastive_ori = []
        for i in range(self.ori_pos_k):
            loss_contrastive_ori.append(F.cross_entropy(bbox_results['ori_logits'][i], bbox_results['labels']) * (1 / self.ori_pos_k))
        loss_contrastive_ori = sum(loss_contrastive_ori) * self.contrastive_lambda_ori


        loss_bbox['loss_contrastive'] = loss_contrastive
        loss_bbox['loss_contrastive_ori'] = loss_contrastive_ori

        bbox_results.update(loss_bbox=loss_bbox)

        # self.dynamic_mem_adjust(bbox_results, sampling_results)

        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_tags, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        bbox_results = self._bbox_forward(x, rois)
        split_list = [sr.bboxes.size(0) for sr in sampling_results]
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'],
                                            bbox_results['mid_cls_score'],
                                            bbox_results['mid_det_score'],
                                            split_list,
                                            rois,
                                            *bbox_targets,
                                            self.train_cfg.sampler.num,
                                            gt_tags, **kwargs)

        bbox_results.update(loss_bbox=loss_bbox)
        
        # self.dynamic_mem_adjust(bbox_results, sampling_results)

        return bbox_results

    def interpolation_feature_augment(self, anchors, features):
        rand_intp = (torch.rand(features.size(0), features.size(1)) / 10 * self.intp_band + self.intp_base).to(anchors.device)
        # rand_intp = rand_intp.expand(anchors.size(0), features.size(1), features.size(2))
        # rand_intp2 = (torch.rand(features.size(0)) / 10 * self.intp_band + self.intp_base).to(anchors.device).expand(features.size(0), features.size(1))
        anchors = anchors.expand(features.size(0), features.size(1))
        return anchors * (1 - rand_intp) + features * rand_intp
    
    def dynamic_mem_adjust(self, bbox_results, sampling_results):
        # TODO add memory head
        device = bbox_results['cls_score'].device
        batch = self.train_cfg.sampler.num
        # gt_feat_ind = torch.zeros([bbox_results['cls_score'].size(0)]).to(device)
        mem_gt_feat_ind = torch.zeros([0]).to(device)
        mem_gt_label = torch.zeros([0]).to(device)
        for i, res in enumerate(sampling_results):
            mem_gt_feat_ind = torch.cat([(torch.nonzero(res.pos_is_gt) + (i * batch)).view(-1), mem_gt_feat_ind])
            mem_gt_label = torch.cat([res.pos_gt_labels[torch.nonzero(res.pos_is_gt).long()].view(-1), mem_gt_label])
            # gt_feat_ind[i * batch:i * batch + len(res.pos_is_gt)] = res.pos_is_gt
        mem_gt_feat = bbox_results['bbox_feats'][mem_gt_feat_ind.long()]

        mem_gt_feat = mem_gt_feat.view(mem_gt_feat.size(0), -1)

        self._dequeue_and_enqueue(mem_gt_feat.detach(), mem_gt_label.detach())

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
    
    def set_epoch(self, epoch):
        self.cur_epoch = epoch + 1

    def mem_weight_update(self):
        for param_q, param_k in zip(self.mem_fc.parameters(), self.fwd_fc.parameters()):
            param_q.data = param_q.data * self.ema + param_k.data * (1. - self.ema)

    def mem_weight_init(self):
        # mem no gradient - teacher weak augment
        for param_q, param_k in zip(self.mem_fc.parameters(), self.fwd_fc.parameters()):
            param_q.data.copy_(param_k.data)
            param_q.requires_grad = False