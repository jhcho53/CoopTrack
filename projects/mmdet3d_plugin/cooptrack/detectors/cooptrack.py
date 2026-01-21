#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import math
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid
from ..dense_heads.track_head_plugin import Instances, RunTimeTracker
from ..modules import CrossAgentSparseInteraction
import mmcv,os
import torch.nn.functional as F
import numpy as np
from ..modules import SpatialTemporalReasoner, MotionExtractor, LatentTransformation
from ..modules import pos2posemb3d
from torchvision.ops import sigmoid_focal_loss

def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result

@DETECTORS.register_module()
class CoopTrack(MVXTwoStageDetector):
    """
    CoopTrack
    """
    def __init__(
        self, 
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        pc_range=None,
        inf_pc_range=None,
        post_center_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        runtime_tracker=None,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
        is_cooperation=False,
        read_track_query_file_root=None,
        drop_rate = 0,
        save_track_query=False,
        save_track_query_file_root='',
        seq_mode=False,
        batch_size=1,
        spatial_temporal_reason=None,
        motion_prediction_ref_update=True,
        if_update_ego=True,
        train_det=False,
        fp_ratio=0.3,
        random_drop=0.1,
        shuffle=False,
        is_motion=False,
        asso_loss_cfg=None,
    ):
        super(CoopTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.queue_length = queue_length
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        
        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        # query initialization for detection
        # reference points, mapping fourier encoding to embed_dims
        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        self.query_feat_embedding = nn.Embedding(self.num_query, self.embed_dims)
        nn.init.zeros_(self.query_feat_embedding.weight)

        self.runtime_tracker = RunTimeTracker(
            **runtime_tracker
        ) 

        self.criterion = build_loss(loss_cfg)
        # for test memory
        self.scene_token = None
        self.timestamp = None
        self.prev_bev = None
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.prev_pos = 0
        self.prev_angle = 0
        
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

        # spatial-temporal reasoning
        self.STReasoner = SpatialTemporalReasoner(**spatial_temporal_reason)
        self.hist_len = self.STReasoner.hist_len
        self.fut_len = self.STReasoner.fut_len
        
        self.motion_prediction_ref_update=motion_prediction_ref_update
        self.if_update_ego = if_update_ego
        self.train_det = train_det
        self.fp_ratio = fp_ratio
        self.random_drop = random_drop
        self.shuffle = shuffle
        self.is_motion = is_motion
        if self.is_motion:
            self.MotionExtractor = MotionExtractor(embed_dims=embed_dims,
                                                   mlp_channels=(3, 64, 64, 256))

        # cross-agent query interaction
        self.is_cooperation = is_cooperation
        self.read_track_query_file_root = read_track_query_file_root
        if self.is_cooperation:
            self.crossview_alignment = LatentTransformation(embed_dims=embed_dims,
                                                            head=16,
                                                            rot_dims=6,
                                                            trans_dims=3,
                                                            pc_range=pc_range,
                                                            inf_pc_range=inf_pc_range
                                                            )
            if self.STReasoner.learn_match:
                self.asso_loss_focal = asso_loss_cfg['loss_focal']
        self.drop_rate = drop_rate

        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root

        self.bev_embed_linear = nn.Linear(embed_dims, embed_dims)
        self.bev_pos_linear = nn.Linear(embed_dims, embed_dims)

        self.seq_mode = seq_mode
        if self.seq_mode:
            self.batch_size = batch_size
            self.test_flag = False
            # for stream train memory
            self.train_prev_infos = {
                'scene_token': [None] * self.batch_size,
                'prev_timestamp': [None] * self.batch_size,
                'prev_bev': [None] * self.batch_size,
                'track_instances': [None] * self.batch_size,
                'l2g_r_mat': [None] * self.batch_size,
                'l2g_t': [None] * self.batch_size,
                'prev_pos': [0] * self.batch_size,
                'prev_angle': [0] * self.batch_size,
            }
            
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        
    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      gt_forecasting_locs=None,
                      gt_forecasting_masks=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      #for coop
                      veh2inf_rt=None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        if self.test_flag: #for interval evaluation
            self.reset_memory()
            self.test_flag = False
        losses = dict()
        if self.seq_mode:
            losses_track = self.forward_track_stream_train(img, gt_bboxes_3d, gt_labels_3d, gt_inds, gt_forecasting_locs, gt_forecasting_masks,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt, **kwargs)
        else:
            NotImplementedError
            # losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_inds,
            #                                             l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt)
        losses.update(losses_track)
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses
    
    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     #for coop
                     veh2inf_rt=None,
                     **kwargs
                    ):
        """Test function
        """
        # import ipdb;ipdb.set_trace()
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt, **kwargs)
        
        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
        return result
    
    def reset_memory(self):
        self.train_prev_infos['scene_token'] = [None] * self.batch_size
        self.train_prev_infos['prev_timestamp'] = [None] * self.batch_size
        self.train_prev_infos['prev_bev'] = [None] * self.batch_size
        self.train_prev_infos['track_instances'] = [None] * self.batch_size
        self.train_prev_infos['l2g_r_mat'] = [None] * self.batch_size
        self.train_prev_infos['l2g_t'] = [None] * self.batch_size
        self.train_prev_infos['prev_pos'] = [0] * self.batch_size
        self.train_prev_infos['prev_angle'] = [0] * self.batch_size

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        device = self.reference_points.weight.device
        
        """Detection queries"""
        # reference points, query embeds, and query targets (features)
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        track_instances.ref_pts = reference_points.clone()
        track_instances.query_embeds = query_embeds.clone()
        track_instances.query_feats = self.query_feat_embedding.weight.clone()
        
        """Tracking information"""
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)
        
        """Current frame information"""
        # classification scores
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # bounding boxes
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), self.pts_bbox_head.code_size), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (len(track_instances)), dtype=torch.float, device=device)
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # motion prediction, not normalized
        track_instances.motion_predictions = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        
        """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        track_instances.cache_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.cache_bboxes = torch.zeros(
            (len(track_instances), self.pts_bbox_head.code_size), dtype=torch.float, device=device)
        track_instances.cache_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.cache_ref_pts = reference_points.clone()
        track_instances.cache_query_embeds = query_embeds.clone()
        track_instances.cache_query_feats = self.query_feat_embedding.weight.clone()
        track_instances.cache_motion_predictions = torch.zeros_like(track_instances.motion_predictions)
        track_instances.cache_motion_feats = torch.zeros_like(query_embeds)

        """History Reasoning"""
        # embeddings
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
        # positions
        track_instances.hist_xyz = torch.zeros(
            (len(track_instances), self.hist_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.hist_position_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.hist_bboxes = torch.zeros(
            (len(track_instances), self.hist_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.hist_logits = torch.zeros(
            (len(track_instances), self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.hist_scores = torch.zeros(
            (len(track_instances), self.hist_len), dtype=torch.float, device=device)
        # motion features
        track_instances.hist_motion_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        
        """Future Reasoning"""
        # embeddings
        track_instances.fut_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.fut_padding_masks = torch.ones(
            (len(track_instances), self.fut_len), dtype=torch.bool, device=device)
        # positions
        track_instances.fut_xyz = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.fut_position_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.fut_bboxes = torch.zeros(
            (len(track_instances), self.fut_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.fut_logits = torch.zeros(
            (len(track_instances), self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.fut_scores = torch.zeros(
            (len(track_instances), self.fut_len), dtype=torch.float, device=device)
        
        return track_instances
    
    def _init_inf_tracks(self, inf_dict):
        # import pdb;pdb.set_trace()
        track_instances = Instances((1, 1))
        device = inf_dict['ref_pts'].device
        
        """Detection queries"""
        # Infra detetion 결과를 track instance 기본 필드로 옮김
        # reference points, query embeds, and query targets (features)
        track_instances.ref_pts = inf_dict['ref_pts'].clone()
        track_instances.query_embeds = inf_dict['query_embeds'].clone()
        track_instances.query_feats = inf_dict['query_feats'].clone()
        track_instances.cache_motion_feats = inf_dict['cache_motion_feats'].clone()
        track_instances.pred_boxes = inf_dict['pred_boxes'].clone()
        """Tracking information"""
        # track instance 초기화
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)
        
        """Current frame information"""
        # classification scores
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (len(track_instances)), dtype=torch.float, device=device)
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # motion prediction, not normalized
        track_instances.motion_predictions = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        
        """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        track_instances.cache_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # track_instances.cache_bboxes = torch.zeros(
        #     (len(track_instances), self.pts_bbox_head.code_size), dtype=torch.float, device=device)
        track_instances.cache_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        """
        현재 프레임 cache는 infra dict 입력을 그대로 사용
        cache는 reasoning이 업데이트 할 대상이기 때문에 원본과 분리해서 유지
        """
        track_instances.cache_ref_pts = inf_dict['ref_pts'].clone()
        track_instances.cache_query_embeds = inf_dict['query_embeds'].clone()
        track_instances.cache_query_feats = inf_dict['query_feats'].clone()
        track_instances.cache_motion_predictions = torch.zeros_like(track_instances.motion_predictions)
        track_instances.cache_bboxes = inf_dict['pred_boxes'].clone()
        """History Reasoning"""
        # embeddings
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
        # positions
        track_instances.hist_xyz = torch.zeros(
            (len(track_instances), self.hist_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.hist_position_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.hist_bboxes = torch.zeros(
            (len(track_instances), self.hist_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.hist_logits = torch.zeros(
            (len(track_instances), self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.hist_scores = torch.zeros(
            (len(track_instances), self.hist_len), dtype=torch.float, device=device)
        # motion features
        track_instances.hist_motion_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        
        """Future Reasoning"""
        # embeddings
        track_instances.fut_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.fut_padding_masks = torch.ones(
            (len(track_instances), self.fut_len), dtype=torch.bool, device=device)
        # positions
        track_instances.fut_xyz = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.fut_position_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.fut_bboxes = torch.zeros(
            (len(track_instances), self.fut_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.fut_logits = torch.zeros(
            (len(track_instances), self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.fut_scores = torch.zeros(
            (len(track_instances), self.fut_len), dtype=torch.float, device=device)
        
        # follow the vehicle setting
        track_instances.cache_query_feats = track_instances.query_feats.clone()
        track_instances.cache_ref_pts = track_instances.ref_pts.clone()
        track_instances.cache_query_embeds = track_instances.query_embeds.clone()
        
        # [:,1:] => [N, hist_len-1] => True, 뒤에 0(False) 한칸 붙임 => 마지막 time stpe만 valid
        track_instances.hist_padding_masks = torch.cat((
            track_instances.hist_padding_masks[:, 1:], 
            torch.zeros((len(track_instances), 1), dtype=torch.bool, device=device)), 
            dim=1)
        
        # hist_embeds는 전부 0이었음, 마지막에 현재 cache query feature를 붙임
        track_instances.hist_embeds = torch.cat((
            track_instances.hist_embeds[:, 1:, :], track_instances.cache_query_feats[:, None, :]), dim=1)
        
        # 현재 ref_pts를 history 마지막에 저장
        track_instances.hist_xyz = torch.cat((
            track_instances.hist_xyz[:, 1:, :], track_instances.cache_ref_pts[:, None, :]), dim=1)
        
        # positional embeds도 마지막 칸에 추가
        track_instances.hist_position_embeds = torch.cat((
            track_instances.hist_position_embeds[:, 1:, :], track_instances.cache_query_embeds[:, None, :]), dim=1)
        track_instances.hist_motion_embeds = torch.cat((
            track_instances.hist_motion_embeds[:, 1:, :], track_instances.cache_motion_feats[:, None, :]), dim=1)
        
        # hist buffer 마지막 칸을 현재 프레임으로 채워서 바로 history transformation에 넣을 수 있게 세팅
        return track_instances

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.reference_points.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, 
                    img_metas=img_metas, 
                    prev_bev=prev_bev)
        self.train()
        return prev_bev

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    def load_detection_output_into_cache(self, track_instances: Instances, out): # detection output을 track instance의 현재 프레임 cache에 넘겨줌
        """ Load output of the detection head into the track_instances cache (inplace)
        """
        with torch.no_grad():
            track_scores = out['all_cls_scores'][-1, :].sigmoid().max(dim=-1).values # query마다 confidence score 생성, [-1,:] => 마지막 decoder layer 결과만 사용
        track_instances.cache_scores = track_scores.clone()
        track_instances.cache_logits = out['all_cls_scores'][-1].clone() # 마지막 layer의 raw logits 저장
        track_instances.cache_query_feats = out['query_feats'][-1].clone() # 마지막 decoder layer의 query hidden state (다음 단계 STReasoner에서 history reasoning 입력으로 사용)
        track_instances.cache_ref_pts = out['ref_pts'].clone() # 현재 frame query의 reference points
        track_instances.cache_bboxes = out['all_bbox_preds'][-1].clone() # 마지막 decoder layer의 bbox prediction 저장
        track_instances.cache_query_embeds = self.query_embedding(pos2posemb3d(track_instances.cache_ref_pts))
        return track_instances
    
    def frame_summarization(self, track_instances, tracking=False):
        """
        [frame_summarization]
        -------------------------------------------------------------
        목적:
        - SpatialTemporalReasoner(STReasoner) 이후, cache_*에 저장된 "현재 프레임의 최신 결과"를
            track_instances의 메인 필드(pred_*, ref_pts, query_feats 등)로 복사/갱신한다.
        - 즉, "현재 프레임에서 확정된 상태"를 track_instances에 반영해서
            다음 프레임에서 reference point update / tracking을 계속할 수 있도록 만드는 단계.

        입력:
        track_instances : Instances 객체 (num_query 개수만큼 query 상태를 담고 있음)
        tracking=False  : True면 inference/track 모드, False면 training 모드

        주요 사용되는 필드 의미:
        - cache_* : 이번 프레임에서 detection + STReasoner가 만든 임시 결과 (현재 프레임 최신)
            cache_logits      : [N, num_cls] class logits (raw)
            cache_scores      : [N] objectness/score (sigmoid(max(logits)))
            cache_bboxes      : [N, box_dim] bbox 예측 (보통 denormalize 상태)
            cache_ref_pts     : [N, 3] 다음 layer/프레임에서 쓰는 reference point (normalize)
            cache_query_feats : [N, C] query hidden state feature
            cache_query_embeds: [N, C] positional embedding (ref_pts 기반)
            cache_motion_predictions: [N, fut_len, 3] 미래 motion delta 예측

        - pred_* : track_instances가 공식적으로 들고 있는 "현재 frame 결과"
            pred_logits : [N, num_cls]
            scores      : [N]
            pred_boxes  : [N, box_dim]

        - ref_pts / query_feats / query_embeds :
            다음 프레임 tracking에서 그대로 재사용되는 핵심 상태
        """

        # =========================================================
        # 1) active_mask 정의: "어떤 query를 살려서 업데이트할지" 결정
        # =========================================================

        # [Inference / Tracking 모드]
        if tracking:
            """
            inference에서는 모든 query를 업데이트하지 않고,
            점수가 충분히 높은 query만 "track 상태로 기록/유지"한다.

            record_threshold 이상인 query만 active로 취급
            - runtime_tracker.record_threshold는 보통 0.4 같은 값
            - 낮은 점수 query는 노이즈로 보고 업데이트하지 않을 수 있음
            """
            active_mask = (track_instances.cache_scores >= self.runtime_tracker.record_threshold)

            # debug용 출력 가능
            # print(f"update instance: {track_instances.obj_idxes[active_mask]}")

        # [Training 모드]
        else:
            """
            training에서는 모든 query를 업데이트 대상으로 두는 편.
            이유:
            - loss 계산 및 학습 안정성을 위해 일단 전부 업데이트하고
            - 이후에 get_active_mask(matched_gt_idxes + IoU) 같은 로직에서 진짜 active만 선택함.
            """
            track_instances.pred_boxes = track_instances.cache_bboxes.clone()   # cache 결과를 pred로 복사
            track_instances.pred_logits = track_instances.cache_logits.clone()
            track_instances.scores = track_instances.cache_scores.clone()

            # training에서는 보통 전체 query 다 업데이트하도록 >= 0.0 사용
            active_mask = (track_instances.cache_scores >= 0.0)

        # =========================================================
        # 2) cache 결과를 pred / 상태 필드로 반영 (active query만)
        # =========================================================

        # (1) classification logits 갱신
        track_instances.pred_logits[active_mask] = track_instances.cache_logits[active_mask]

        # (2) score 갱신
        track_instances.scores[active_mask] = track_instances.cache_scores[active_mask]

        # (3) bbox prediction 갱신
        track_instances.pred_boxes[active_mask] = track_instances.cache_bboxes[active_mask]

        # (4) reference point 갱신
        """
        ref_pts는 다음 프레임에서 tracking query의 "초기 위치(anchor)" 역할을 한다.
        cache_ref_pts는 이번 프레임 reasoning/box refine 이후 최신 ref point이므로
        active query만 업데이트해서 ref_pts에 반영한다.
        """
        ref_pts = track_instances.ref_pts.clone()
        ref_pts[active_mask] = track_instances.cache_ref_pts[active_mask]
        track_instances.ref_pts = ref_pts

        # (5) positional embedding(query_embeds) 갱신
        """
        query_embeds는 ref_pts로부터 생성된 positional embedding이므로
        ref_pts가 바뀌면 query_embeds도 함께 최신 값으로 바꿔야 한다.
        """
        query_embeds = track_instances.query_embeds.clone()
        query_embeds[active_mask] = track_instances.cache_query_embeds[active_mask]
        track_instances.query_embeds = query_embeds

        # (6) query feature(query_feats) 갱신
        """
        query_feats는 Transformer decoder가 만든 hidden state이고,
        다음 프레임 temporal reasoning / tracking에 직접 쓰이는 핵심 feature다.
        """
        query_feats = track_instances.query_feats.clone()
        query_feats[active_mask] = track_instances.cache_query_feats[active_mask]
        track_instances.query_feats = query_feats

        # (7) motion prediction 갱신
        """
        cache_motion_predictions는 STReasoner의 future reasoning/motion head 등이 만든 결과.
        active query만 최신 motion prediction으로 교체.
        """
        track_instances.motion_predictions[active_mask] = track_instances.cache_motion_predictions[active_mask]

        # =========================================================
        # 3) Future Reasoning이 켜져 있다면, fut_xyz / fut_bboxes 업데이트
        # =========================================================
        if self.STReasoner.future_reasoning:
            """
            motion_predictions: [N_active, fut_len, 3]
            - 보통 (dx, dy, dz) 형태의 step-wise motion delta
            - 여기서는 x,y만 누적해서 미래 위치를 만듦

            fut_xyz/fut_bboxes는:
            - 다음 프레임 reference point update에서 쓰이거나
            - forecasting loss 계산에서 쓰이거나
            - planning 모듈에서 trajectory로 쓰일 수 있음
            """

            # active query들에 대해서만 motion prediction을 가져옴
            motion_predictions = track_instances.motion_predictions[active_mask]

            # (1) fut_xyz 초기화: 현재 ref_pts를 fut_len 만큼 복사해서 시작점으로 만든다
            #     shape: [N_active, fut_len, 3]
            track_instances.fut_xyz[active_mask] = (
                track_instances.ref_pts[active_mask].clone()[:, None, :]
                .repeat(1, self.fut_len, 1)
            )

            # (2) fut_bboxes 초기화: 현재 pred_boxes를 fut_len 만큼 복사
            #     shape: [N_active, fut_len, box_dim]
            track_instances.fut_bboxes[active_mask] = (
                track_instances.pred_boxes[active_mask].clone()[:, None, :]
                .repeat(1, self.fut_len, 1)
            )

            # (3) motion delta를 time axis로 누적합해서 "미래까지의 총 이동량"으로 변환
            #     예: [dx1, dx2, dx3] -> [dx1, dx1+dx2, dx1+dx2+dx3]
            # detach 이유: 미래 예측을 다음 상태 업데이트에 쓰더라도 gradient 경로를 끊어 안정화
            motion_add = torch.cumsum(motion_predictions.clone().detach(), dim=1)  # [N_active, fut_len, 3]

            # (4) fut_xyz는 normalized 좌표 기반이므로,
            #     world delta(motion_add)를 normalized delta로 바꿔서 더해줘야 함
            motion_add_normalized = motion_add.clone()
            motion_add_normalized[..., 0] /= (self.pc_range[3] - self.pc_range[0])  # x range로 정규화
            motion_add_normalized[..., 1] /= (self.pc_range[4] - self.pc_range[1])  # y range로 정규화

            # (5) normalized future ref point 업데이트 (x,y만)
            track_instances.fut_xyz[active_mask, :, 0] += motion_add_normalized[..., 0]
            track_instances.fut_xyz[active_mask, :, 1] += motion_add_normalized[..., 1]

            # (6) fut_bboxes는 world 좌표(box center)로 저장되는 경우가 많아서
            #     motion_add(world delta)를 그대로 더해줌
            track_instances.fut_bboxes[active_mask, :, 0] += motion_add[..., 0]  # cx
            track_instances.fut_bboxes[active_mask, :, 1] += motion_add[..., 1]  # cy

        # 최종적으로 업데이트된 track_instances 반환
        return track_instances
    
    def loss_single_batch(self, gt_bboxes_3d, gt_labels_3d, gt_inds, pred_dict):
        """
        [한 프레임의 detection loss 계산 + matching 결과를 track_instances에 기록하는 함수]

        역할 요약:
        1) GT 박스/라벨/ID(gt_inds)를 Instances 형태로 구성해서 criterion에 등록
        2) DETR-style decoder layer별 예측값(cls, bbox)을 track_instances에 넣어줌
        3) 각 decoder layer마다 Hungarian matching을 수행해서
        - 어떤 query가 어떤 GT에 매칭되는지 결정
        - 매칭된 결과를 track_instances에 저장 (matched_gt_idxes, iou 등)
        4) 최종 decoder layer(nb_dec-1)의 track_instances를 리턴

        입력:
        gt_bboxes_3d: list 형태 (batch 차원 포함)
            - 여기서는 bs=1 구조라서 gt_bboxes_3d[0]을 사용
            - gt_bboxes_3d[0].tensor: [N_gt, 9 또는 10] 형태의 GT box

        gt_labels_3d: list
            - gt_labels_3d[0]: [N_gt] 각 GT box의 class id

        gt_inds: list
            - gt_inds[0]: [N_gt] 각 GT 객체의 "고유 object id"
            - tracking에서 동일 객체를 추적하기 위해 사용됨

        pred_dict: detector output과 track_instances가 묶인 dict
            - pred_dict['all_cls_scores']: [num_decoder_layers, num_query, num_cls]
            - pred_dict['all_bbox_preds']: [num_decoder_layers, num_query, box_dim]
            - pred_dict['track_instances']: 현재 프레임의 query container (Instances)
        """

        # ------------------------------------------------------------
        # 0) GT Instances 구성 (criterion이 matching/loss 계산할 수 있게 준비)
        # ------------------------------------------------------------

        gt_instances_list = []  # criterion이 받는 형태: GT Instances 리스트
        device = self.reference_points.weight.device  # 모델이 올라간 device

        # GT도 track_instances와 같은 "Instances" 포맷으로 만들어준다.
        # Instances((1,1))은 보통 더미 shape로 생성한 뒤 필드를 직접 채우는 방식
        gt_instances = Instances((1, 1))

        # GT 박스 가져오기 (bs=1이므로 첫 번째만 사용)
        # gt_bboxes_3d[0].tensor: 실제 3D box tensor
        boxes = gt_bboxes_3d[0].tensor.to(device)

        # normalize_bbox:
        # - DETR 계열에서는 박스를 [0~1] 범위의 normalized 좌표로 두고 학습하는 경우가 많음
        # - pc_range(예: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]) 기준으로 normalize
        boxes = normalize_bbox(boxes, self.pc_range)

        # GT Instances 내부 필드 채우기
        gt_instances.boxes = boxes                 # [N_gt, box_dim] normalized GT boxes
        gt_instances.labels = gt_labels_3d[0]      # [N_gt] GT class label
        gt_instances.obj_ids = gt_inds[0]          # [N_gt] GT object tracking id

        # criterion은 list로 GT를 받도록 설계된 경우가 많음
        gt_instances_list.append(gt_instances)

        # criterion 내부에 GT 정보 저장 (clip 단위/프레임 단위 초기화)
        # ex) matcher가 GT를 알고 있어야 query↔GT assignment 가능
        self.criterion.initialize_for_single_clip(gt_instances_list)

        # ------------------------------------------------------------
        # 1) detector 예측 결과 꺼내오기
        # ------------------------------------------------------------
        output_classes = pred_dict['all_cls_scores']  # [nb_dec, num_query, num_cls]
        output_coords  = pred_dict['all_bbox_preds']  # [nb_dec, num_query, box_dim]
        track_instances = pred_dict['track_instances']  # query container

        # ------------------------------------------------------------
        # 2) track_scores 계산 (objectness처럼 쓰는 score)
        # ------------------------------------------------------------
        # DETR류에서는 per-query class logit이 나오므로
        # sigmoid 후 가장 높은 클래스 확률(max)을 그 query의 confidence로 사용
        # track_scores: [num_query]
        # with torch.no_grad():
        #   - score 계산은 loss backprop에 불필요하므로 gradient 끊음
        with torch.no_grad():
            track_scores = output_classes[-1].sigmoid().max(dim=-1).values
            # output_classes[-1] -> 마지막 decoder layer output 사용
            # .sigmoid() -> multi-label 형태 확률화
            # .max(dim=-1) -> 클래스 중 가장 높은 값 = query confidence
            # .values -> max값만 가져옴 (argmax는 버림)

        # ------------------------------------------------------------
        # 3) decoder layer 개수 구하기
        # ------------------------------------------------------------
        # output_classes.size(0) == nb_dec (decoder layer 수)
        nb_dec = output_classes.size(0)

        # ------------------------------------------------------------
        # 4) 각 decoder layer별 track_instances 준비
        # ------------------------------------------------------------
        # 중요한 이유:
        # - decoder layer마다 예측이 달라지고, 각 layer별로 matching/loss를 계산하는 구조
        # - 하지만 track_instances는 "하나의 객체"라서,
        #   layer별로 pred_boxes/pred_logits 등을 저장하려면 복사본이 필요함
        #
        # 그래서:
        #   - 앞의 (nb_dec-1)개 layer는 _copy_tracks_for_loss()로 "복제본"을 만들고
        #   - 마지막 layer는 원본 track_instances를 그대로 사용
        #
        # copy는 obj_idxes/matched_gt_idxes 같은 tracking 상태는 유지하면서
        # pred_logits/pred_boxes 같은 값은 새로 채우는 용도로 쓰임
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances_list.append(track_instances)  # 마지막 layer는 원본을 그대로 사용

        # ------------------------------------------------------------
        # 5) decoder layer별 matching 수행 (Hungarian assignment)
        # ------------------------------------------------------------
        single_out = {}  # criterion.match_for_single_frame에 넣을 dict 형태

        for i in range(nb_dec):
            # i번째 decoder layer에 대응하는 track_instances 가져오기
            track_instances_tmp = track_instances_list[i]

            # (1) scores 기록
            # track_instances_tmp.scores: [num_query]
            # 여기서 score는 "query confidence"로 활용됨
            track_instances_tmp.scores = track_scores

            # (2) logits 기록
            # output_classes[i]: [num_query, num_cls]
            # query별 classification prediction
            track_instances_tmp.pred_logits = output_classes[i]

            # (3) bbox 기록
            # output_coords[i]: [num_query, box_dim]
            # query별 bbox regression prediction
            track_instances_tmp.pred_boxes = output_coords[i]

            # criterion 내부 matcher는 "single_out['track_instances']"를 보고
            # query 예측과 GT를 비교해서 matching 수행함
            single_out["track_instances"] = track_instances_tmp

            # match_for_single_frame:
            # - Hungarian matching 수행
            # - query ↔ GT 최적 할당을 계산
            # - 그 결과를 track_instances_tmp 내부 필드에 저장
            #   (예: matched_gt_idxes, iou, obj_idxes 등)
            #
            # if_step = True인 경우:
            # - 보통 마지막 decoder layer에서만 최종 step으로 처리
            # - 예: tracking id 업데이트 / loss weight 적용 / 기록 등
            track_instances_tmp, matched_indices = self.criterion.match_for_single_frame(
                single_out,
                i,
                if_step=(i == (nb_dec - 1))  # 마지막 decoder layer만 True
            )

        # ------------------------------------------------------------
        # 6) 최종 decoder layer의 track_instances 반환
        # ------------------------------------------------------------
        # 즉, "matching과 loss 계산이 반영된 최종 layer 결과"를 반환
        return track_instances_tmp

    def forward_loss_prediction(self, 
                                active_track_instances,
                                gt_trajs,
                                gt_traj_masks,
                                instance_inds,
                                img_metas):
        active_gt_trajs, active_gt_traj_masks = list(), list()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(
            instance_inds.detach().cpu().numpy().tolist())}

        active_gt_trajs = torch.ones_like(active_track_instances.motion_predictions)
        active_gt_trajs[..., -1] = 0.0
        active_gt_traj_masks = torch.zeros_like(active_gt_trajs)[..., 0]

        for track_idx, id in enumerate(active_track_instances.obj_idxes):
            cpu_id = id.cpu().numpy().tolist()
            if cpu_id not in obj_idx_to_gt_idx.keys():
                continue
            index = obj_idx_to_gt_idx[cpu_id]
            traj = gt_trajs[index:index+1, :self.fut_len + 1, :]

            gt_motion = traj[:, torch.arange(1, self.fut_len + 1)] - traj[:, torch.arange(0, self.fut_len)]
            active_gt_trajs[track_idx: track_idx + 1] = gt_motion
            active_gt_traj_masks[track_idx: track_idx + 1] = \
                gt_traj_masks[index: index+1, 1: self.fut_len + 1] * gt_traj_masks[index: index+1, : self.fut_len]
        
        loss_dict = self.criterion.loss_prediction(active_gt_trajs[..., :2],
                                                   active_gt_traj_masks,
                                                   active_track_instances.cache_motion_predictions[..., :2])
        return loss_dict

    def update_reference_points(self, track_instances, time_delta=None, use_prediction=True, tracking=False):
        """Update the reference points according to the motion prediction/velocities
        """
        track_instances = self.STReasoner.update_reference_points(
            track_instances, time_delta, use_prediction, tracking)
        return track_instances
    
    def update_ego(self, track_instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        """
        track_instances = self.STReasoner.update_ego(track_instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2)
        return track_instances
    
    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train_bs(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None,
        prev_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        gt_forecasting_locs=None,
        gt_forecasting_masks=None,
        **kwargs,
    ):
        """
        [단일 프레임 학습 forward]
        - seq_mode에서 streaming train을 할 때, "현재 프레임 1장"에 대해서만
        tracking + detection loss + (optional) motion/future/history reasoning loss를 계산하고,
        다음 프레임으로 넘길 track_instances(active instances) 를 구성한다.
        - 기본적으로 BS>1도 처리하는 형태로 작성되어 있지만, 주석대로 clip 단위 train에서 BS=1을 가정하는 코드가 많음.

        Args:
            img: [B, num_cam, 3, H, W]
            img_metas: batch 크기 만큼의 meta dict list
            track_instances: 길이 B짜리 list
                - 각 원소는 Instances이며, 이전 프레임에서 살아남은(active) track query 들만 들어있음
                - None이면 첫 프레임으로 취급하여 empty tracks를 생성함
            prev_bev: 이전 프레임의 BEV feature (streaming train에서는 외부에서 유지)
            l2g_r1/t1, l2g_r2/t2: 이전 프레임과 현재 프레임 사이 ego motion 보정용 (local->global 변환)
            time_delta: 프레임 간 시간 차이
            veh2inf_rt: 협력(cooperation)일 때 vehicle 좌표계 -> infra 좌표계 정렬을 위한 변환
            gt_*: detection/tracking loss, motion forecasting loss를 위한 GT 정보
            kwargs: cooperation에서 infra model 쿼리 결과(혹은 저장된 결과)가 넘어오는 곳

        Returns:
            out: {
                'track_instances': [B개],   # 다음 프레임으로 넘길 active track instances
                'bev_embed': bev_embed,      # 현재 프레임 BEV embedding
                'bev_pos': bev_pos,          # BEV positional embedding
            }
            losses: batch 평균 처리된 loss dict
        """

        # track_instances는 batch 크기만큼 리스트로 들어오며, self.batch_size와 일치해야 한다.
        assert self.batch_size == len(track_instances)

        # =========================================================================
        # (A) 이전 프레임에서 넘어온 track_instances를 "현재 프레임 입력 query"로 재구성
        #     - 이전 프레임 active query들을 기반으로 reference point/ego를 업데이트
        #     - 부족한 개수는 empty queries로 채워서 "고정 num_query" 형태로 맞춘다.
        # =========================================================================
        for i in range(self.batch_size):
            prev_active_track_instances = track_instances[i]  # 이전 프레임에서 살아남은(active) query들

            # (A-1) 첫 프레임이거나 scene이 바뀌어서 이전 track이 없으면, 완전 초기 query set 생성
            if prev_active_track_instances is None:
                # num_query개의 빈 트랙(참조점/ref_pts, query_embeds, query_feats 포함) 생성
                track_instances[i] = self._generate_empty_tracks()

            # (A-2) 이전 프레임 track이 있다면, 다음 프레임으로 "예측 기반 ref 갱신 + ego 보정" 수행
            else:
                # 1) 시간차(time_delta)를 이용하여 motion prediction(또는 velocity)을 기반으로
                #    reference point(ref_pts)를 다음 프레임 위치로 업데이트
                prev_active_track_instances = self.update_reference_points(
                    prev_active_track_instances,
                    time_delta[i],
                    use_prediction=self.motion_prediction_ref_update,  # 예측값 기반 ref update 사용 여부
                    tracking=False
                )

                # 2) ego motion 보정이 켜져있다면, local->global 변환(l2g_r/t)로 이전 track들을 현재 ego 기준으로 변환
                if self.if_update_ego:
                    prev_active_track_instances = self.update_ego(
                        prev_active_track_instances,
                        l2g_r1[i], l2g_t1[i], l2g_r2[i], l2g_t2[i]
                    )

                # 3) ref_pts가 바뀌었으니 positional embedding(query_embeds)을 최신 ref_pts 기준으로 동기화
                #    (query_embedding(pos2posemb3d(ref_pts)) 같은 방식으로 업데이트)
                prev_active_track_instances = self.STReasoner.sync_pos_embedding(
                    prev_active_track_instances,
                    self.query_embedding
                )

                # 4) 현재 프레임에서 detection head에 넣을 query 개수는 고정(num_query)이므로
                #    active query 수가 적으면 empty query로 채워 넣어야 한다.
                empty_track_instances = self._generate_empty_tracks()
                full_length = len(empty_track_instances)             # 전체 query 개수 (예: 900)
                active_length = len(prev_active_track_instances)     # 살아남은 query 개수

                # active query가 존재하면, empty query에서 (full-active) 개를 랜덤 샘플링하여 가져온다.
                # 즉, "empty 일부 + prev_active 전부"를 concat해서 최종 num_query로 맞춤
                if active_length > 0:
                    random_index = torch.randperm(full_length)
                    selected = random_index[:full_length - active_length]
                    empty_track_instances = empty_track_instances[selected]

                # empty + active를 이어붙여서 "num_query"로 구성된 Instances 생성
                out_track_instances = Instances.cat([empty_track_instances, prev_active_track_instances])
                track_instances[i] = out_track_instances

            # (A-3) optional: query 순서를 섞어서 학습 시 편향을 줄이는 용도(데이터 증강 느낌)
            if self.shuffle:
                shuffle_index = torch.randperm(len(track_instances[i]))
                track_instances[i] = track_instances[i][shuffle_index]

        # =========================================================================
        # (B) 현재 프레임 이미지 -> BEV 생성 (BEVFormer encoder)
        # =========================================================================
        # prev_bev가 들어오면 temporal fusion 방식으로 현재 bev_embed를 만들 수 있음
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        # =========================================================================
        # (C) Detection Head 수행: track query를 input query로 넣어서 디텍션 출력 얻기
        # =========================================================================
        # query_feats/query_embeds/ref_pts는 Instances에 저장되어 있던 값을 꺼내 stack 한다.
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            query_feats=torch.stack([ins.query_feats for ins in track_instances]),   # [B, num_query, C]
            query_embeds=torch.stack([ins.query_embeds for ins in track_instances]), # [B, num_query, C]
            ref_points=torch.stack([ins.ref_pts for ins in track_instances]),        # [B, num_query, 3]
            img_metas=img_metas,
        )

        # det_output은 decoder layer별 결과를 포함
        output_classes = det_output["all_cls_scores"]   # [num_layers, B, num_query, num_cls]
        output_coords  = det_output["all_bbox_preds"]   # [num_layers, B, num_query, box_dim]
        last_ref_pts   = det_output["last_ref_points"]  # [B, num_query, 3] (마지막 layer의 ref pts)
        query_feats    = det_output["query_feats"]      # [num_layers, B, num_query, C]

        # =========================================================================
        # (D) Loss / Reasoning을 위한 loop (batch별 처리)
        # =========================================================================
        losses = {}  # batch 전체 loss 누적용
        out = {
            'track_instances': [],  # 다음 프레임으로 넘길 active instances들
            'bev_embed': bev_embed,
            'bev_pos': bev_pos,
        }
        avg_factors = {}  # loss normalization을 위한 avg_factor 누적

        for j in range(self.batch_size):
            # ---------------------------------------------
            # (D-0) 현재 batch의 예측 결과 패키징
            # ---------------------------------------------
            cur_loss = dict()
            cur_track_instances = track_instances[j]

            # 현재 batch j에 대한 detection head output slice
            cur_out = {
                'all_cls_scores': output_classes[:, j, :, :],  # [num_layers, num_query, num_cls]
                'all_bbox_preds': output_coords[:, j, :, :],   # [num_layers, num_query, box_dim]
                'ref_pts': last_ref_pts[j, :, :],              # [num_query, 3]
                'query_feats': query_feats[:, j, :, :],        # [num_layers, num_query, C]
            }

            # ---------------------------------------------
            # (D-1) detection 결과를 track_instances.cache_* 에 기록
            #       (cache_logits/cache_bboxes/cache_scores/cache_query_feats/cache_ref_pts 등)
            # ---------------------------------------------
            cur_track_instances = self.load_detection_output_into_cache(cur_track_instances, cur_out)
            cur_out['track_instances'] = cur_track_instances

            # ---------------------------------------------
            # (D-2) Detection loss 계산 (cooperation이 아닐 때만)
            #       - Hungarian matching을 통해 query <-> GT를 매칭하고 loss 계산
            # ---------------------------------------------
            if not self.is_cooperation:
                cur_track_instances = self.loss_single_batch(
                    gt_bboxes_3d[j],
                    gt_labels_3d[j],
                    gt_inds[j],
                    cur_out
                )
                # criterion 내부에서 계산된 losses_dict를 현재 프레임 loss에 합침
                cur_loss.update(self.criterion.losses_dict)

            # ---------------------------------------------
            # (D-3) optional: motion feature 추출 (bbox 기반)
            #       - STReasoner나 future prediction 등에 사용할 motion embedding 생성
            # ---------------------------------------------
            if self.is_motion:
                cur_track_instances = self.MotionExtractor(cur_track_instances, img_metas[j])

            # ---------------------------------------------
            # (D-4) Cooperation mode: Infra side query를 받아서 inf_instances 구성
            #       - kwargs에는 infra model 출력(혹은 미리 저장된 출력)이 들어온다.
            #       - crossview_alignment로 vehicle 좌표계와 infra 좌표계를 정렬(CAA)
            #       - learn_match가 켜져있으면 association label 생성용 GT 매칭도 수행
            # ---------------------------------------------
            inf_instances = None
            if self.is_cooperation:
                # infra에서 전달된 query 출력 dict 구성
                inf_dcit = {
                    'query_feats': kwargs['query_feats'][j][0],
                    'query_embeds': kwargs['query_embeds'][j][0],
                    'cache_motion_feats': kwargs['cache_motion_feats'][j][0],
                    'ref_pts': kwargs['ref_pts'][j][0],
                    'pred_boxes': kwargs['pred_boxes'][j][0],
                }

                # infra query가 존재할 때만 사용
                if inf_dcit['query_feats'].shape[0] > 0:
                    # vehicle->infra 변환(veh2inf_rt)을 이용해 latent feature alignment 수행
                    inf_dcit = self.crossview_alignment(inf_dcit, veh2inf_rt[j])

                    # infra query들을 Instances 컨테이너로 초기화하고 history buffer 세팅
                    inf_instances = self._init_inf_tracks(inf_dcit)

                # veh-inf association 학습(learn_match)일 때 GT 기반 positive matrix 생성
                if self.STReasoner.learn_match:
                    # vehicle query 중 confidence가 일정 threshold 이상인 것만 association에 사용
                    # (배경/빈 query를 줄여서 매칭 노이즈 감소)
                    mask = cur_track_instances.cache_scores > self.STReasoner.veh_thre

                    veh_boxes = cur_track_instances[mask].cache_bboxes.clone()
                    inf_boxes = inf_instances.cache_bboxes.clone()

                    # GT 기준으로 (veh<->GT) & (inf<->GT) Hungarian matching 후
                    # 같은 GT에 붙은 veh-inf pair를 label=1로 만든 matrix 생성
                    asso_label = self.STReasoner._gen_asso_label(
                        gt_bboxes_3d[j],
                        inf_boxes,
                        veh_boxes,
                        img_metas[j]['sample_idx']
                    )

            # ---------------------------------------------
            # (D-5) Spatial-Temporal Reasoning 수행 (STReasoner)
            #       - cur_track_instances.cache_* 를 업데이트/refine
            #       - cooperation이면 inf_instances까지 함께 reasoning
            #       - affinity: veh-inf 유사도(association) 예측값
            # ---------------------------------------------
            cur_track_instances, affinity = self.STReasoner(cur_track_instances, inf_instances)

            # ---------------------------------------------
            # (D-6) Cooperation mode에서 fused 결과로 detection loss 다시 계산
            #       - STReasoner가 업데이트한 cache_logits/cache_bboxes를 "최종 예측"으로 보고 loss 계산
            #       - association loss(affinity vs asso_label)도 추가 학습 가능
            # ---------------------------------------------
            if self.is_cooperation:
                cur_out = {
                    'all_cls_scores': cur_track_instances.cache_logits[None, :, :],  # [1, num_query, num_cls]
                    'all_bbox_preds': cur_track_instances.cache_bboxes[None, :, :],  # [1, num_query, box_dim]
                    'track_instances': cur_track_instances
                }

                # fused 예측 기반으로 loss 계산
                cur_track_instances = self.loss_single_batch(
                    gt_bboxes_3d[j],
                    gt_labels_3d[j],
                    gt_inds[j],
                    cur_out
                )

                # fused loss는 prefix를 붙여 구분
                prefix = 'fused_'
                cur_loss.update({prefix + key: value for key, value in self.criterion.losses_dict.items()})

                # association 학습이 켜져있으면 affinity를 focal loss로 supervision
                if self.STReasoner.learn_match:
                    # affinity 또는 label이 비정상/empty면 loss=0 처리
                    if affinity.shape[0] == 0 or affinity.shape[1] == 0 or torch.all(asso_label.eq(0)):
                        loss_focal = torch.tensor(0.0, requires_grad=True).to(affinity.device)
                    else:
                        affinity = affinity.view(-1, 1)
                        target = asso_label.view(-1, 1).float()
                        loss_focal = self.asso_loss_focal['loss_weight'] * sigmoid_focal_loss(
                            affinity, target,
                            alpha=self.asso_loss_focal['alpha'],
                            gamma=self.asso_loss_focal['gamma'],
                            reduction='mean'
                        )

                    # avg_factor는 보통 1로 두고 loss만 추가하는 형태
                    cur_loss.update({
                        'asso_loss': loss_focal,
                        'asso_avg_factor': torch.tensor([1.0], device=affinity.device)
                    })

            # ---------------------------------------------
            # (D-7) History reasoning loss (memory bank loss)
            #       - 과거 히스토리를 잘 유지/활용하도록 하는 loss
            # ---------------------------------------------
            if self.STReasoner.history_reasoning:
                loss_hist = self.criterion.loss_mem_bank(
                    gt_bboxes_3d[j],
                    gt_labels_3d[j],
                    gt_inds[j],
                    cur_track_instances
                )
                cur_loss.update(loss_hist)

            # ---------------------------------------------
            # (D-8) Future reasoning loss (motion forecasting)
            #       - obj_idxes >= 0 인 active track에 대해서만 미래 궤적 예측 loss를 건다.
            # ---------------------------------------------
            if self.STReasoner.future_reasoning:
                active_mask = (cur_track_instances.obj_idxes >= 0)
                loss_fut = self.forward_loss_prediction(
                    cur_track_instances[active_mask],
                    gt_forecasting_locs[j][0],
                    gt_forecasting_masks[j][0],
                    gt_inds[j][0],
                    img_metas[j]
                )
                cur_loss.update(loss_fut)

            # =========================================================================
            # (E) 다음 프레임을 위한 준비: cache -> pred 로 반영 + active track만 남김
            # =========================================================================
            # frame_summarization:
            # - cache_*에 있는 "현재 프레임 최종 결과"를 pred_*로 옮기고
            # - ref_pts/query_feats/query_embeds 등도 갱신
            cur_track_instances = self.frame_summarization(cur_track_instances, tracking=False)

            # runtime_tracker가 정의한 기준으로 active query 결정 (training용)
            active_mask = self.runtime_tracker.get_active_mask(cur_track_instances, training=True)

            # active query에는 track_query_mask를 True로 설정 (추적 중인 query로 표시)
            cur_track_instances.track_query_mask[active_mask] = True

            # 다음 프레임으로 넘길 active instances만 추출
            active_track_instances = cur_track_instances[active_mask]

            # (E-1) random drop: active track 중 일부를 랜덤으로 제거하여 robustness 학습
            if self.random_drop > 0.0:
                active_track_instances = self._random_drop_tracks(active_track_instances)

            # (E-2) false positive 추가: inactive query 중 일부를 FP로 섞어 학습 안정성/강건성 증가
            if self.fp_ratio > 0.0:
                active_track_instances = self._add_fp_tracks(cur_track_instances, active_track_instances)

            # out에는 batch별 "다음 프레임 입력으로 사용할 active instances"만 저장
            out['track_instances'].append(active_track_instances)

            # =========================================================================
            # (F) loss 누적: avg_factor로 정규화하기 위해 (loss * avg_factor)로 모았다가 마지막에 나눔
            # =========================================================================
            for key, value in cur_loss.items():
                if 'loss' not in key:
                    continue

                # mmdet loss 규약: loss_xxx 와 avg_factor가 함께 있음
                af_key = key.replace('loss', 'avg_factor')
                avg_factor = cur_loss[af_key]

                # losses에 누적 (가중합 형태)
                if key not in losses:
                    losses[key] = value * avg_factor
                    avg_factors[af_key] = avg_factor
                else:
                    losses[key] = losses[key] + value * avg_factor
                    avg_factors[af_key] = avg_factors[af_key] + avg_factor

        # =========================================================================
        # (G) 최종 loss 정규화: 누적된 (loss*avg_factor) / (sum avg_factor)
        # =========================================================================
        for key, value in losses.items():
            af_key = key.replace('loss', 'avg_factor')
            avg_factor = avg_factors[af_key]
            losses[key] = value / avg_factor

        return out, losses

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        drop_probability = self.random_drop
        if drop_probability > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
            track_instances = track_instances[keep_idxes]
        return track_instances
    
    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:
        """
        self.fp_ratio is used to control num(add_fp) / num(active)
        """
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(
            active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]
        num_fp = len(selected_active_track_instances)

        if len(inactive_instances) > 0 and num_fp > 0:
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                # randomly select num_fp from inactive_instances
                # fp_indexes = np.random.permutation(len(inactive_instances))
                # fp_indexes = fp_indexes[:num_fp]
                # fp_track_instances = inactive_instances[fp_indexes]

                # v2: select the fps with top scores rather than random selection
                fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat(
                [active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances
    
    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        # result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict
    

    def forward_track_stream_train(self,
                                    img,
                                    gt_bboxes_3d,
                                    gt_labels_3d,
                                    gt_inds,
                                    gt_forecasting_locs,
                                    gt_forecasting_masks,
                                    l2g_t,
                                    l2g_r_mat,
                                    img_metas,
                                    timestamp,
                                    veh2inf_rt,
                                    **kwargs):
        """
        [Streaming Track Train Forward]
        - seq_mode(스트리밍 학습)에서 한 step(현재 frame)만 forward 하는 함수
        - 이전 프레임의 상태(train_prev_infos)를 이용해서
        시간 차(time_delta), ego pose 변화량(can_bus), 이전 BEV(prev_bev), 이전 track_instances를 세팅하고
        _forward_single_frame_train_bs()로 현재 프레임 학습을 수행한 뒤
        다시 train_prev_infos를 업데이트해서 다음 프레임으로 전달한다.

        입력 텐서/자료 구조 핵심:
        ---------------------------------------------------------
        img: shape 보통 [B, num_frame(=1), num_cam, C, H, W] 형태로 들어오는 경우가 많음
            => 코드에서 img_[0]을 뽑는 걸 보면 num_frame 축이 존재함
        img_metas: [B][num_frame] 구조의 list of dict
            예) img_metas[i][0] : i번째 batch의 현재 프레임 메타
        timestamp: [B][num_frame] 구조
            예) timestamp[i][0] : i번째 batch의 현재 프레임 시간

        l2g_r_mat, l2g_t: 로컬->글로벌 좌표 변환 (ego pose)
            l2g_r_mat[i][0], l2g_t[i][0]이 "현재 frame pose"
        """

        # 배치 크기 검증: 설정한 batch_size와 입력 img의 batch 차원이 일치해야 함
        assert self.batch_size == img.size(0)

        # 각 batch마다 time delta / 이전-현재 ego 변환 정보를 저장할 리스트 초기화
        time_delta = [None] * self.batch_size
        l2g_r1 = [None] * self.batch_size   # 이전 frame rotation
        l2g_t1 = [None] * self.batch_size   # 이전 frame translation
        l2g_r2 = [None] * self.batch_size   # 현재 frame rotation
        l2g_t2 = [None] * self.batch_size   # 현재 frame translation

        # ============================================================
        # 1) batch별로 "이전 프레임 정보"를 기반으로 현재 프레임 입력을 보정
        # ============================================================
        for i in range(self.batch_size):

            # tmp_pos, tmp_angle은 "현재 can_bus 원본값"을 저장해둔다 (나중에 prev_infos 업데이트용)
            # img_metas[i][0]['can_bus'] 구조는 보통:
            #   can_bus[:3]  -> ego position (x,y,z) 또는 (x,y,?)  
            #   can_bus[-1]  -> ego yaw angle (heading)
            tmp_pos = copy.deepcopy(img_metas[i][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[i][0]['can_bus'][-1])

            # ------------------------------------------------------------
            # [Scene reset 조건]
            # 1) scene_token이 바뀐 경우 : 새로운 장면 시작
            # 2) timestamp gap이 너무 큰 경우(>0.5s) : 중간 프레임이 끊겼다고 판단
            # 3) timestamp가 감소한 경우 : 시계열이 뒤틀림(데이터 순서 문제)
            # => 이 경우는 "이전 프레임과 연속성이 없다"고 보고
            #    tracking memory를 초기화한다.
            # ------------------------------------------------------------
            if img_metas[i][0]['scene_token'] != self.train_prev_infos['scene_token'][i] or \
                timestamp[i][0] - self.train_prev_infos['prev_timestamp'][i] > 0.5 or \
                timestamp[i][0] < self.train_prev_infos['prev_timestamp'][i]:

                # 첫 샘플이거나 시계열이 끊겼으므로, 이전 track / prev_bev는 무효
                self.train_prev_infos['track_instances'][i] = None
                self.train_prev_infos['prev_bev'][i] = None

                # 이번 프레임은 "이전 프레임 기반 보정"을 할 수 없으므로 None
                time_delta[i], l2g_r1[i], l2g_t1[i], l2g_r2[i], l2g_t2[i] = None, None, None, None, None

                # can_bus를 0으로 만들어서 "이 프레임은 ego motion이 없다"고 처리
                # (BEVFormer류 temporal alignment에서 첫 프레임 처리를 동일하게 맞추려는 목적)
                img_metas[i][0]['can_bus'][:3] = 0
                img_metas[i][0]['can_bus'][-1] = 0

            else:
                # ------------------------------------------------------------
                # [연속 프레임이라면]
                # 1) time_delta 계산 (현재 - 이전)
                # 2) 이전 ego pose(l2g_r1,t1), 현재 ego pose(l2g_r2,t2) 준비
                # 3) can_bus를 "차분(delta)" 형태로 바꾼다
                #    => 현재 pose - 이전 pose (relative motion)
                # ------------------------------------------------------------

                # time_delta[i] = 현재프레임 timestamp - 이전프레임 timestamp
                time_delta[i] = timestamp[i][0] - self.train_prev_infos['prev_timestamp'][i]
                assert time_delta[i] > 0

                # 이전 frame의 pose
                l2g_r1[i] = self.train_prev_infos['l2g_r_mat'][i]
                l2g_t1[i] = self.train_prev_infos['l2g_t'][i]

                # 현재 frame의 pose (입력으로 들어온 l2g_r_mat/l2g_t에서 추출)
                # l2g_r_mat[i][0] : i번째 batch의 현재 frame rotation
                # l2g_t[i][0]     : i번째 batch의 현재 frame translation
                l2g_r2[i] = l2g_r_mat[i][0]
                l2g_t2[i] = l2g_t[i][0]

                # can_bus를 절대 위치가 아닌 "변화량"으로 만든다.
                # 즉, 현재 position - 이전 position
                img_metas[i][0]['can_bus'][:3] -= self.train_prev_infos['prev_pos'][i]
                img_metas[i][0]['can_bus'][-1] -= self.train_prev_infos['prev_angle'][i]

            # ============================================================
            # 2) prev_infos 업데이트 (다음 프레임을 위해 현재 프레임을 저장)
            # ============================================================

            # scene_token 저장 (scene 경계 판별용)
            self.train_prev_infos['scene_token'][i] = img_metas[i][0]['scene_token']

            # 현재 timestamp 저장 (다음 step에서 time_delta 계산용)
            self.train_prev_infos['prev_timestamp'][i] = timestamp[i][0]

            # 현재 pose 저장 (다음 step에서 l2g_r1/t1로 사용됨)
            self.train_prev_infos['l2g_r_mat'][i] = l2g_r_mat[i][0]
            self.train_prev_infos['l2g_t'][i] = l2g_t[i][0]

            # "현재 can_bus 원본 절대값"을 저장 (다음 step에서 delta 계산 기준이 됨)
            self.train_prev_infos['prev_pos'][i] = tmp_pos
            self.train_prev_infos['prev_angle'][i] = tmp_angle

        # ============================================================
        # 3) prev_bev 구성
        # ============================================================
        """
        self.train_prev_infos['prev_bev']는 batch마다 prev_bev가 저장되어 있음
        - 첫 프레임에서는 None일 수 있으므로
        None이면 "0 BEV feature"로 채운다.

        최종 prev_bev shape:
        [B, bev_h*bev_w, C] 형태로 stacking됨
        (아래 torch.zeros도 그 shape에 맞춰 만듦)
        """
        prev_bev = torch.stack([
            bev if isinstance(bev, torch.Tensor)
            else torch.zeros(
                [self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w,
                self.pts_bbox_head.in_channels]
            ).to(img.device)
            for bev in self.train_prev_infos['prev_bev']
        ])

        # ============================================================
        # 4) 이전 track_instances 가져오기
        # ============================================================
        """
        train_det=True면 "detection만 학습"하거나 tracking memory를 사용하지 않는 모드라서
        track_instances를 전부 None으로 초기화한다.

        train_det=False면 streaming tracking 학습이므로
        이전 프레임에서 저장해둔 track_instances를 불러와서 이어서 사용한다.
        """
        if self.train_det:
            track_instances = [None for i in range(self.batch_size)]
        else:
            track_instances = self.train_prev_infos['track_instances']

        # ============================================================
        # 5) 현재 frame만 뽑아서 single-frame forward 수행
        # ============================================================
        """
        img는 [B, num_frame, num_cam, C, H, W] 구조일 가능성이 높음
        그래서 img_[0]으로 "현재 프레임(첫 프레임)"만 가져옴

        img_single shape:
        [B, num_cam, C, H, W]
        """
        img_single = torch.stack([img_[0] for img_ in img], dim=0)

        # img_metas도 마찬가지로 현재 frame meta만 뽑음
        # deepcopy를 쓰는 이유:
        #   - 앞에서 can_bus를 delta 형태로 바꾸는 등의 수정이 들어갔고
        #   - downstream에서 meta를 in-place로 만질 수 있으니 안전하게 복사
        img_metas_single = [copy.deepcopy(img_metas[i][0]) for i in range(self.batch_size)]

        # ============================================================
        # 6) 핵심: 한 프레임 학습 forward 수행
        # ============================================================
        """
        _forward_single_frame_train_bs()는 실제로
        - BEV 생성(get_bevs)
        - detection head(Transformer decoder)
        - loss 계산(loss_single_batch, history/future reasoning 등)
        - 다음 프레임용 track_instances 준비(frame_summarization + active_mask)
        을 수행하고

        반환:
        frame_res: dict
            - frame_res["track_instances"] : 다음 프레임으로 넘길 active track_instances (batch list)
            - frame_res["bev_embed"]       : 현재 프레임 BEV embedding
        losses: dict
        """
        frame_res, losses = self._forward_single_frame_train_bs(
            img_single,
            img_metas_single,
            track_instances,
            None,  # prev_img (stream 모드에서는 prev_img 직접 안 쓰고 prev_bev만 사용)
            None,  # prev_img_metas
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            veh2inf_rt,
            prev_bev=prev_bev,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            gt_forecasting_locs=gt_forecasting_locs,
            gt_forecasting_masks=gt_forecasting_masks,
            **kwargs,
        )

        # 다음 프레임용 track_instances (batch list 형태)
        track_instances = frame_res["track_instances"]

        # ============================================================
        # 7) train_prev_infos 업데이트 (다음 프레임으로 streaming memory 전달)
        # ============================================================
        """
        frame_res['bev_embed']는 현재 frame의 BEV embedding인데,
        다음 프레임을 위해 train_prev_infos['prev_bev']에 저장해야 한다.

        detach().clone() 하는 이유:
        - prev_bev는 "다음 프레임에서 입력으로 쓸 메모리"이지
            gradient를 계속 연결하면 BPTT처럼 그래프가 길게 이어져서 메모리 터짐
        - 따라서 여기서 gradient 연결을 끊는다(detach)
        """
        bev_embed = frame_res['bev_embed'].detach().clone()

        for i in range(self.batch_size):
            # bev_embed[:, i, :] : i번째 batch의 BEV embedding
            # shape: [bev_h*bev_w, C]
            self.train_prev_infos['prev_bev'][i] = bev_embed[:, i, :]

            # track_instances[i]도 다음 프레임 입력으로 쓰기 때문에
            # detach_and_clone()으로 그래프를 끊고 저장
            self.train_prev_infos['track_instances'][i] = track_instances[i].detach_and_clone()

        # 최종적으로 현재 프레임의 loss들을 반환
        return losses

    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None,
        sample_idx=None,
        **kwargs
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """
        prev_active_track_instances = track_instances
        if prev_active_track_instances is None:
            track_instances = self._generate_empty_tracks()
        else:
            prev_active_track_instances = self.update_reference_points(prev_active_track_instances,
                                                                       time_delta.type(torch.float),
                                                                       use_prediction=self.motion_prediction_ref_update,
                                                                       tracking=True)
            if self.if_update_ego:
                prev_active_track_instances = self.update_ego(prev_active_track_instances, 
                                                                l2g_r1[0], l2g_t1[0], l2g_r2[0], l2g_t2[0])
            prev_active_track_instances = self.STReasoner.sync_pos_embedding(prev_active_track_instances, self.query_embedding)
            
            empty_track_instances = self._generate_empty_tracks()
            full_length = len(empty_track_instances)
            active_length = len(prev_active_track_instances)
            if active_length > 0:
                random_index = torch.randperm(full_length)
                selected = random_index[:full_length-active_length]
                empty_track_instances = empty_track_instances[selected]
            out_track_instances = Instances.cat([empty_track_instances, prev_active_track_instances])
            track_instances = out_track_instances

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            query_feats=track_instances.query_feats[None, :, :],
            query_embeds=track_instances.query_embeds[None, :, :],
            ref_points=track_instances.ref_pts[None, :, :],
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"].clone()
        output_coords = det_output["all_bbox_preds"].clone()
        last_ref_pts = det_output["last_ref_points"].clone()
        query_feats = det_output["query_feats"].clone()

        out = {
            "all_cls_scores": output_classes[:, 0, :, :],
            "all_bbox_preds": output_coords[:, 0, :, :],
            "ref_pts": last_ref_pts[0, :, :],
            "query_feats": query_feats[:, 0, :, :],
        }

        track_instances = self.load_detection_output_into_cache(track_instances, out)
        out['track_instances'] = track_instances

        # extract motion features
        if self.is_motion:
            track_instances = self.MotionExtractor(track_instances, img_metas)
        
        inf_instances = None
        # import pdb;pdb.set_trace()
        if self.is_cooperation:
            inf_dcit = {
                'query_feats': kwargs['query_feats'][0][0],
                'query_embeds': kwargs['query_embeds'][0][0],
                'cache_motion_feats': kwargs['cache_motion_feats'][0][0],
                'ref_pts': kwargs['ref_pts'][0][0],
                'pred_boxes': kwargs['pred_boxes'][0][0],
            }
            if inf_dcit['query_feats'].shape[0] > 0:
                inf_dcit = self.crossview_alignment(inf_dcit, veh2inf_rt[0])
                inf_instances = self._init_inf_tracks(inf_dcit)
        # Spatial-temporal Reasoning
        track_instances, _ = self.STReasoner(track_instances, inf_instances, sample_idx)
        track_instances = self.frame_summarization(track_instances, tracking=True)
        out['all_cls_scores'][-1] = track_instances.pred_logits
        out['all_bbox_preds'][-1] = track_instances.pred_boxes

        if self.STReasoner.future_reasoning:
            # motion forecasting has the shape of [num_query, T, 2]
            out['all_motion_forecasting'] = track_instances.motion_predictions.clone()
        else:
            out['all_motion_forecasting'] = None

        # assign ids
        active_mask = (track_instances.scores > self.runtime_tracker.threshold)
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] < 0:
                track_instances.obj_idxes[i] = self.runtime_tracker.current_id 
                self.runtime_tracker.current_id += 1
                if active_mask[i]:
                    track_instances.track_query_mask[i] = True
        out['track_instances'] = track_instances
        # output track results
        active_index = (track_instances.scores >= self.runtime_tracker.output_threshold)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))

        next_instances = self.runtime_tracker.update_active_tracks(track_instances, active_mask)
        out["track_instances"] = next_instances
        out.update(self._det_instances2results(out, img_metas))
        out["track_obj_idxes"] = track_instances.obj_idxes
        out["bev_embed"] = bev_embed
        return out
    
    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        veh2inf_rt=None,
        **kwargs,
    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
            or timestamp - self.timestamp > 0.5
            or timestamp < self.timestamp
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = None
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            img_metas[0]['can_bus'][:3] = 0
            img_metas[0]['can_bus'][-1] = 0
            self.runtime_tracker.empty()
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
            img_metas[0]['can_bus'][:3] -= self.prev_pos
            img_metas[0]['can_bus'][-1] -= self.prev_angle
            # print(time_delta, img_metas[0]['can_bus'][:3], img_metas[0]['can_bus'][-1])
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat
        self.prev_pos = tmp_pos
        self.prev_angle = tmp_angle

        """ predict and update """
        prev_bev = self.prev_bev
        if self.train_det:
            track_instances = None
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            veh2inf_rt,
            img_metas[0]['sample_idx'],
            **kwargs
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]

        self.test_track_instances = track_instances
                
        results = [dict()]
        get_keys = ["track_bbox_results", "boxes_3d_det", "scores_3d_det", "labels_3d_det",
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        results[0].update({k: frame_res[k] for k in get_keys})

        if self.save_track_query:
            tensor_to_cpu = torch.zeros(1)
            save_path = os.path.join(self.save_track_query_file_root, img_metas[0]['sample_idx'] +'.pkl')
            track_instances = track_instances.to(tensor_to_cpu)
            mmcv.dump(track_instances, save_path)

        return results
    
    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        bboxes_dict = self.pts_bbox_head.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, pred_dict, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # import pdb;pdb.set_trace()
        cls_score = pred_dict['all_cls_scores'].clone()
        bbox_preds = pred_dict['all_bbox_preds'].clone()
        scores = cls_score[-1].sigmoid().max(dim=-1).values
        obj_idxes = torch.ones_like(scores)
        bbox_dict = dict(
            cls_scores=cls_score[-1],
            bbox_preds=bbox_preds[-1],
            track_scores=scores,
            obj_idxes=obj_idxes,
        )
        bboxes_dict = self.pts_bbox_head.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )

        return result_dict_det

