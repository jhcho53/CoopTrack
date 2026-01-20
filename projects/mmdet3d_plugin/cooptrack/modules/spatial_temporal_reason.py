# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
""" Spatial-temporal Reasoning Module
"""
import os
import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ..dense_heads.track_head_plugin import Instances
from mmcv.cnn import Conv2d, Linear
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import build_transformer
from .pf_utils import time_position_embedding, xyz_ego_transformation, normalize, denormalize
from scipy.optimize import linear_sum_assignment
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox

color_mapping = [
    np.array([1.0, 0.078, 0.576]), # 鲜艳的粉色
    np.array([1.0, 1.0, 0.0]),   # 鲜艳的黄色
    np.array([1.0, 0.647, 0.0]), # 鲜艳的橙色
    np.array([0.502, 0.0, 0.502]), # 鲜艳的紫色
    np.array([0.0, 1.0, 1.0]),   # 鲜艳的青色
    np.array([1.0, 0.0, 1.0]),   # 鲜艳的洋红色
    np.array([0.0, 1.0, 0.502]), # 鲜艳的青绿色
    np.array([1.0, 0.843, 0.0]),  # 鲜艳的金色
    np.array([0.0, 0.8, 0.0]),
    np.array([0.4, 0.0, 0.8]),   # 鲜艳的蓝色
]

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

class SpatialTemporalReasoner(nn.Module):
    def __init__(self, 
                 history_reasoning=True,
                 future_reasoning=True,
                 embed_dims=256, 
                 hist_len=3, 
                 fut_len=4,
                 num_reg_fcs=2,
                 code_size=10,
                 num_classes=10,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 hist_temporal_transformer=None,
                 fut_temporal_transformer=None,
                 spatial_transformer=None,
                 is_motion=False,
                 is_cooperation=False,
                 learn_match=False,
                 veh_thre=0.05):
        super(SpatialTemporalReasoner, self).__init__()

        self.embed_dims = embed_dims
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.num_reg_fcs = num_reg_fcs
        self.pc_range = pc_range

        self.num_classes = num_classes
        self.code_size = code_size

        self.history_reasoning = history_reasoning
        self.future_reasoning = future_reasoning
        self.is_motion = is_motion
        self.is_cooperation = is_cooperation
        self.learn_match = learn_match
        self.veh_thre = veh_thre
        self.hist_temporal_transformer = hist_temporal_transformer
        self.fut_temporal_transformer = fut_temporal_transformer
        self.spatial_transformer = spatial_transformer

        self.init_params_and_layers()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, track_instances, inf_track_instances=None, sample_idx=None):
        # 1. Prepare the spatial-temporal features
        track_instances = self.frame_shift(track_instances) # 가장 오래된 프레임 버퍼를 날리고 현재 프레임 데이터 삽입

        affinity = None
        # 2. History reasoning
        if self.history_reasoning:
            track_instances = self.forward_history_reasoning(track_instances)
            if inf_track_instances or self.learn_match:
                track_instances, affinity = self.aggregation(track_instances, inf_track_instances, sample_idx)
            track_instances = self.forward_history_refine(track_instances)

        # 3. Future reasoning
        if self.future_reasoning:
            track_instances = self.forward_future_reasoning(track_instances)
            track_instances = self.forward_future_prediction(track_instances)
        return track_instances, affinity

    def forward_history_reasoning(self, track_instances: Instances):
        """
        [History Reasoning]
        - 과거(history buffer)에 저장된 정보(hist_embeds 등)를 이용해서
        "현재 프레임의 query feature(cache_query_feats)"를 더 정교하게 보정(refine)하는 단계

        큰 흐름:
        1) history가 존재하는(valid) track query만 골라냄
        2) Temporal Transformer로 "시간축(t-2,t-1,t) 정보"를 반영해서 현재 query feature를 개선
        3) Spatial Transformer로 "현재 프레임의 여러 query 간 관계(공간적인 상호작용)"까지 보정
        4) (옵션) motion feature(cache_motion_feats)도 temporal transformer로 동일하게 업데이트
        5) 업데이트된 결과를 cache_query_feats / hist_embeds 마지막 슬롯에 저장

        track_instances 안 주요 필드:
        - cache_query_feats: [N, C]      # 현재 프레임 detection head가 만든 query feature
        - hist_embeds:       [N, T, C]   # 과거 T개 프레임의 query feature history
        - hist_padding_masks:[N, T]      # history slot이 비어있으면 1(True), 있으면 0(False)
        - cache_query_embeds:[N, C]      # 현재 프레임 query positional embedding (ref_pts로부터)
        """

        # ----------------------------------------
        # (0) query가 하나도 없으면 바로 종료
        # ----------------------------------------
        if len(track_instances) == 0:
            return track_instances

        # ----------------------------------------
        # (1) "history가 존재하는 query"만 선택
        # ----------------------------------------
        # hist_padding_masks: [N, hist_len]
        #   - 1(True)  => padding(값 없음)
        #   - 0(False) => valid(값 있음)
        #
        # [:, -1]은 history buffer의 가장 "마지막 슬롯"(즉 가장 최근 timestep)을 의미
        # -> 최근 history가 들어있는 query만 reasoning 수행하겠다는 의미
        valid_idxes = (track_instances.hist_padding_masks[:, -1] == 0)

        # 현재 프레임 query feature (temporal reasoning에서 "query/target" 역할)
        # cache_query_feats: [N, C] -> valid만 추리면 [Nv, C]
        embed = track_instances.cache_query_feats[valid_idxes]

        # valid query가 없으면 종료
        if len(embed) == 0:
            return track_instances

        # ----------------------------------------
        # (2) Temporal Transformer용 memory 준비
        # ----------------------------------------
        # hist_embeds: [N, T, C] -> [Nv, T, C]
        # 이게 temporal transformer에서 key/value가 되는 "과거 sequence"
        hist_embed = track_instances.hist_embeds[valid_idxes]

        # hist_padding_masks: [N, T] -> [Nv, T]
        # temporal attention에서 padding된 time slot을 무시하기 위해 사용
        hist_padding_mask = track_instances.hist_padding_masks[valid_idxes]

        # ----------------------------------------
        # (3) Time positional embedding 생성
        # ----------------------------------------
        """
        시간축 정보(t-2, t-1, t ...)가 없으면
        transformer 입장에서는 "sequence 순서"를 구분할 수 없음

        그래서 hist_len 길이의 time positional embedding(ts_pe)을 만들어줌

        ts_pe shape 예시:
        - time_position_embedding(batch=Nv, T=self.hist_len, C=embed_dims)
        - 결과: [Nv, T, C]
        """
        ts_pe = time_position_embedding(
            hist_embed.shape[0],     # Nv
            self.hist_len,           # history 길이
            self.embed_dims,         # feature dim (C=256)
            hist_embed.device
        )

        # ts_pe를 latent space로 한 번 더 projection (학습 가능한 embedding으로 만들기)
        # ts_query_embed: [Nv, T, C] -> [Nv, T, C]
        ts_pe = self.ts_query_embed(ts_pe)

        # ----------------------------------------
        # (4) Temporal Transformer 수행 (시간축 reasoning)
        # ----------------------------------------
        """
        hist_temporal_transformer 입력 의미:
        - target: 현재 query feature (우리가 refine 하고 싶은 대상)
        - x: 과거 history sequence (memory / key-value)
        - query_embed: query의 positional embedding (현재 timestep용)
        - pos_embed: memory의 positional embedding (전체 timestep용)
        - padding mask로 "없는 history slot"은 attention에서 제외

        shape 정리:
        embed: [Nv, C]
        embed[:, None, :]: [Nv, 1, C]       # 현재 query를 길이=1짜리 sequence처럼 취급

        hist_embed: [Nv, T, C]              # memory
        ts_pe: [Nv, T, C]                   # time positional embedding
        ts_pe[:, -1:, :]: [Nv, 1, C]        # query가 참조할 time embedding(가장 최신 slot)
        """
        temporal_embed = self.hist_temporal_transformer(
            target=embed[:, None, :],            # [Nv, 1, C] (현재 query)
            x=hist_embed,                        # [Nv, T, C] (과거 memory)
            query_embed=ts_pe[:, -1:, :],        # [Nv, 1, C] (현재 time embedding)
            pos_embed=ts_pe,                     # [Nv, T, C] (history time embedding)
            query_key_padding_mask=hist_padding_mask[:, -1:],  # [Nv, 1]
            key_padding_mask=hist_padding_mask                 # [Nv, T]
        )
        # temporal_embed: [Nv, 1, C]
        # => 현재 query가 history를 보고 보정된 feature

        # ----------------------------------------
        # (5) Spatial Transformer용 positional embedding 준비
        # ----------------------------------------
        # cache_query_embeds는 ref_pts 기반 positional embedding
        # spatial transformer에서는 "현재 query가 어디에 위치한 객체인지"가 매우 중요하므로 사용
        # shape: [Nv, 1, C]
        hist_pe = track_instances.cache_query_embeds[valid_idxes, None, :]

        # ----------------------------------------
        # (6) Spatial Transformer 수행 (현재 프레임 query들 사이 관계 reasoning)
        # ----------------------------------------
        """
        여기서 하고 싶은 것:
        - temporal_embed로 시간정보까지 반영된 query들을,
            "현재 프레임 내부에서 서로 self-attention"시키며 한 번 더 정제한다.

        중요한 포인트:
        - temporal transformer에서는 시간축(K/V)이 hist_embed였고,
        - spatial transformer에서는 같은 프레임의 query들끼리 관계를 보므로
            temporal_embed 자체를 query/key/value로 넣고 self-attn을 한다.

        transpose를 왜 하냐?
        - mmcv transformer 구현체는 보통 입력을 [num_query, batch, C] 형태로 받는다.
        - 지금 temporal_embed는 [Nv, 1, C] (batch-first 처럼 보이는 형태)라
            이를 transformer가 기대하는 shape로 바꿔주는 과정이다.

        temporal_embed.transpose(0,1):
        - [Nv, 1, C] -> [1, Nv, C]
        - 여기서 1이 "batch or sequence length" 역할이고
            Nv가 query 차원처럼 들어가게 된다.
        - 즉, 현재 프레임의 Nv개 query를 attention 대상으로 처리하게 됨
        """
        spatial_embed = self.spatial_transformer(
            target=temporal_embed.transpose(0, 1),  # self-attn을 위해 target = temporal_embed
            x=temporal_embed.transpose(0, 1),       # key/value도 동일하게 temporal_embed
            query_embed=hist_pe.transpose(0, 1),    # [1, Nv, C] query positional embedding
            pos_embed=hist_pe.transpose(0, 1),      # [1, Nv, C] key/value positional embedding
            query_key_padding_mask=hist_padding_mask[:, -1:].transpose(0, 1),  # [1, Nv]
            key_padding_mask=hist_padding_mask[:, -1:].transpose(0, 1)         # [1, Nv]
        )[0]
        # spatial_embed: [1, Nv, C] 형태로 나오고,
        # [0]을 통해 최종 feature tensor를 꺼내온다.
        #
        # 최종적으로 spatial_embed는 "현재 프레임에서 query들끼리 상호작용 반영된 feature"

        # ----------------------------------------
        # (7) Motion feature도 history reasoning 적용 (옵션)
        # ----------------------------------------
        if self.is_motion:
            """
            motion feature(cache_motion_feats)는
            - bbox geometry + 속도(vx,vy) 등에서 만든 motion embedding
            - temporal consistency를 크게 필요로 함
            (프레임 간 움직임은 시간축 reasoning이 특히 효과적)

            따라서 hist_motion_embeds에 대해 temporal transformer를 동일하게 적용한다.
            """

            # 현재 프레임 motion feature: [Nv, C]
            mot_embed = track_instances.cache_motion_feats[valid_idxes]

            # 과거 motion history: [Nv, T, C]
            hist_mot_embed = track_instances.hist_motion_embeds[valid_idxes]

            # padding mask: [Nv, T]
            hist_padding_mask = track_instances.hist_padding_masks[valid_idxes]

            # motion용 time positional embedding 생성: [Nv, T, C]
            ts_pe = time_position_embedding(
                hist_mot_embed.shape[0],
                self.hist_len,
                self.embed_dims,
                hist_mot_embed.device
            )
            ts_pe = self.ts_query_embed(ts_pe)

            # motion temporal transformer 수행
            mot_embed = self.hist_motion_transformer(
                target=mot_embed[:, None, :],        # [Nv, 1, C]
                x=hist_mot_embed,                    # [Nv, T, C]
                query_embed=ts_pe[:, -1:, :],        # [Nv, 1, C]
                pos_embed=ts_pe,                     # [Nv, T, C]
                query_key_padding_mask=hist_padding_mask[:, -1:],  # [Nv, 1]
                key_padding_mask=hist_padding_mask               # [Nv, T]
            )[:, 0, :]
            # 결과 shape: [Nv, C]
            # ([:, 0, :] => 길이 1인 sequence 차원을 제거)

            # 현재 프레임에서 바로 사용하는 값이라 cache에는 detach 없이 업데이트
            track_instances.cache_motion_feats[valid_idxes] = mot_embed.clone()

            # history buffer에는 "학습 그래프가 계속 이어지는 것"을 방지하기 위해 detach 저장
            # (메모리뱅크는 보통 gradient를 막고 저장하는게 안정적)
            track_instances.hist_motion_embeds[valid_idxes, -1] = mot_embed.clone().detach()

        # ----------------------------------------
        # (8) 최종 spatial_embed를 cache/query history에 저장
        # ----------------------------------------
        # cache_query_feats는 "현재 프레임에서 reasoning 이후 결과"를 저장하는 슬롯
        # spatial_embed는 현재 [1, Nv, C]이므로 squeeze해서 [Nv, C]로 맞춰야 하는 경우도 있음
        # (여기 구현에서는 shape가 맞다고 가정하고 바로 넣고 있음)
        track_instances.cache_query_feats[valid_idxes] = spatial_embed.clone()

        # history buffer 마지막 슬롯(-1)은 "현재 프레임 feature"를 저장하는 slot
        # detach해서 저장해서, 다음 frame에서 history로 사용될 때 gradient가 연결되지 않게 함
        track_instances.hist_embeds[valid_idxes, -1] = spatial_embed.clone().detach()

        return track_instances

    
    def forward_history_refine(self, track_instances: Instances):
        """
        [History Refine]
        - forward_history_reasoning()을 통해 업데이트된 feature(cache_query_feats / cache_motion_feats)를 이용해
        "현재 프레임의 classification(logits)"과 "bbox regression(bboxes)"를 한 번 더 업데이트하는 단계

        즉,
        ✅ history reasoning -> feature refine
        ✅ feature refine -> prediction(logits/bbox) refine

        결과는 모두 track_instances.cache_*에 저장된다.
        - cache_logits : refined classification logits
        - cache_scores : refined score (objectness처럼 사용)
        - cache_ref_pts: refined reference point (다음 단계의 refine/association에 사용)
        - cache_bboxes : refined bbox (world 좌표로 저장)
        """

        # ----------------------------------------
        # (0) query가 없으면 종료
        # ----------------------------------------
        if len(track_instances) == 0:
            return track_instances

        # ----------------------------------------
        # (1) refine 가능한(valid history가 있는) query만 선택
        # ----------------------------------------
        valid_idxes = (track_instances.hist_padding_masks[:, -1] == 0)

        # reasoning 이후의 query feature
        # cache_query_feats: [N, C] -> [Nv, C]
        embed = track_instances.cache_query_feats[valid_idxes]

        # valid query가 없으면 종료
        if len(embed) == 0:
            return track_instances

        # ============================================================
        # (2) Classification Head를 이용해 logits 업데이트
        # ============================================================
        """
        track_cls는 refined query feature를 기반으로 class logits을 재예측한다.
        logits shape: [Nv, num_classes]
        """
        logits = self.track_cls(track_instances.cache_query_feats[valid_idxes])

        # refined logits을 cache_logits에 저장
        # cache_logits: [N, num_classes]
        track_instances.cache_logits[valid_idxes] = logits.clone()

        # cache_scores는 "objectness score"처럼 사용되는 값
        # DETR류에서는 보통:
        #   sigmoid(logits) -> class probability
        #   그 중 max class 확률을 score로 사용 (가장 confident한 class)
        #
        # shape:
        #   logits.sigmoid(): [Nv, num_classes]
        #   max(dim=-1).values: [Nv]
        track_instances.cache_scores = logits.sigmoid().max(dim=-1).values

        # ============================================================
        # (3) Localization(BBox Regression) 업데이트
        # ============================================================
        """
        track_reg는 bbox delta를 예측하는 head.
        여기서 특이한 점:
        - is_motion=True면 motion feature(cache_motion_feats)를 통해 bbox를 reg
            (움직임/속도를 반영해서 박스를 더 잘 보정하려는 의도)
        - 아니면 일반 query feature(cache_query_feats)로 bbox reg
        """
        if self.is_motion:
            # deltas: [Nv, box_dim]
            deltas = self.track_reg(track_instances.cache_motion_feats[valid_idxes])
        else:
            deltas = self.track_reg(track_instances.cache_query_feats[valid_idxes])

        # ============================================================
        # (4) DETR box refine 트릭 적용 (reference point 기반 residual update)
        # ============================================================
        """
        DETR류 box refine 방식:
        - bbox를 직접 absolute로 예측하지 않고
        - "reference point를 기준으로 한 residual(delta)"를 예측
        - 이후 sigmoid/inverse_sigmoid로 안정적인 업데이트 수행

        cache_ref_pts는 normalized 형태(0~1)의 reference point로 저장되어 있음
        하지만 refinement는 "logit 공간"에서 더하는 게 안정적이라 inverse_sigmoid로 변환해서 더한다.
        """

        # cache_ref_pts: [N, 3] (x,y,z 혹은 x,y,cz 형태)
        # valid만 선택: [Nv, 3]
        reference = inverse_sigmoid(track_instances.cache_ref_pts[valid_idxes].clone())

        # deltas는 box_dim 차원 예측값인데,
        # 여기서는 center(x,y,z=4)만 reference 기반으로 residual 업데이트함
        #
        # index 설명(너가 다른 코드에서 사용한 형식 기반):
        #   deltas[..., 0] = center x (normalized)
        #   deltas[..., 1] = center y (normalized)
        #   deltas[..., 4] = center z(or cz) (normalized에서 사용되는 z축 index)
        #
        # 즉 [0,1,4]만 reference 기반 refine하고,
        # 나머지(w,l,h,rot,vx,vy 등)는 reg head가 직접 예측한 값을 사용
        deltas[..., [0, 1, 4]] += reference              # reference를 기준으로 residual 적용
        deltas[..., [0, 1, 4]] = deltas[..., [0, 1, 4]].sigmoid()  # 다시 0~1로

        # ============================================================
        # (5) refined reference point를 cache_ref_pts에 저장
        # ============================================================
        # cache_ref_pts는 이후:
        #   - 다음 reasoning 단계
        #   - 다음 frame tracking(update_reference_points)
        #   - association 등
        # 에서 중요한 역할을 한다.
        track_instances.cache_ref_pts[valid_idxes] = deltas[..., [0, 1, 4]].clone()

        # ============================================================
        # (6) cache_bboxes에 저장하기 전에 실제 좌표로 변환(denormalize)
        # ============================================================
        """
        deltas[..., [0,1,4]]는 normalized(0~1)이므로,
        cache_bboxes는 실제 meter 단위 좌표로 저장하려고 denormalize를 수행한다.

        주의:
        - cache_ref_pts는 normalized로 유지
        - cache_bboxes는 absolute(world) 좌표로 저장
        """
        deltas[..., [0, 1, 4]] = denormalize(deltas[..., [0, 1, 4]], self.pc_range)

        # refined bbox를 cache_bboxes에 저장
        # cache_bboxes: [N, box_dim]
        track_instances.cache_bboxes[valid_idxes, :] = deltas

        return track_instances


    def forward_future_reasoning(self, track_instances: Instances):
        hist_embeds = track_instances.hist_embeds
        hist_padding_masks = track_instances.hist_padding_masks
        ts_pe = time_position_embedding(hist_embeds.shape[0], self.hist_len + self.fut_len, 
                                        self.embed_dims, hist_embeds.device)
        ts_pe = self.ts_query_embed(ts_pe)
        fut_embeds = self.fut_temporal_transformer(
            target=torch.zeros_like(ts_pe[:, self.hist_len:, :]),
            x=hist_embeds,
            query_embed=ts_pe[:, self.hist_len:],
            pos_embed=ts_pe[:, :self.hist_len],
            key_padding_mask=hist_padding_masks)
        track_instances.fut_embeds = fut_embeds
        return track_instances
    
    def forward_future_prediction(self, track_instances):
        """Predict the future motions"""
        motion_predictions = self.future_reg(track_instances.fut_embeds)
        track_instances.cache_motion_predictions = motion_predictions
        return track_instances

    def aggregation(self, veh_instances, inf_instances, sample_idx):
        """
        [Aggregation 함수]
        - vehicle agent(차량)와 infra agent(인프라)의 track/query 정보를 "하나의 query set"으로 통합하는 단계
        - 통합 대상은 현재 frame의 cache 정보들:
            - cache_query_feats  : query feature (appearance/semantic feature)
            - cache_motion_feats : motion feature (geometry + velocity embedding)
            - cache_ref_pts      : reference point (normalized 좌표, 0~1)

        전체 흐름:
        1) Association (veh query ↔ inf query 매칭)
            - rule-based matching 또는 learnable matching으로 어떤 veh query가 어떤 inf query랑 같은 객체인지 결정
        2) Aggregation / Fusion
            - 매칭된 veh/inf query는 feature를 합쳐서 fused query로 만들고
            - 매칭되지 않은 infra query는 보완(complementation)으로 추가하거나
            일부 veh query도 유지해서 최종 query set 구성
        3) Cross-domain projection
            - 인프라/차량 도메인 차이를 줄이기 위해 positional embedding(cache_query_embeds)을 concat해서
            feature를 다시 한번 projection(MLP)하여 domain gap 완화

        반환:
        - res_instances : vehicle+infra 통합된 최종 query set (Instances)
        - affinity      : learn_match인 경우 attention 기반 유사도(association score) (학습 loss에 사용)
                            rule-based면 None
        """

        # ============================================================
        # 0) Association을 위해 reference point 가져오기
        # ============================================================

        # vehicle query의 reference points (normalized 좌표)
        # shape 예: [Nv_total, 3]
        #    [:,0] = x (0~1 normalized)
        #    [:,1] = y (0~1 normalized)
        #    [:,2] = z or cz (0~1 normalized)
        veh_ref_pts = veh_instances.cache_ref_pts.clone()

        # infra query의 reference points (normalized 좌표)
        # shape 예: [Ni_total, 3]
        inf_ref_pts = inf_instances.cache_ref_pts.clone()

        # ============================================================
        # 1) Matching은 거리 기반이므로 normalized -> absolute(world) 변환
        # ============================================================

        # normalized 좌표(0~1)를 pc_range 기준으로 실제 월드 좌표(m 단위)로 복원
        # veh_abs_pts shape: [Nv_total, 3]
        veh_abs_pts = self._loc_denorm(veh_ref_pts, self.pc_range)

        # inf_abs_pts shape: [Ni_total, 3]
        inf_abs_pts = self._loc_denorm(inf_ref_pts, self.pc_range)

        # ============================================================
        # 2) 차량 query 중 confidence가 낮은 것 제거하기 위한 mask 생성
        # ============================================================

        # cache_scores는 query마다 objectness처럼 쓰는 confidence score
        # shape: [Nv_total]
        # veh_thre보다 큰 query만 "실제 객체 후보"로 match 대상에 포함
        mask = veh_instances.cache_scores > self.veh_thre
        # mask shape: [Nv_total] (torch.bool)

        # ============================================================
        # 3) Association(veh ↔ inf query matching)
        # ============================================================

        affinity = None  # learnable matching이면 affinity(attn score)를 반환해서 loss 계산에 활용

        if not self.learn_match:
            # ------------------------------------------------------------
            # 3-1) Rule-based matching
            # ------------------------------------------------------------
            """
            _query_matching()은 일반적으로 "거리 기반" 매칭을 수행
            - 입력: inf_abs_pts, veh_abs_pts, mask
            - 출력: veh_idx, inf_idx, (optional) cost

            여기서 veh_idx / inf_idx는 "서로 매칭된 query index 쌍"을 의미함
            예)
                veh_idx = [3, 10, 25]
                inf_idx = [0,  4,  7]
            => (veh 3 ↔ inf 0), (veh 10 ↔ inf 4), (veh 25 ↔ inf 7)
            """
            veh_idx, inf_idx, _ = self._query_matching(inf_abs_pts, veh_abs_pts, mask)

        else:
            # ------------------------------------------------------------
            # 3-2) Learnable matching (graph-based association)
            # ------------------------------------------------------------
            """
            Learnable matching의 목적:
            - 단순 거리 기준만으로는 어려운 상황(가려짐, 오차, 시야차이 등)에서
                feature 기반으로 "같은 객체일 확률"을 더 잘 학습하도록 함

            내부 아이디어(너가 위에 써둔 설명 그대로 확장):
            - veh / inf query feature + motion feature를 node feature로 보고
            - 거리 차이를 edge feature로 보고
            - q, k, e 기반 attention score(attn) 생성
            - sigmoid → affinity(0~1 유사도)
            - Hungarian assignment로 최종 matching idx를 뽑을 수도 있음
            """

            veh_idx, inf_idx, affinity = self._learn_matching(
                inf_instances, veh_instances,
                inf_abs_pts, veh_abs_pts,
                mask
            )

            # ------------------------------------------------------------
            # 3-3) 학습 시 안정적인 idx를 위해 rule-based 결과를 사용
            # ------------------------------------------------------------
            """
            매우 중요한 구현 트릭:

            learnable matching은 학습 초기에 불안정할 수 있음
            - affinity score는 나오지만 Hungarian assignment 결과가 흔들릴 수 있음
            - 또는 학습 중에는 _learn_matching이 idx를 반환하지 않는 구현일 수도 있음

            그래서 학습 중(self.training=True)에는
            - "matching idx는 안정적인 rule-based 결과를 사용"
            - 대신 affinity는 loss 계산에 유지(asso_loss 학습용)

            즉,
            ✅ forward의 동작 안정성 = rule-based idx
            ✅ association 학습 supervision = affinity loss
            """
            if self.training:
                veh_idx, inf_idx, _ = self._query_matching(inf_abs_pts, veh_abs_pts, mask)

        # ============================================================
        # 4) Aggregation (Query Fusion + Complementation)
        # ============================================================

        # ------------------------------------------------------------
        # 4-1) Query Fusion
        # ------------------------------------------------------------
        """
        _query_fusion() 역할:
        - 매칭된 veh_idx ↔ inf_idx 쌍에 대해
            vehicle query와 infra query의 feature를 결합해서 "fused query" 생성

        흔한 구현 방식:
        - concat([veh_feat, inf_feat]) -> MLP -> fused_feat
        - motion_feat도 concat해서 fused_motion_feat 생성할 수도 있음

        반환 fused_instances:
        - 매칭된 query들에 대한 fused 결과를 담는 Instances
        """
        fused_instances = self._query_fusion(
            inf_instances, veh_instances,
            inf_idx, veh_idx
        )

        # ------------------------------------------------------------
        # 4-2) Query Complementation
        # ------------------------------------------------------------
        """
        _query_complementation() 역할:
        - 매칭되지 않은 query들을 처리해서 최종 query set 구성

        일반적인 정책:
        1) 매칭된 query → fused query로 포함
        2) 매칭되지 않은 infra query
            - vehicle이 못 본 객체를 infra가 봤을 수 있으므로 추가(complement)
        3) 매칭되지 않은 vehicle query도 일부 유지
            - infra가 못 본 객체를 vehicle이 봤을 수 있으니 유지

        최종 반환 res_instances:
        - fused + complemented query들의 집합
        - 즉, "최종 통합 query set"
        """
        res_instances = self._query_complementation(
            inf_instances, veh_instances,
            inf_idx, veh_idx,
            fused_instances
        )

        # ============================================================
        # 5) Cross-domain projection (Domain gap 보정)
        # ============================================================

        """
        왜 이 단계가 필요하냐?
        - vehicle query feature와 infra query feature는 도메인이 다르다.
            (센서 위치, 시야, 특징 분포, 노이즈 패턴이 다름)
        - 단순 concat/fusion만으로는 feature space가 잘 정렬되지 않을 수 있음

        해결:
        - positional embedding(cache_query_embeds)을 함께 concat해서
            feature가 "어디에 있는 객체인지" 정보를 명확히 주고,
        - cross_domain MLP로 다시 projection해서 도메인 갭을 줄인다.

        입력 shape 예시:
        cache_query_feats  : [N_final, C]
        cache_query_embeds : [N_final, C]
        concat -> [N_final, 2C]
        cross_domain_query -> [N_final, C]
        """

        # query feature + position embedding 결합 후 projection
        res_instances.cache_query_feats = self.cross_domain_query(
            torch.cat([res_instances.cache_query_feats, res_instances.cache_query_embeds], dim=-1)
        )

        # motion feature + position embedding 결합 후 projection
        res_instances.cache_motion_feats = self.cross_domain_motion(
            torch.cat([res_instances.cache_motion_feats, res_instances.cache_query_embeds], dim=-1)
        )

        # ============================================================
        # 6) 최종 통합 query set + affinity 반환
        # ============================================================

        # res_instances: 최종 통합된 query set (이후 detection refine / tracking에 사용)
        # affinity: learnable matching score (association loss에 사용), rule-based면 None
        return res_instances, affinity


    def _query_fusion(self, inf, veh, inf_idx, veh_idx):
        """
        Query fusion: 
            replacement for scores, ref_pts and pos_embed according to confidence_score
            fusion for features via MLP
        
        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_idx: matched idxs for inf side
        veh_idx: matched idxs for veh side
        cost_matrix
        """
        matched_veh = veh[veh_idx]
        matched_inf = inf[inf_idx]
        matched_veh.cache_query_feats = self.fuse_feats(torch.cat([matched_veh.cache_query_feats, matched_inf.cache_query_feats], dim=-1))
        matched_veh.cache_motion_feats = self.fuse_motion(torch.cat([matched_veh.cache_motion_feats, matched_inf.cache_motion_feats], dim=-1))
        matched_veh.cache_query_embeds = self.fuse_embed(torch.cat([matched_veh.cache_query_embeds, matched_inf.cache_query_embeds], dim=-1))
        matched_veh.cache_ref_pts = (matched_veh.cache_ref_pts + matched_inf.cache_ref_pts) / 2.0
        return matched_veh

    def _query_complementation(self, inf, veh, inf_accept_idx, veh_accept_idx, fused):
        """
        Query complementation: replace low-confidence vehicle-side query with unmatched inf-side query

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_accept_idx: idxs of matched instances
        """
        veh_num = len(veh)
        inf_num = len(inf)

        mask = torch.ones(veh_num, dtype=bool)
        mask[veh_accept_idx] = False
        unmatched_veh = veh[mask]

        mask = torch.ones(inf_num, dtype=bool)
        mask[inf_accept_idx] = False
        unmatched_inf = inf[mask]
        res_instances = Instances((1, 1))
        res_instances = Instances.cat([res_instances, fused])
        res_instances = Instances.cat([res_instances, unmatched_inf])

        select_num = veh_num - inf_num
        _, topk_indexes = torch.topk(unmatched_veh.cache_scores, select_num, dim=0)
        res_instances = Instances.cat([res_instances, unmatched_veh[topk_indexes]])

        return res_instances
    
    def _loc_denorm(self, ref_pts, pc_range):
        """
        normalized (x,y,z) ---> absolute (x,y,z) in global coordinate system
        """
        locs = ref_pts.clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs
    
    def _learn_matching(self, inf, veh, inf_pts, veh_pts, mask):
        # filter veh
        if self.training:
            indices = torch.arange(len(veh))
            filter_indices = indices[mask]
            veh = veh[mask]

        inf_query = inf.cache_query_feats.clone()
        inf_motion = inf.cache_motion_feats.clone()
        inf_ref_pts = inf.cache_ref_pts.clone()
        veh_query = veh.cache_query_feats.clone()
        veh_motion = veh.cache_motion_feats.clone()
        veh_ref_pts = veh.cache_ref_pts.clone()
        # 1. construct graph
        # 1.1 prepare nodes
        inf_nodes = self.get_node_inf(torch.cat([inf_query, inf_motion], dim=-1))
        veh_nodes = self.get_node_veh(torch.cat([veh_query, veh_motion], dim=-1))

        # 1.2 prepare edges
        dis = veh_ref_pts.unsqueeze(1) - inf_ref_pts
        edges = self.get_edge(dis)

        # 2. generate affine matrix
        q = torch.matmul(veh_nodes, self.wq)
        k = torch.matmul(inf_nodes, self.wk)
        e = torch.matmul(edges, self.we)

        attn = torch.matmul(q, k.transpose(0, 1)).unsqueeze(-1)
        attn = attn + e
        attn = self.ffn(attn).squeeze(-1)

        if self.training:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), attn
        # 3. output associated idxes
        affinity = self.sigmoid(attn)
        if affinity.shape[0] == 0 or affinity.shape[1] == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), attn
        
        # affinity: close to 1 means the objects are same
        norm_diff_aff = 1 - affinity.detach().cpu()
        diff = norm_diff_aff

        cost = diff.numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        cost_mask = cost[row_ind, col_ind] < 1e5
        veh_accept_idx = row_ind[cost_mask]
        inf_accept_idx = col_ind[cost_mask]
        if self.training:
            veh_accept_idx = filter_indices[veh_accept_idx]
        return torch.tensor(veh_accept_idx), torch.tensor(inf_accept_idx), attn

    def _query_matching(self, inf_ref_pts, veh_ref_pts, mask):
        """
        inf_ref_pts: [..., 3] (xyz)
        veh_ref_pts: [..., 3] (xyz)
        veh_pred_dims: [..., 3] (dx, dy, dz)
        """
        inf_nums = inf_ref_pts.shape[0]
        veh_nums = veh_ref_pts.shape[0]
        cost_matrix = np.ones((veh_nums, inf_nums)) * 1e6

        veh_ref_pts_np = veh_ref_pts.detach().cpu().numpy()
        inf_ref_pts_np = inf_ref_pts.cpu().numpy()

        veh_ref_pts_repeat = np.repeat(veh_ref_pts_np[:, np.newaxis], inf_nums, axis=1)
        distances = np.sqrt(np.sum((veh_ref_pts_repeat - inf_ref_pts_np)**2, axis=2))

        mask = distances > 3.5
        cost_matrix[~mask] = distances[~mask]
        
        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)

        cost_mask = cost_matrix[idx_veh, idx_inf] < 1e5
        veh_accept_idx = idx_veh[cost_mask]
        inf_accept_idx = idx_inf[cost_mask]

        return veh_accept_idx, inf_accept_idx, cost_matrix
    
    def _gen_asso_label(self, gt_boxes, inf_boxes, veh_boxes, sample_idx):
        """
        [Association Label 생성 함수]
        - Vehicle query(veh_boxes)와 Infra query(inf_boxes)가 "같은 객체를 보고 있는지"를 판단하는
        supervision label(정답 행렬)을 생성하는 함수이다.

        최종 출력:
            label: [veh_num, inf_num]
            label[i, j] = 1  -> vehicle의 i번째 box와 infra의 j번째 box가 같은 GT 객체를 가리킨다(positive pair)
            label[i, j] = 0  -> 서로 다른 객체이거나 매칭 불가

        핵심 아이디어:
        1) veh_boxes와 GT를 Hungarian matching으로 1:1 매칭
        2) inf_boxes와 GT를 Hungarian matching으로 1:1 매칭
        3) "같은 GT에 매칭된 veh box"와 "같은 GT에 매칭된 inf box"를 묶어서
        veh-inf association positive label을 생성한다.
        
        즉, veh-inf 직접 비교를 안 하고,
        GT를 매개체로 해서 "veh ↔ GT ↔ inf" 연결을 만든다.
        (GT가 같은 애들끼리 veh-inf도 같은 객체라고 보겠다!)

        Args:
            gt_boxes:
                - 학습용 GT 3D box 리스트 (배치 구조 포함)
                - 여기서는 gt_boxes[0].tensor 형태로 가져와서 사용

            inf_boxes (Tensor):
                - infra 에이전트가 예측한 bbox들
                - shape: [inf_num, box_dim]
                - 여기서는 중심좌표(x,y)만 사용해서 GT와 매칭

            veh_boxes (Tensor):
                - vehicle 에이전트가 예측한 bbox들
                - shape: [veh_num, box_dim]

            sample_idx:
                - 디버깅/로그용 sample id (여기서는 함수 내부에서 사용하지 않음)

        Returns:
            label (Tensor):
                - shape: [veh_num, inf_num]
                - dtype: torch.long
        """

        # torch.no_grad()를 쓰는 이유:
        # - Hungarian matching 과정은 학습 가능한 연산이 아니고,
        # - label 생성은 "정답 만들기" 작업이기 때문에 gradient를 계산할 필요가 없음
        with torch.no_grad():

            # infra / vehicle box 개수 파악
            inf_num = inf_boxes.shape[0]  # infra box 수
            veh_num = veh_boxes.shape[0]  # vehicle box 수

            # label matrix 생성 (초기값 0)
            # label[i, j] = 1이면 positive association
            device = veh_boxes.device
            label = torch.zeros((veh_num, inf_num), dtype=torch.long, device=device)

            # ============================================================
            # (1) GT Box 준비 (normalize)
            # ============================================================
            # gt_boxes는 배치 구조라서 gt_boxes[0]을 가져온다.
            # gt_boxes[0].tensor: [num_gt, box_dim]
            gt_boxes = gt_boxes[0].tensor.to(device)

            # GT bbox를 pc_range 기준으로 normalize (0~1)
            # -> 아래 거리 계산에서 차량/infra 예측 box와 scale을 맞추려는 목적
            gt_boxes = normalize_bbox(gt_boxes, self.pc_range)

            # ============================================================
            # (2) Hungarian Matching 함수 정의
            # ============================================================
            # box1 (veh 또는 inf) 과 box2 (GT) 를 거리 기반 cost로 매칭한다.
            #
            # Hungarian 알고리즘은 "전체 매칭 비용(cost)의 합을 최소화"하는
            # 1:1 assignment를 찾아준다.
            #
            # 여기서는 cost = 거리(distance)
            # 단, 너무 멀리 떨어진 pair는 매칭 후보에서 제외하기 위해 cost=1e6으로 둔다.
            def hungarian_matching(box1, box2, thre=5.0):

                # box1: [N1, box_dim], box2: [N2, box_dim]
                box1_nums = box1.shape[0]
                box2_nums = box2.shape[0]

                # cost matrix 초기화: [N1, N2]
                # 기본값을 매우 큰 값(1e6)으로 두고,
                # threshold 안에 들어오는 pair만 실제 거리 cost를 채워준다.
                cost_matrix = np.ones((box1_nums, box2_nums)) * 1e6

                # numpy로 변환해서 거리 계산
                box1_np = box1.cpu().numpy()
                box2_np = box2.cpu().numpy()

                # bbox 비교는 중심점 (x,y)만 사용
                # index 의미:
                #   box[:, 0] = center x
                #   box[:, 1] = center y
                box1_np = box1_np[:, 0:2]
                box2_np = box2_np[:, 0:2]

                # 모든 pair (box1_i, box2_j)의 유클리드 거리 계산
                # box1_repeat: [N1, N2, 2]
                box1_repeat = np.repeat(box1_np[:, np.newaxis], box2_nums, axis=1)
                distances = np.sqrt(np.sum((box1_repeat - box2_np) ** 2, axis=2))  # [N1, N2]

                # threshold 밖(여기서는 3.0m 이상)은 invalid 처리
                # -> 해당 pair는 매칭 후보에서 제외된다.
                # mask=True는 "멀다(매칭 불가)"
                mask = distances > 3.0

                # threshold 안에 있는 pair만 cost_matrix를 거리로 채우기
                cost_matrix[~mask] = distances[~mask]

                # Hungarian assignment 수행
                # idx_row: box1 인덱스 배열
                # idx_col: box2 인덱스 배열
                # (둘은 같은 길이이며 1:1 매칭 결과)
                idx_row, idx_col = linear_sum_assignment(cost_matrix)

                # 매칭 결과 중에서도 cost가 너무 큰 것(=사실상 1e6에 가까운 것)은 제거
                # cost<1e5이면 "유효한 매칭"으로 인정
                cost_mask = cost_matrix[idx_row, idx_col] < 1e5

                # 최종적으로 accept된 box index만 남김
                row_accept_idx = idx_row[cost_mask]  # box1 쪽 index
                col_accept_idx = idx_col[cost_mask]  # box2(GT) 쪽 index

                return row_accept_idx, col_accept_idx

            # ============================================================
            # (3) vehicle box ↔ GT box 매칭
            # ============================================================
            # matched_veh_indices: vehicle box index들
            # matched_gt_indices_from_v: 그 vehicle box가 매칭된 GT index들
            matched_veh_indices, mathced_gt_indices_from_v = hungarian_matching(
                veh_boxes, gt_boxes
            )

            # ============================================================
            # (4) infra box ↔ GT box 매칭
            # ============================================================
            matched_inf_indices, mathced_gt_indices_from_i = hungarian_matching(
                inf_boxes, gt_boxes
            )

            # ============================================================
            # (5) GT를 기준으로 veh/inf box를 그룹핑하기 위한 map 생성
            # ============================================================
            # 예를 들어,
            #   GT #3에 매칭된 veh box가 [1, 5]이고
            #   GT #3에 매칭된 inf box가 [0]이면
            #   (veh 1 ↔ inf 0), (veh 5 ↔ inf 0)를 positive로 만든다.

            from collections import defaultdict

            # gt_to_veh[gt_idx] = [veh_idx1, veh_idx2, ...]
            gt_to_veh = defaultdict(list)
            for veh_idx, gt_idx in zip(matched_veh_indices, mathced_gt_indices_from_v):
                gt_to_veh[gt_idx].append(veh_idx)

            # gt_to_inf[gt_idx] = [inf_idx1, inf_idx2, ...]
            gt_to_inf = defaultdict(list)
            for inf_idx, gt_idx in zip(matched_inf_indices, mathced_gt_indices_from_i):
                gt_to_inf[gt_idx].append(inf_idx)

            # ============================================================
            # (6) "같은 GT를 공유하는 veh-inf pair"를 positive(1)로 표시
            # ============================================================
            # 같은 GT 객체에 대해 veh도 매칭되고 inf도 매칭되었다면
            # => veh query와 inf query는 같은 객체를 가리키는 것으로 판단한다.
            pairs = []  # 디버깅/분석용 (veh_idx, inf_idx, gt_idx) 기록

            for gt_idx in gt_to_veh:
                # 해당 GT에 대해 infra도 매칭되어 있어야 association label 가능
                if gt_idx in gt_to_inf:
                    for veh_idx in gt_to_veh[gt_idx]:
                        for inf_idx in gt_to_inf[gt_idx]:
                            label[veh_idx, inf_idx] = 1  # positive pair
                            pairs.append((veh_idx, inf_idx, gt_idx))

        # label 반환
        # label[i, j] = 1이면 vehicle query i와 infra query j가 같은 GT 객체를 공유함
        return label

    
    def init_params_and_layers(self):
        # Modules for history reasoning
        if self.history_reasoning:
            # temporal transformer
            self.hist_temporal_transformer = build_transformer(self.hist_temporal_transformer)
            self.spatial_transformer = build_transformer(self.spatial_transformer)
            if self.is_motion:
                self.hist_motion_transformer = copy.deepcopy(self.hist_temporal_transformer)
            # classification refinement from history
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.embed_dims, self.num_classes))
            self.track_cls = nn.Sequential(*cls_branch)
    
            # localization refinement from history
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.LayerNorm(self.embed_dims))
                reg_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            self.track_reg = nn.Sequential(*reg_branch)
        
        # Modules for future reasoning
        if self.future_reasoning:
            # temporal transformer
            self.fut_temporal_transformer = build_transformer(self.fut_temporal_transformer)

            # regression head
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.LayerNorm(self.embed_dims))
                reg_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(Linear(self.embed_dims, 3))
            self.future_reg = nn.Sequential(*reg_branch)
        
        if self.future_reasoning or self.history_reasoning:
            self.ts_query_embed = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        
        if self.is_cooperation:
            self.fuse_feats = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.fuse_motion = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.fuse_embed = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.cross_domain_query = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.cross_domain_motion = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )

            if self.learn_match:
                self.get_node_veh = nn.Sequential(
                    nn.Linear(self.embed_dims*2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.get_node_inf = nn.Sequential(
                    nn.Linear(self.embed_dims*2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.get_edge = nn.Sequential(
                    nn.Linear(3, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.get_dummy_node = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.get_dummy_pts = nn.Sequential(
                    nn.Linear(3, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, 3)
                )
                self.wq = nn.Parameter(torch.Tensor(self.embed_dims, self.embed_dims))
                self.wk = nn.Parameter(torch.Tensor(self.embed_dims, self.embed_dims))
                self.we = nn.Parameter(torch.Tensor(self.embed_dims, self.embed_dims))
                torch.nn.init.kaiming_uniform_(self.wq, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.wk, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.we, a=math.sqrt(5))
                self.ffn = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, 1)
                )
                self.sigmoid = nn.Sigmoid()
        return
    
    def sync_pos_embedding(self, track_instances: Instances, mlp_embed: nn.Module = None):
        """
        [Positional Embedding 동기화 함수]
        - tracking / temporal reasoning 과정에서 reference point(ref_pts) 또는 history 위치(hist_xyz)가 바뀌면,
        그에 대응되는 positional embedding(query_embeds / hist_position_embeds)도 반드시 다시 계산해야 한다.
        - 이 함수는 "현재 프레임 query의 위치 임베딩" + "history buffer의 위치 임베딩"을
        ref_pts / hist_xyz 기준으로 최신화(sync)하는 역할을 한다.

        왜 필요한가?
        - ref_pts는 다음 프레임으로 넘어가기 전에 motion prediction 또는 ego-motion 보정으로 계속 갱신됨
        - 하지만 query_embeds는 이전 ref_pts 기준으로 만들어져 있을 수 있음
        - positional embedding이 실제 위치(ref_pts)와 안 맞으면 transformer attention에서
        "현재 객체가 어디 있는지"를 잘못 인식할 수 있음
        => 따라서 ref_pts가 바뀔 때마다 query_embeds도 반드시 다시 만들어줘야 함

        Args:
            track_instances (Instances):
                - Track query 상태를 담는 컨테이너
                - 주요 필드:
                    - ref_pts: [N, 3]
                        * 각 query의 현재 reference point (정규화된 x,y,z)
                    - query_embeds: [N, C]
                        * ref_pts를 positional embedding으로 변환한 결과 (query position embedding)
                    - hist_xyz: [N, hist_len, 3]
                        * 각 query의 history 위치(과거 reference point들)
                    - hist_position_embeds: [N, hist_len, C]
                        * hist_xyz를 positional embedding으로 변환한 history positional embedding

            mlp_embed (nn.Module):
                - pos2posemb3d()로 만든 3D positional encoding을
                최종 embed_dims(C)로 매핑하는 MLP 모듈
                - 보통 self.query_embedding 같은 형태이며,
                입력 차원(예: 3D sine/cos encoding)을 embed_dims로 바꿔준다.

        Returns:
            track_instances:
                - query_embeds, hist_position_embeds가 최신 ref_pts/hist_xyz 기준으로 업데이트된 Instances
        """

        # mlp_embed가 주어졌을 때만 positional embedding 업데이트 수행
        # (mlp_embed가 None이면 아무 것도 하지 않고 그대로 반환)
        if mlp_embed is not None:

            # ============================================================
            # (1) 현재 프레임 query positional embedding 업데이트
            # ============================================================
            # track_instances.ref_pts: [N, 3]  (현재 reference point)
            #
            # pos2posemb3d(ref_pts):
            #   - (x,y,z) 좌표를 sine/cos 기반의 positional encoding 벡터로 변환
            #   - 출력 shape은 구현에 따라 [N, something] (예: embed_dims*3/2 등)
            #
            # mlp_embed(...):
            #   - 위 positional encoding을 transformer query embedding 차원(embed_dims)으로 변환
            #   - 결과: [N, C]
            #
            # 즉 query_embeds는 "현재 query가 어디 위치하는지"를 나타내는 좌표 임베딩이다.
            track_instances.query_embeds = mlp_embed(
                pos2posemb3d(track_instances.ref_pts)
            )

            # ============================================================
            # (2) history buffer 전체에 대한 positional embedding 업데이트
            # ============================================================
            # track_instances.hist_xyz: [N, hist_len, 3]
            # - 각 query가 과거 hist_len 프레임 동안 존재했던 위치(ref point들)를 저장한 버퍼
            #
            # pos2posemb3d(hist_xyz):
            # - [N, hist_len, 3] -> [N, hist_len, ?] 형태로 변환 (각 시점 위치를 embedding으로)
            #
            # mlp_embed(...)를 적용하면:
            # - [N, hist_len, ?] -> [N, hist_len, C]
            #
            # 즉 hist_position_embeds는 "시간별 위치 임베딩"이며,
            # temporal transformer에서 과거 feature들을 attention할 때
            # "이 feature가 어느 위치에 있었는지" 정보를 전달해준다.
            track_instances.hist_position_embeds = mlp_embed(
                pos2posemb3d(track_instances.hist_xyz)
            )

        # 최신 positional embedding이 반영된 track_instances 반환
        return track_instances

    def update_ego(self, track_instances: Instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        Args:
            track_instances: objects
            l2g0: a [4x4] matrix for current frame lidar-to-global transformation 
            l2g1: a [4x4] matrix for target frame lidar-to-global transformation
        Return:
            transformed track_instances (inplace)
        """
        # TODO: orientation of the bounding boxes
        """1. Current states"""
        ref_points = track_instances.ref_pts.clone()
        physical_ref_points = xyz_ego_transformation(ref_points, l2g_r1, l2g_t1, l2g_r2, l2g_t2, self.pc_range,
                                                     src_normalized=True, tgt_normalized=False)
        track_instances.pred_boxes[..., [0, 1, 4]] = physical_ref_points.clone()
        track_instances.ref_pts = normalize(physical_ref_points, self.pc_range)
        
        """2. History states"""
        inst_num = len(track_instances)
        hist_ref_xyz = track_instances.hist_xyz.clone().view(inst_num * self.hist_len, 3)
        physical_hist_ref = xyz_ego_transformation(hist_ref_xyz, l2g_r1, l2g_t1, l2g_r2, l2g_t2, self.pc_range,
                                                   src_normalized=True, tgt_normalized=False)
        physical_hist_ref = physical_hist_ref.reshape(inst_num, self.hist_len, 3)
        track_instances.hist_bboxes[..., [0, 1, 4]] = physical_hist_ref
        track_instances.hist_xyz = normalize(physical_hist_ref, self.pc_range)
        
        """3. Future states"""
        inst_num = len(track_instances)
        fut_ref_xyz = track_instances.fut_xyz.clone().view(inst_num * self.fut_len, 3)
        physical_fut_ref = xyz_ego_transformation(fut_ref_xyz, l2g_r1, l2g_t1, l2g_r2, l2g_t2, self.pc_range,
                                                   src_normalized=True, tgt_normalized=False)
        physical_fut_ref = physical_fut_ref.reshape(inst_num, self.fut_len, 3)
        track_instances.fut_bboxes[..., [0, 1, 4]] = physical_fut_ref
        track_instances.fut_xyz = normalize(physical_fut_ref, self.pc_range)

        return track_instances
    
    def update_reference_points(self, track_instances, time_deltas, use_prediction=True, tracking=False):
        """
        [Reference Point 업데이트 함수]
        - 다음 프레임으로 넘어가기 전에, 현재 track query가 들고 있는 reference point(ref_pts)를
        "미래 위치 예측(motion prediction)" 또는 "bbox velocity"를 사용해서 앞으로 이동시킨다.
        - 핵심 목적:
            ✅ 다음 프레임의 detection/decoder가 더 좋은 초기 위치(ref point)에서 시작하도록 유도
            ✅ tracking에서 query들이 객체를 계속 따라갈 수 있도록 "초기화 위치"를 보정

        Args:
            track_instances (Instances):
                - 현재 프레임의 track query 상태를 담고 있는 컨테이너
                - 주요 필드:
                    - ref_pts: [N, 3]  (normalized reference points, 보통 (x, y, z) 0~1)
                    - pred_boxes: [N, box_dim] (bbox prediction 결과)
                    - motion_predictions: [N, fut_len, 3] (미래 이동량 예측, 보통 (dx, dy, dz))
                    - fut_xyz: [N, fut_len, 3] (미래 위치를 normalized ref_pts 형태로 저장해둔 버퍼)
                    - fut_bboxes: [N, fut_len, box_dim] (미래 bbox를 저장해둔 버퍼)

            time_deltas:
                - 프레임 간 시간차 Δt
                - use_prediction=False 인 경우 velocity 기반 이동에서 필요
                - shape은 보통 scalar 또는 [N] 형태일 수 있음

            use_prediction (bool):
                - True : 모델이 예측한 미래 motion 결과를 이용하여 ref point 업데이트
                - False: bbox에 포함된 velocity(속도) 값을 이용하여 ref point 업데이트

            tracking (bool):
                - True : inference(tracking) 모드
                        -> multi-step forecasting 결과(track_instances.motion_predictions)를 써서 ref point 이동
                - False: training 모드
                        -> single-step 예측값(track_instances.fut_xyz[0])을 바로 ref point로 사용

        Returns:
            track_instances:
                - ref_pts가 업데이트된 track_instances 반환
        """

        # ---------------------------------------------------------------------
        # 1) use_prediction=True : "모션 예측 기반"으로 ref_pts를 업데이트
        # ---------------------------------------------------------------------
        if use_prediction:

            # ================================
            # (1-A) inference / tracking 모드
            # ================================
            # - motion_predictions의 "첫 번째 스텝" 이동량을 ref_pts에 더해서
            #   다음 프레임의 초기 ref point로 사용한다.
            # - 이 방식은 "현재 프레임에서 예측한 미래 이동량"을 이용하므로
            #   객체가 빠르게 움직여도 query가 따라가기가 쉬워진다.
            if tracking:
                # motion_predictions: [N, fut_len, 3]
                # 여기서 [:, 0, :2] 는 "미래 첫 스텝 이동량"의 (dx, dy)만 사용한다.
                #   - 첫 번째 인덱스 [:] : 모든 query(track) 선택
                #   - 두 번째 인덱스 [0] : 미래 예측 step 중 0번째 (= 다음 스텝)
                #   - 세 번째 인덱스 [:2]: (dx, dy)만 사용 (z 변화는 여기서는 무시)
                motions = track_instances.motion_predictions[:, 0, :2].clone()  # [N, 2]

                # ref_pts: [N, 3] (normalized)
                reference_points = track_instances.ref_pts.clone()

                # motions는 보통 "실제 meter 단위 이동량(dx, dy)"로 나올 수 있음
                # 그런데 ref_pts는 normalized 좌표(0~1)로 관리하기 때문에
                # pc_range 기반으로 정규화(normalize)해서 단위를 맞춰준다.
                #
                # pc_range 예시: [x_min, y_min, z_min, x_max, y_max, z_max]
                # index 설명:
                #   pc_range[0] = x_min
                #   pc_range[3] = x_max   => (pc_range[3] - pc_range[0]) = x 축 범위
                #   pc_range[1] = y_min
                #   pc_range[4] = y_max   => (pc_range[4] - pc_range[1]) = y 축 범위
                motions[:, 0] /= (self.pc_range[3] - self.pc_range[0])  # dx 정규화
                motions[:, 1] /= (self.pc_range[4] - self.pc_range[1])  # dy 정규화

                # ref_pts[..., :2] 는 (x, y) 좌표를 의미한다.
                # index 설명:
                #   ref_pts shape: [N, 3] = [x, y, z] (normalized)
                #   ref_pts[..., 0] = x
                #   ref_pts[..., 1] = y
                #   ref_pts[..., 2] = z
                #
                # motions는 detach()해서 gradient가 과거로 연결되는 것을 방지
                # (tracking/inference에서는 gradient 학습이 목적이 아니고,
                #  ref_pts 업데이트는 단순 state update이기 때문)
                reference_points[..., :2] += motions.clone().detach()

                # 업데이트된 ref_pts 저장
                track_instances.ref_pts = reference_points

            # ================================
            # (1-B) training 모드
            # ================================
            # - training에서는 inference처럼 motion_predictions를 직접 쓰기보다,
            #   STReasoner/FrameSummarization 과정에서 만들어둔 fut_xyz / fut_bboxes의 첫 스텝 값을
            #   다음 프레임 초기값으로 그대로 사용한다.
            #
            # fut_xyz: [N, fut_len, 3]
            # fut_bboxes: [N, fut_len, box_dim]
            else:
                # fut_xyz[:, 0, :] :
                #   - 모든 query에서
                #   - 미래 첫 step(0번째)의 위치 [x,y,z] 를 가져온다.
                track_instances.ref_pts = track_instances.fut_xyz[:, 0, :].clone()      # [N, 3]

                # fut_bboxes[:, 0, :] :
                #   - 미래 첫 step의 bbox를 다음 프레임 초기 bbox로 사용
                track_instances.pred_boxes = track_instances.fut_bboxes[:, 0, :].clone()  # [N, box_dim]

        # ---------------------------------------------------------------------
        # 2) use_prediction=False : "bbox에 포함된 velocity(속도)" 기반으로 ref_pts 업데이트
        # ---------------------------------------------------------------------
        else:
            # pred_boxes에서 속도 값을 읽어오는 방식
            # velos = pred_boxes[..., 8:10]
            #
            # ✅ index 설명 (매우 중요):
            # pred_boxes의 box_dim 구성은 보통 nuscenes 계열/BEVFormer 계열에서
            # 다음처럼 정의되는 경우가 많다 (코드/설정마다 조금씩 다를 수 있음)
            #
            # 예시 (10차원):
            #   [0] x        : center x
            #   [1] y        : center y
            #   [2] z        : center z
            #   [3] w        : width
            #   [4] l        : length
            #   [5] h        : height
            #   [6] yaw      : heading angle
            #   [7] ?        : (optional) velocity magnitude / sin/cos yaw 등
            #   [8] vx       : x축 속도
            #   [9] vy       : y축 속도
            #
            # 이 코드에서는 [8:10]을 vx, vy로 가정하고 사용한다.
            velos = track_instances.pred_boxes[..., 8:10].clone()  # [N, 2] = (vx, vy)

            reference_points = track_instances.ref_pts.clone()     # [N, 3] normalized

            # 속도도 meters/sec 단위라면 ref_pts(0~1 normalized)로 변환하기 위해 정규화 필요
            velos[:, 0] /= (self.pc_range[3] - self.pc_range[0])   # vx normalize
            velos[:, 1] /= (self.pc_range[4] - self.pc_range[1])   # vy normalize

            # ref_pts 업데이트:
            # reference_points[..., :2] += velos * time_deltas
            #
            # - velos: [N,2]
            # - time_deltas: Δt
            # 따라서 이동량 = 속도 * 시간
            #
            # detach()를 적용해서 state update가 학습 그래프를 복잡하게 만들지 않도록 함
            reference_points[..., :2] += (velos * time_deltas).clone().detach()

            # 업데이트된 ref_pts 저장
            track_instances.ref_pts = reference_points

        return track_instances
        
    def frame_shift(self, track_instances: Instances):
        """
        Shift the information for the newest frame before spatial-temporal reasoning happens. 
        Pay attention to the order.
        가장 오래된 frame 정보를 버리고 현재 frame 업데이트
        """
        device = track_instances.query_feats.device
        
        """1. History reasoining"""
        # embeds
        track_instances.hist_embeds = track_instances.hist_embeds.clone()
        track_instances.hist_embeds = torch.cat((
            # hist_embeds[:,1:,:] => 가장 오래된 슬롯(0번째)을 버리고 뒤만 남김, [:, None, :] => 현재 프레임 query feature를 time dim으로 확장해서 붙임
            track_instances.hist_embeds[:, 1:, :], track_instances.cache_query_feats[:, None, :]), dim=1) 
        '''
            before: [t-h+1, t-h+2, ..., t-1]
            after : [t-h+2, t-h+3, ..., t]
            즉, 현재 cache_query_feats가 history의 “마지막 프레임”이 됨
        '''
        
        # padding masks
        # MHA 스타일 padding mask, True = pad, False = valid
        track_instances.hist_padding_masks = torch.cat((
            track_instances.hist_padding_masks[:, 1:], 
            torch.zeros((len(track_instances), 1), dtype=torch.bool, device=device)),  # oldest를 버리고 False(0)를 붙임
            dim=1)
        # positions
        track_instances.hist_xyz = torch.cat((
            track_instances.hist_xyz[:, 1:, :], track_instances.cache_ref_pts[:, None, :]), dim=1)
        # positional embeds
        track_instances.hist_position_embeds = torch.cat((
            track_instances.hist_position_embeds[:, 1:, :], track_instances.cache_query_embeds[:, None, :]), dim=1)
        # bboxes
        track_instances.hist_bboxes = torch.cat((
            track_instances.hist_bboxes[:, 1:, :], track_instances.cache_bboxes[:, None, :]), dim=1)
        # logits
        track_instances.hist_logits = torch.cat((
            track_instances.hist_logits[:, 1:, :], track_instances.cache_logits[:, None, :]), dim=1)
        # scores
        track_instances.hist_scores = torch.cat((
            track_instances.hist_scores[:, 1:], track_instances.cache_scores[:, None]), dim=1)
        # motion features
        track_instances.hist_motion_embeds = track_instances.hist_motion_embeds.clone()
        track_instances.hist_motion_embeds = torch.cat((
            track_instances.hist_motion_embeds[:, 1:, :], track_instances.cache_motion_feats[:, None, :]), dim=1)
        """2. Temporarily load motion predicted results as final results"""
        if self.future_reasoning:
            track_instances.ref_pts = track_instances.fut_xyz[:, 0, :].clone()
            track_instances.pred_boxes = track_instances.fut_bboxes[:, 0, :].clone()
            track_instances.scores = track_instances.fut_scores[:, 0].clone() + 0.01
            track_instances.pred_logits = track_instances.fut_logits[:, 0, :].clone()

        """3. Future reasoning"""
        # TODO: shift the future information, useful for occlusion handling
        track_instances.motion_predictions = torch.cat((
            track_instances.motion_predictions[:, 1:, :], torch.zeros_like(track_instances.motion_predictions[:, 0:1, :])), dim=1)
        track_instances.fut_embeds = torch.cat((
            track_instances.fut_embeds[:, 1:, :], torch.zeros_like(track_instances.fut_embeds[:, 0:1, :])), dim=1)
        track_instances.fut_padding_masks = torch.cat((
            track_instances.fut_padding_masks[:, 1:], torch.ones_like(track_instances.fut_padding_masks[:, 0:1]).bool()), dim=1)
        track_instances.fut_xyz = torch.cat((
            track_instances.fut_xyz[:, 1:, :], torch.zeros_like(track_instances.fut_xyz[:, 0:1, :])), dim=1)
        track_instances.fut_position_embeds = torch.cat((
            track_instances.fut_position_embeds[:, 1:, :], torch.zeros_like(track_instances.fut_position_embeds[:, 0:1, :])), dim=1)
        track_instances.fut_bboxes = torch.cat((
            track_instances.fut_bboxes[:, 1:, :], torch.zeros_like(track_instances.fut_bboxes[:, 0:1, :])), dim=1)
        track_instances.fut_logits = torch.cat((
            track_instances.fut_logits[:, 1:, :], torch.zeros_like(track_instances.fut_logits[:, 0:1, :])), dim=1)
        track_instances.fut_scores = torch.cat((
            track_instances.fut_scores[:, 1:], torch.zeros_like(track_instances.fut_scores[:, 0:1])), dim=1)
        return track_instances