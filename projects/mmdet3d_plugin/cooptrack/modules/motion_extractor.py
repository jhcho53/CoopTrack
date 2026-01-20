import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

class MotionExtractor(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 mlp_channels=(3, 64, 64, 256),
                 ):
        """
        [MotionExtractor]
        - Detection head가 예측한 bbox(cache_bboxes)를 기반으로
          "motion reasoning에 사용할 motion feature(cache_motion_feats)"를 생성하는 모듈

        목표:
        ✅ bbox의 기하학적 형태(크기/방향) + 속도 정보(vx, vy)를 함께 임베딩해서
           하나의 motion embedding(256차원)을 만든다.
        ✅ 이 motion embedding은 이후 STReasoner(temporal reasoning / association 등)에서 사용됨.

        Args:
            embed_dims (int):
                - 최종 motion feature의 차원 (기본 256)
            mlp_channels (tuple):
                - PointNet/VelocityNet에서 사용하는 MLP 채널 구성
                - 예: (3, 64, 64, 256)
                  입력 3차원 → 64 → 64 → 256으로 점(feature)을 임베딩
        """
        super(MotionExtractor, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_channels = mlp_channels

        # ============================================================
        # (1) Geometry Embedding용 PointNet 구성
        # ============================================================
        """
        [PointNet 구축]
        - bbox를 8개 corner 점들로 표현한 뒤,
          각 점(3차원 xyz)을 MLP로 임베딩 → max pooling으로 하나의 box feature로 만들기

        PointNet 흐름:
            normalized_corners: [N, 8, 3]
                -> MLP(3→64→64→256) 적용: [N, 8, 256]
                -> max pooling over points(8): [N, 256]
        """
        pointnet = []
        for i in range(len(self.mlp_channels) - 1):
            # Linear(in_dim, out_dim)
            pointnet.append(Linear(self.mlp_channels[i], self.mlp_channels[i + 1]))
            # LayerNorm(out_dim): feature 안정화
            pointnet.append(nn.LayerNorm(self.mlp_channels[i + 1]))
            # ReLU: 비선형성 추가
            pointnet.append(nn.ReLU())
        self.pointnet = nn.Sequential(*pointnet)

        # ============================================================
        # (2) Velocity Embedding용 MLP(VeloNet) 구성
        # ============================================================
        """
        [VeloNet 구축]
        - 속도(vx, vy)와 방향(rot) 정보를 embedding하기 위한 MLP
        - pointnet과 동일 구조로 구성하되 입력이 motion vector가 된다.

        주의:
        - 코드에서는 bboxes_motion = decode_bboxes[:, 6:9] 를 넣는다.
        - 즉 (rot, vx, vy) 3차원 정보를 입력으로 넣는 구조다.
          (index 6: rot, index 7: vx, index 8: vy 라고 가정)
        """
        velonet = []
        for i in range(len(self.mlp_channels) - 1):
            velonet.append(Linear(self.mlp_channels[i], self.mlp_channels[i + 1]))
            velonet.append(nn.LayerNorm(self.mlp_channels[i + 1]))
            velonet.append(nn.ReLU())
        self.velonet = nn.Sequential(*velonet)

        # ============================================================
        # (3) Geometry + Velocity Feature Fusion 네트워크
        # ============================================================
        """
        [Fusion Net]
        - point_feat: [N, 256]
        - motion_feat: [N, 256]
        -> concat: [N, 512]
        -> Linear(512→256)로 다시 256차원으로 압축/융합

        최종 결과:
            fusion_feat: [N, 256]
            -> track_instances.cache_motion_feats 로 저장
        """
        fusion_net = []
        fusion_net.append(Linear(self.embed_dims * 2, self.embed_dims))
        fusion_net.append(nn.LayerNorm(self.embed_dims))
        fusion_net.append(nn.ReLU())
        self.fusion_net = nn.Sequential(*fusion_net)

    def forward(self, track_instances, img_metas):
        """
        [Forward]
        - track_instances.cache_bboxes 에 들어있는 bbox prediction을 사용해서
          motion embedding을 만든 뒤 cache_motion_feats에 저장한다.

        Args:
            track_instances (Instances):
                - cache_bboxes: [N, box_dim]
                  예측 bbox (보통 normalized 상태)
            img_metas:
                - 여기서는 직접 사용하지 않지만, 필요하면 scale/ego 정보 등을 참조 가능

        Returns:
            track_instances:
                - cache_motion_feats가 채워진 Instances
        """

        # ------------------------------------------------------------
        # (0) 현재 프레임 bbox prediction 가져오기
        # ------------------------------------------------------------
        # cache_bboxes는 load_detection_output_into_cache()에서 채워진 값
        pred_bboxes = track_instances.cache_bboxes.clone()

        # ------------------------------------------------------------
        # (1) bbox를 실제 좌표계로 복원(denormalize)
        # ------------------------------------------------------------
        # pred_bboxes는 보통 normalize_bbox()를 통해 0~1 범위로 저장되어 있음
        # denormalize_bbox()로 실제 meter 단위 좌표로 복원
        #
        # decode_bboxes 예시 구성(주석 기준):
        #   [cx, cy, cz, w, l, h, rot, vx, vy]  (총 9차원)
        decode_bboxes = denormalize_bbox(pred_bboxes, None)

        # ------------------------------------------------------------
        # (2) Geometry 정보 추출: bbox의 위치/크기/방향(속도 제외)
        # ------------------------------------------------------------
        # bboxes_geometric = decode_bboxes[:, 0:7]
        #
        # index 설명:
        #   0: cx  (center x)
        #   1: cy  (center y)
        #   2: cz  (center z)
        #   3: w   (width)
        #   4: l   (length)
        #   5: h   (height)
        #   6: rot (yaw/heading)
        #
        # => 속도(vx,vy)는 제외하고 "박스 형태/방향"만 사용
        bboxes_geometric = decode_bboxes[:, 0:7]

        # LiDAR 3D Box 객체로 생성
        # - corners, center 등의 geometry 연산을 쉽게 하기 위해 사용
        bbox = LiDARInstance3DBoxes(
            bboxes_geometric,
            box_dim=7
        )

        # corners: [N, 8, 3]
        # - 각 bbox를 8개의 꼭짓점 좌표로 변환한 결과
        corners = bbox.corners

        # gravity_center: [N, 3]
        # - bbox의 중심점(보통 바닥 중심이 아니라 무게중심 형태)
        center = bbox.gravity_center

        # ------------------------------------------------------------
        # (3) corner 좌표를 중심 기준으로 정규화(normalize)
        # ------------------------------------------------------------
        # normalized_corners = corners - center
        #
        # 의미:
        #   - 절대 위치(cx,cy,cz)는 제거하고
        #   - 박스의 "모양/크기/방향"만 남기기 위해 중심을 기준으로 이동
        #
        # shape:
        #   corners: [N, 8, 3]
        #   center.unsqueeze(1): [N, 1, 3] (broadcast 가능)
        #   결과 normalized_corners: [N, 8, 3]
        normalized_corners = corners - center.unsqueeze(1)

        # ============================================================
        # ( Geometry Embedding (PointNet)
        # ============================================================
        # pointnet 입력: [N, 8, 3]
        # pointnet 출력: [N, 8, 256]
        point_feat = self.pointnet(normalized_corners)

        # PointNet의 핵심: 점들(8개)에 대해 max pooling
        # torch.max(point_feat, dim=1)[0] :
        #   - dim=1(8개의 corner 축)에서 최대값 pooling
        #   - 결과: [N, 256] (bbox를 대표하는 geometry feature)
        point_feat = torch.max(point_feat, dim=1)[0]

        # ============================================================
        #   Motion(velocity) Embedding (VeloNet)
        # ============================================================
        # bboxes_motion = decode_bboxes[:, 6:9]
        #
        # index 설명:
        #   6: rot (yaw)
        #   7: vx  (x 방향 속도)
        #   8: vy  (y 방향 속도)
        #
        # => "방향 + 속도"를 하나의 3차원 벡터로 사용
        #
        # shape: [N, 3]
        bboxes_motion = decode_bboxes[:, 6:9]

        # velonet 입력: [N, 3]
        # velonet 출력: [N, 256]
        motion_feat = self.velonet(bboxes_motion)

        # ============================================================
        #   Fusion Embedding (Geometry + Motion)
        # ============================================================
        # concat: [N, 256] + [N, 256] -> [N, 512]
        fusion_feat = torch.cat([point_feat, motion_feat], dim=1)

        # fusion_net: [N, 512] -> [N, 256]
        fusion_feat = self.fusion_net(fusion_feat)

        # ------------------------------------------------------------
        # (4) track_instances의 cache_motion_feats에 저장
        # ------------------------------------------------------------
        # 이후 STReasoner.forward_history_reasoning 등에서
        # hist_motion_embeds / cache_motion_feats 를 attention 입력으로 사용 가능
        track_instances.cache_motion_feats = fusion_feat.clone()

        return track_instances