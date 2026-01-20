import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import Linear
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox

class LatentTransformation(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 head=16,
                 rot_dims=6,
                 trans_dims=3,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 inf_pc_range=[0, -51.2, -5.0, 102.4, 51.2, 3.0],
                 ):
        super(LatentTransformation, self).__init__()
        self.embed_dims = embed_dims
        self.head = head
        self.rot_dims = rot_dims
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range

        rot_final_dim = int((embed_dims / head) * (embed_dims / head) * head)
        trans_final_dim = embed_dims

        layers = []
        dims = [rot_dims, embed_dims, embed_dims, rot_final_dim]
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.rot_mlp = nn.Sequential(*layers)

        layers = []
        dims = [trans_dims, embed_dims, embed_dims, trans_final_dim]
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.trans_mlp = nn.Sequential(*layers)

        self.feat_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.embed_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.motion_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )

        self.feat_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.embed_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.motion_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
    
    def continuous_rot(self, rot):
        ret = rot[:, :2].clone()
        ret = ret.reshape(1, -1)
        return ret

    def fill_tensor(self, original_tensor):
        h = self.head
        k = self.embed_dims // self.head
        d = self.embed_dims
        
        blocks_indices = torch.arange(h, device=original_tensor.device) * k
        
        offset = torch.arange(k, device=original_tensor.device)
        rows_offset, cols_offset = torch.meshgrid(offset, offset)
        
        base_rows = blocks_indices.view(h, 1, 1)
        global_rows = base_rows + rows_offset.unsqueeze(0)
        base_cols = blocks_indices.view(h, 1, 1)
        global_cols = base_cols + cols_offset.unsqueeze(0)
        
        all_rows = global_rows.reshape(-1)
        all_cols = global_cols.reshape(-1)
        data = original_tensor.view(-1)
        
        target_tensor = torch.zeros((d, d), device=original_tensor.device)
        target_tensor[all_rows, all_cols] = data
        
        return target_tensor

    def transform_pts(self, points, transformation):
        # relative -> absolute (in inf pc range)
        locs = points.clone()
        locs[:, 0:1] = (locs[:, 0:1] * (self.inf_pc_range[3] - self.inf_pc_range[0]) + self.inf_pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (self.inf_pc_range[4] - self.inf_pc_range[1]) + self.inf_pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (self.inf_pc_range[5] - self.inf_pc_range[2]) + self.inf_pc_range[2])

        # transformation
        locs = torch.cat((locs, torch.ones_like(locs[..., :1])), -1).unsqueeze(-1)
        locs = torch.matmul(transformation, locs).squeeze(-1)[..., :3]
        
        # filter
        mask = (self.pc_range[0] <= locs[:, 0]) & (locs[:, 0] <= self.pc_range[3]) & \
                    (self.pc_range[1] <= locs[:, 1]) & (locs[:, 1] <= self.pc_range[4])
        locs = locs[mask]
        # absolute -> relative (in veh pc range)
        locs[..., 0:1] = (locs[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        return locs, mask
    
    def transform_boxes(self, pred_boxes, transformation):
        device = transformation.device
        transformation = transformation.cpu()
        pred_boxes = pred_boxes.cpu()

        pred_boxes = denormalize_bbox(pred_boxes, self.inf_pc_range)
        pred_boxes = LiDARInstance3DBoxes(pred_boxes, 9)
        rot = transformation[:3, :3].T
        trans = transformation[:3, 3:4].reshape(1, 3)
        pred_boxes.rotate(rot)
        pred_boxes.translate(trans)
        pred_boxes = normalize_bbox(pred_boxes.tensor, self.pc_range)
        pred_boxes = pred_boxes.to(device)
        
        return pred_boxes

    def forward(self, instances, veh2inf_rt):
        """
        [LatentTransformation.forward]
        - 인프라(infra) 에이전트가 만든 query/box 결과(instances)를
        차량(vehicle) 좌표계 기준으로 정렬(alignment)해주는 모듈

        핵심 목표:
        1) infra 좌표계에서 정규화된 reference point(ref_pts)를
        vehicle 좌표계로 변환하고(vehicle pc_range 기준으로 재정규화),
        vehicle ROI 밖으로 나가는 query는 mask로 제거한다.
        2) 좌표계를 바꿨으니, latent feature(query_feats/query_embeds/motion_feats)도
        같은 변환을 반영해서 "feature space에서도 정렬"해준다.
        (rotation + translation을 feature 공간에 적용 + residual 형태로 보정)
        3) infra가 예측한 pred_boxes도 vehicle 좌표계로 회전/이동 변환한다.

        Args:
            instances (dict):
                infra 모델이 뱉어온 query 상태를 dict로 전달받는 형태
                주요 key:
                - 'ref_pts'           : [N, 3]  (infra pc_range 기준 normalized)
                - 'query_feats'       : [N, C]
                - 'query_embeds'      : [N, C]
                - 'cache_motion_feats': [N, C]
                - 'pred_boxes'        : [N, box_dim] (infra 기준 normalized bbox)

            veh2inf_rt (Tensor):
                - shape: [4, 4] (혹은 [3,4] 기반 확장)
                - vehicle → infra 변환 행렬 (캘리브레이션)
                - 즉 vehicle 좌표의 점을 infra 좌표로 옮기는 변환

        Returns:
            instances (dict):
                - vehicle 좌표계로 정렬된 instances 반환
                - ref_pts 및 feature/bbox가 변환 완료된 결과
        """

        # ============================================================
        # (1) 차량→인프라(veh2inf) 변환을 인프라→차량(inf2veh)로 뒤집기
        # ============================================================
        # veh2inf_rt는 vehicle 좌표 -> infra 좌표로 변환하는 행렬
        # 그런데 우리는 infra 결과(instances)를 vehicle 좌표계로 가져와야 하므로,
        # inverse를 통해 inf2veh로 변환해야 함.
        #
        # ⚠️ 여기서 특이한 점:
        #   np.linalg.inv(veh2inf_rt.cpu().numpy().T)
        #   => 행렬을 transpose한 다음 inverse를 취함
        #   이는 veh2inf_rt 저장 형태가 일반적인 (4x4)와 축 방향이 다르거나,
        #   row-major/column-major 적용 방식이 달라서 보정해준 것으로 보임
        calib_inf2veh = np.linalg.inv(veh2inf_rt.cpu().numpy().T)  # veh2inf => inf2veh로 변환
        calib_inf2veh = instances['ref_pts'].new_tensor(calib_inf2veh)  # 같은 device/dtype으로 Tensor 변환

        # ============================================================
        # (2) 변환 행렬에서 rotation / translation 분리
        # ============================================================
        # calib_inf2veh 형태 (4x4)라 가정하면:
        #   [ R(3x3)  t(3x1) ]
        #   [ 0 0 0    1    ]
        rot = calib_inf2veh[:3, :3].clone()   # [3,3] 회전 행렬
        trans = calib_inf2veh[:3, 3:4].clone()  # [3,1] translation 벡터

        # ============================================================
        # (3) rotation을 "6D representation"으로 변환
        # ============================================================
        # 일반적으로 rotation을 6D로 표현하는 이유:
        # - Euler angle은 discontinuity 문제가 있고
        # - quaternion도 sign ambiguity 문제가 있음
        # - 6D(rot matrix의 앞 2 column 등)는 연속적인 표현이라 학습에 유리함
        #
        # 여기서는 continuous_rot()에서 rot[:, :2]만 뽑아서 6차원으로 만듦
        # rot[:, :2] shape = [3,2] -> flatten -> [1,6]
        if self.rot_dims == 6:
            con_rot = self.continuous_rot(rot)  # [1,6]
            assert con_rot.size(1) == 6

        # translation도 MLP 입력 형태로 flatten
        # trans: [3,1] -> [1,3]
        trans = trans.reshape(1, -1)

        # ============================================================
        # (4) rot/trans를 MLP로 latent space 파라미터로 변환
        # ============================================================
        # rot_mlp 출력: rot_para
        # - rot_final_dim = (embed/head)^2 * head
        #   예) embed=256, head=16 -> (16^2)*16 = 4096
        #
        # trans_mlp 출력: trans_para
        # - trans_final_dim = embed_dims = 256
        #
        # 즉, rotation은 "head별 block diagonal rotation matrix"를 만들 재료로,
        # translation은 "feature 공간에서 더해줄 256차원 bias"로 사용된다.
        rot_para = self.rot_mlp(con_rot)     # [1, 4096] 같은 형태
        trans_para = self.trans_mlp(trans)   # [1, 256]

        # ============================================================
        # (5) rot_para(4096)를 실제 256x256 rotation matrix로 채우기
        # ============================================================
        # fill_tensor는 아래 성질을 가진 rotation matrix를 만든다:
        # - full (256x256) dense matrix가 아니라,
        # - head 단위로 block-diagonal 형태로 채움
        #
        # 예) head=16, embed=256 => head당 16차원 block
        # 256x256에서 16개 block만 채워지는 구조
        #
        # 최종 rot_mat: [256, 256]
        rot_mat = self.fill_tensor(rot_para)

        # ============================================================
        # (6) reference point를 infra 좌표계 -> vehicle 좌표계로 변환
        # ============================================================
        # instances['ref_pts']는 현재 infra pc_range 기준 normalized 좌표 (0~1)
        #
        # transform_pts() 내부 과정:
        #   (1) infra normalized -> infra absolute 좌표(meter)로 denormalize
        #   (2) calib_inf2veh로 4x4 homogeneous 변환 적용 (inf -> veh)
        #   (3) vehicle pc_range 밖으로 나가는 점은 mask로 제거
        #   (4) vehicle absolute -> vehicle normalized(0~1)로 normalize
        #
        # 반환:
        #   - 변환된 ref_pts : [N', 3] (vehicle 기준 normalized)
        #   - mask           : [N] boolean mask (유효한 query만 True)
        instances['ref_pts'], mask = self.transform_pts(instances['ref_pts'], calib_inf2veh)

        # ============================================================
        # (7) mask 기반으로 query 관련 모든 필드도 동일하게 필터링
        # ============================================================
        # ref_pts에서 제거된 query는 feature들도 함께 제거해야 "index 정합"이 맞음
        # 예: 특정 query가 ROI 밖으로 나가서 ref_pts가 제거되었는데,
        #     query_feats가 그대로 남으면 query index가 어긋나 버림
        instances['query_feats'] = instances['query_feats'][mask]
        instances['query_embeds'] = instances['query_embeds'][mask]
        instances['cache_motion_feats'] = instances['cache_motion_feats'][mask]
        instances['pred_boxes'] = instances['pred_boxes'][mask]

        # ============================================================
        # (8) residual 연결을 위한 identity feature 백업
        # ============================================================
        # feature alignment를 적용한 뒤에도 원본 특징을 일부 유지하도록
        # residual 형태로 identity를 더한다.
        identity_query_feats = instances['query_feats'].clone()
        identity_query_embeds = instances['query_embeds'].clone()
        identity_cache_motion_feats = instances['cache_motion_feats'].clone()

        # ============================================================
        # (9) feature 공간에서 rotation + translation 변환 적용 (latent alignment)
        # ============================================================
        # 왜 feature에도 rotation/translation을 적용하나?
        # - ref_pts만 좌표계 변환하면 "위치"는 맞지만
        #   infra query feature는 infra 관측(센서 세팅/좌표계)에 맞춰 학습된 특징임
        # - vehicle feature와 결합하려면 latent feature도
        #   좌표계 변화에 맞춰 정렬되어야 cross-agent fusion이 잘 된다.
        #
        # 적용 방식:
        #   input_proj(feature) @ rot_mat.T + trans_para
        #   -> 좌표계 변환을 feature 공간의 선형 변환으로 근사
        #   + identity residual로 안정화
        #   -> output_proj로 다시 한 번 정리(projection)

        # ---- (9-A) query_feats 정렬 ----
        # feat_input_proj: [N',256] -> [N',256] (전처리)
        # @ rot_mat.T     : [N',256] @ [256,256] -> [N',256]
        # + trans_para    : [1,256] broadcast -> [N',256]
        # + residual      : + identity_query_feats
        # feat_output_proj: 후처리 projection
        instances['query_feats'] = self.feat_output_proj(
            (self.feat_input_proj(instances['query_feats']) @ rot_mat.T + trans_para) + identity_query_feats
        )

        # ---- (9-B) query_embeds 정렬 ----
        # query_embeds는 positional embedding 계열이지만,
        # infra 기준 pos embedding을 vehicle feature와 더 잘 맞추도록 동일하게 정렬
        instances['query_embeds'] = self.embed_output_proj(
            (self.embed_input_proj(instances['query_embeds']) @ rot_mat.T + trans_para) + identity_query_embeds
        )

        # ---- (9-C) cache_motion_feats 정렬 ----
        # motion embedding도 좌표계 영향(특히 rot/trans)에 민감하므로 동일하게 변환
        instances['cache_motion_feats'] = self.motion_output_proj(
            (self.motion_input_proj(instances['cache_motion_feats']) @ rot_mat.T + trans_para) + identity_cache_motion_feats
        )

        # ============================================================
        # (10) pred_boxes도 실제 box 변환 수행 (inf -> veh)
        # ============================================================
        # transform_boxes 내부 과정:
        #   - pred_boxes(infra normalized)를 infra absolute로 denormalize
        #   - LiDARInstance3DBoxes로 만들어 rotate/translate 수행
        #   - vehicle pc_range 기준으로 다시 normalize
        #
        # 주의:
        # - 여기서는 rotation/translation을 정확한 기하학 변환으로 적용하는 반면,
        # - feature 변환은 "학습된 latent linear 변환"으로 근사하는 구조임
        instances['pred_boxes'] = self.transform_boxes(instances['pred_boxes'], calib_inf2veh)

        # ============================================================
        # (11) 변환 완료된 instances 반환
        # ============================================================
        return instances


        