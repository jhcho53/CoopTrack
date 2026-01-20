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
        """
        [Block-Diagonal Matrix 생성 함수]
        ------------------------------------------------------------
        목적:
        - original_tensor에 들어있는 값을 이용해서 (d x d) 행렬을 만든다.
        - 단, 전체를 채우는 게 아니라 "head별로 분리된 block" 형태로 채운다.
            즉, Multi-Head Attention 처럼
            head0: [k x k]
            head1: [k x k]
            ...
            head(h-1): [k x k]
            이런 블록들이 대각선에 배치된 block-diagonal 행렬을 만든다.

        예시:
        embed_dims = d = 256
        head = h = 16
        그러면 head당 차원 k = d / h = 16

        최종 행렬은 [256 x 256]이고,
        대각선에 16개의 [16 x 16] 블록이 들어감.
        """

        # head 개수 (예: 16)
        h = self.head

        # head당 embedding 차원 (예: 256//16 = 16)
        k = self.embed_dims // self.head

        # 전체 embedding 차원 = 최종 행렬 크기 (예: 256)
        d = self.embed_dims

        # ------------------------------------------------------------
        # 1) 각 head 블록이 시작하는 "행/열 시작 index" 만들기
        # ------------------------------------------------------------
        """
        blocks_indices: [h] 크기의 텐서
        head 0 블록 시작 index = 0*k
        head 1 블록 시작 index = 1*k
        head 2 블록 시작 index = 2*k
        ...
        head (h-1) 블록 시작 index = (h-1)*k
        
        예: k=16이면
        blocks_indices = [0, 16, 32, 48, ..., 240]
        """
        blocks_indices = torch.arange(h, device=original_tensor.device) * k

        # ------------------------------------------------------------
        # 2) 블록 내부에서 사용할 offset 좌표 생성 (0~k-1)
        # ------------------------------------------------------------
        """
        offset: [k]
        0, 1, 2, ..., k-1
        
        블록 크기가 kxk니까
        row offset / col offset을 만들기 위해 사용
        """
        offset = torch.arange(k, device=original_tensor.device)

        """
        rows_offset, cols_offset: 각각 [k, k]
        - torch.meshgrid(offset, offset)은
        (i,j) 격자 인덱스를 만들어줘서
        kxk 좌표 전체를 구성할 수 있음

        예: k=3이면
        rows_offset =
            [[0,0,0],
            [1,1,1],
            [2,2,2]]
        cols_offset =
            [[0,1,2],
            [0,1,2],
            [0,1,2]]
        """
        rows_offset, cols_offset = torch.meshgrid(offset, offset)

        # ------------------------------------------------------------
        # 3) head별로 블록의 "전체 행/열 좌표" 만들기
        # ------------------------------------------------------------
        """
        base_rows: [h,1,1]
        - 각 head의 시작 row index를 (브로드캐스트 가능하게) reshape한 것

        예: blocks_indices가 [0,16,32,...]라면
        base_rows =
            [[[0]],
            [[16]],
            [[32]],
            ...
            ]
        """
        base_rows = blocks_indices.view(h, 1, 1)

        """
        global_rows: [h, k, k]
        - 각 head 블록에 대해,
        row index를 "전체 행렬 좌표계"로 확장한 것

        head t에 대해:
        global_rows[t] = base_rows[t] + rows_offset

        즉, head별 블록이 들어갈 row 좌표들.
        """
        global_rows = base_rows + rows_offset.unsqueeze(0)

        """
        base_cols도 row와 동일한 방식으로
        각 head의 시작 col index를 만들고,
        global_cols는 전체 col 좌표를 head별로 생성한다.
        """
        base_cols = blocks_indices.view(h, 1, 1)
        global_cols = base_cols + cols_offset.unsqueeze(0)

        # ------------------------------------------------------------
        # 4) head별 block 좌표들을 1차원 index 리스트로 펼치기
        # ------------------------------------------------------------
        """
        all_rows: [h*k*k]
        all_cols: [h*k*k]
        - (head, block_row, block_col)을 하나의 리스트로 flatten한 형태
        - 이 (all_rows[n], all_cols[n]) 위치에 data[n] 값을 넣을 예정
        """
        all_rows = global_rows.reshape(-1)
        all_cols = global_cols.reshape(-1)

        # ------------------------------------------------------------
        # 5) original_tensor를 flatten해서 data로 사용
        # ------------------------------------------------------------
        """
        original_tensor는 보통 [h*k*k] 또는 [h, k, k] 형태의 값이 들어있다고 가정
        예: rot_mlp 출력이 rot_final_dim = h*k*k = 4096

        data: [h*k*k]
        - 이 값들이 head별 kxk 블록에 순서대로 들어가게 됨
        """
        data = original_tensor.view(-1)

        # ------------------------------------------------------------
        # 6) 최종 (d x d) 행렬을 만들고 block-diagonal 위치에 값 채우기
        # ------------------------------------------------------------
        """
        target_tensor: [d, d]
        - 전체는 0으로 초기화
        - 이후 block-diagonal 위치에만 data 값이 들어감
        """
        target_tensor = torch.zeros((d, d), device=original_tensor.device)

        """
        핵심!
        target_tensor[all_rows, all_cols] = data

        즉,
        head0 블록 영역 (0~k-1, 0~k-1)에 data 일부가 채워지고
        head1 블록 영역 (k~2k-1, k~2k-1)에 data 다음 일부가 채워지고
        ...
        나머지 영역은 전부 0 (블록 밖은 연결 없음)
        """
        target_tensor[all_rows, all_cols] = data

        # block-diagonal matrix 반환
        return target_tensor

    def transform_pts(self, points, transformation):
        """
        [transform_pts]
        -------------------------------------------------------------
        목적:
        - 인프라(infra) 관측 결과의 reference point(points)를
            차량(vehicle) 좌표계 기준 reference point로 변환하는 함수

        입력:
        points : [N, 3]
            - infra pc_range 기준으로 normalize된 좌표 (0~1 범위)
            - 각 row는 하나의 query reference point
            - points[:, 0] = x (normalized)
            - points[:, 1] = y (normalized)
            - points[:, 2] = z (normalized)

        transformation : [4, 4]
            - inf → veh 좌표계 변환 행렬 (homogeneous transform)
            - 즉, infra world 좌표를 vehicle world 좌표로 변환하는 행렬

        출력:
        locs : [M, 3]
            - vehicle pc_range 기준으로 normalize된 좌표 (0~1 범위)
            - 단, pc_range 밖은 제거되므로 M ≤ N

        mask : [N]
            - True인 query만 살아남음 (vehicle 범위 내부)
            - 이후 query_feats/query_embeds/pred_boxes 등을 동일하게 필터링할 때 사용
        """

        # =========================================================
        # 1) infra normalized 좌표 → infra world(절대좌표)로 복원(denormalize)
        # =========================================================
        # points는 [0~1] 스케일이므로 실제 공간 단위(meter 등)로 복원해야
        # 4x4 rigid transformation(회전/이동)을 적용할 수 있음
        locs = points.clone()

        # x 복원:
        #   x_abs = x_norm * (x_max - x_min) + x_min
        # index 설명:
        #   locs[:, 0:1] => [N, 1] 형태 유지 (broadcast 안정적)
        locs[:, 0:1] = (
            locs[:, 0:1] * (self.inf_pc_range[3] - self.inf_pc_range[0]) + self.inf_pc_range[0]
        )

        # y 복원
        locs[:, 1:2] = (
            locs[:, 1:2] * (self.inf_pc_range[4] - self.inf_pc_range[1]) + self.inf_pc_range[1]
        )

        # z 복원
        locs[:, 2:3] = (
            locs[:, 2:3] * (self.inf_pc_range[5] - self.inf_pc_range[2]) + self.inf_pc_range[2]
        )

        # =========================================================
        # 2) 4x4 homogeneous transformation 적용 (inf world → veh world)
        # =========================================================
        # 4x4 변환행렬을 적용하려면 [x, y, z]를 [x, y, z, 1]로 확장해야 함
        # torch.ones_like(locs[..., :1]) => [N, 1] shape의 1벡터 생성
        # torch.cat(...) => [N, 4] (homogeneous coordinate)
        # unsqueeze(-1) => [N, 4, 1] (행렬곱을 위해 column vector로 변경)
        locs = torch.cat((locs, torch.ones_like(locs[..., :1])), -1).unsqueeze(-1)

        # transformation: [4,4], locs: [N,4,1]
        # 결과: [N,4,1] (각 point에 동일한 rigid transform 적용)
        locs = torch.matmul(transformation, locs)

        # squeeze(-1) => [N,4]
        # [..., :3]  => [N,3] (x,y,z만 사용하고 homogeneous w는 버림)
        locs = locs.squeeze(-1)[..., :3]

        # =========================================================
        # 3) vehicle pc_range 밖의 point 제거 (filtering)
        # =========================================================
        # 이유:
        #   infra에서 본 객체가 vehicle의 관심영역(pc_range)에 없을 수 있음
        #   vehicle 범위 밖 query는 이후 fusion/association에서 방해가 되므로 제거
        #
        # 여기서는 x,y에 대해서만 범위 체크함
        # (보통 BEV 기반이므로 xy만 컷해도 충분한 경우가 많음)
        mask = (
            (self.pc_range[0] <= locs[:, 0]) & (locs[:, 0] <= self.pc_range[3]) &
            (self.pc_range[1] <= locs[:, 1]) & (locs[:, 1] <= self.pc_range[4])
        )

        # mask가 True인 point만 유지
        # locs shape이 [N,3] → [M,3] 로 줄어듦
        locs = locs[mask]

        # =========================================================
        # 4) vehicle world 좌표 → vehicle normalized 좌표로 변환(normalize)
        # =========================================================
        # 이후 vehicle model 내부에서는 ref_pts를 normalize된 형태(0~1)로 쓰는 경우가 많음
        # 따라서 vehicle pc_range 기준으로 다시 normalize 해서 반환
        #
        # x_norm = (x_abs - x_min) / (x_max - x_min)
        locs[..., 0:1] = (locs[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        # locs : [M,3] (veh normalized ref_pts)
        # mask : [N]  (살아남은 query 표시)
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


        