# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
from .track_instance import Instances
import torch
import numpy as np


class RunTimeTracker:
    """
    [RunTimeTracker]
    - tracking 과정에서 track query(Instances)들의 "생존 여부(live)"를 관리하는 모듈
    - 매 프레임마다 현재 score/active 여부를 보고,
      1) 활성(active) track은 유지
      2) 비활성인데 과거에 track으로 한번 등록(track_query_mask=True)된 track은
         max_age_since_update 동안만 "죽이지 않고" 잠깐 유지 (temporal continuity 위해)

    핵심 역할:
      ✅ track id 부여(current_id 증가)
      ✅ active query mask 기반으로 disappear_time 업데이트
      ✅ disappear_time이 너무 크면 query를 제거(drop)

    Instances 주요 필드와 연결:
      - obj_idxes         : [N] 각 query의 track ID (-1이면 아직 미부여)
      - track_query_mask  : [N] True면 "한번 track으로 활성화된 적 있음"
      - disappear_time    : [N] 최근 업데이트가 안 된 프레임 수
      - scores            : [N] confidence score
      - matched_gt_idxes  : [N] GT 매칭 index (학습에서 active 판정 기준에 사용)
      - iou               : [N] GT와 IoU (학습에서 active 판정에 사용)
    """

    def __init__(self,
                 output_threshold=0.2,
                 score_threshold=0.4,
                 record_threshold=0.4,
                 max_age_since_update=1):
        """
        Args:
            output_threshold (float):
                - 최종 결과로 "출력(output)"할 때 사용할 score 임계값
                - inference에서 최종 bbox 출력 필터링에 주로 쓰임

            score_threshold (float):
                - "active object"라고 판단하는 기준 임계값 (tracking 유지 여부에 영향)
                - inference에서 active_mask 만들 때 사용되는 경우가 많음

            record_threshold (float):
                - tracking 모드에서 "기록(record)"할 query를 결정하는 임계값
                - 예: frame_summarization(tracking=True)에서 active_mask를 만들 때 사용

            max_age_since_update (int):
                - active가 아니더라도 "바로 제거하지 않고 유지할 프레임 수"
                - disappear_time이 이 값 이상이면 제거
                - 즉, occlusion(잠깐 가림) 복원을 위한 grace period
        """

        # 새로운 track id를 부여하기 위한 카운터
        # 새로운 객체가 등장할 때 1,2,3,... 이런 식으로 증가
        self.current_id = 1

        # tracking 판단에 쓰는 threshold들 저장
        self.threshold = score_threshold
        self.output_threshold = output_threshold
        self.record_threshold = record_threshold

        # "몇 프레임까지 업데이트 없어도 살아있게 둘지"
        self.max_age_since_update = max_age_since_update

    def update_active_tracks(self, track_instances, active_mask):
        """
        [update_active_tracks]
        - 현재 프레임에서 "살아남을 track query"를 결정해서 Instances를 필터링해서 반환

        Inputs:
            track_instances : Instances (길이 N)
            active_mask     : torch.bool tensor, shape [N]
                - active_mask[i] = True  -> i번째 query는 이번 프레임에서 확실히 active
                - active_mask[i] = False -> i번째 query는 이번 프레임에서 비활성

        동작:
            1) active query는 disappear_time=0으로 초기화하고 유지
            2) active가 아니더라도 과거에 track으로 등록된(track_query_mask=True) query는
               disappear_time을 +1 증가시키고,
               max_age_since_update 미만이면 임시로 유지
            3) 그 외는 제거

        Returns:
            track_instances[live_mask]
                - live_mask=True인 query만 남긴 Instances
                - 즉 다음 프레임으로 전달될 track 후보들
        """

        # live_mask: "최종적으로 살아남을 query mask"
        # track_instances.obj_idxes와 동일한 shape의 bool mask 생성
        # detach()는 불필요한 그래프 연결 방지(여기서는 tracking/lifecycle 용이므로 gradient 필요 없음)
        live_mask = torch.zeros_like(track_instances.obj_idxes).bool().detach()

        # 모든 query를 순회하면서 살릴지 죽일지 결정
        for i in range(len(track_instances)):

            # ------------------------------------------------------
            # (1) 이번 프레임에서 active라면 무조건 살림
            # ------------------------------------------------------
            if active_mask[i]:
                # "이번 프레임에서 업데이트됨" -> disappear_time 초기화
                track_instances.disappear_time[i] = 0
                live_mask[i] = True

            # ------------------------------------------------------
            # (2) 이번 프레임에서 active는 아니지만,
            #     과거에 track으로 등록된 적이 있는 query(track_query_mask=True)라면
            #     잠깐 유지할 기회를 줌 (occlusion 대비)
            # ------------------------------------------------------
            elif track_instances.track_query_mask[i]:
                # 업데이트가 없었으니 disappear_time 증가
                track_instances.disappear_time[i] += 1

                # 아직 grace period(max_age_since_update) 안이면 살림
                # 즉, 잠깐 detection이 안 된 경우에도 track이 끊기지 않게 함
                if track_instances.disappear_time[i] < self.max_age_since_update:
                    live_mask[i] = True

            # ------------------------------------------------------
            # (3) active도 아니고 track으로 등록된 적도 없으면 제거
            # ------------------------------------------------------
            # else:
            #   live_mask[i] stays False

        # live_mask로 필터링해서 다음 프레임에 들고갈 track_instances 반환
        return track_instances[live_mask]

    def get_active_mask(self, track_instances, training=True):
        """
        [get_active_mask]
        - 현재 프레임에서 "active track"을 정의하는 기준을 만드는 함수

        training=True일 때:
            active_mask = (matched_gt_idxes >= 0) AND (iou > 0.5)

        의미:
            - matched_gt_idxes >= 0 :
                i번째 query가 GT와 Hungarian matching 등으로 실제 GT 객체에 매칭되었다는 뜻
                (즉, i번째 query는 "진짜 객체와 대응되는 query"라는 의미)

            - iou > 0.5 :
                매칭된 GT와 bbox가 충분히 잘 맞는지 체크하는 조건
                (학습에서 질 낮은 query까지 active로 취급하지 않게 하기 위함)

        Outputs:
            active_mask: torch.bool tensor shape [N]
        """

        if training:
            # matched_gt_idxes: [N]
            # iou            : [N]
            # -> 둘 다 만족하는 query만 active로 본다.
            active_mask = (track_instances.matched_gt_idxes >= 0) & (track_instances.iou > 0.5)

            # 실험적으로 iou 조건 없이 "매칭만 되면 active"로 할 수도 있음
            # active_mask = (track_instances.matched_gt_idxes >= 0)

        return active_mask

    def empty(self):
        """
        [empty]
        - tracker 상태 초기화 함수

        주로 새로운 scene 시작 시 또는 evaluation reset 시 호출
        - current_id를 1로 다시 초기화
        """
        self.current_id = 1

