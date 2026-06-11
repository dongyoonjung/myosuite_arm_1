"""KIMHu 골격 로딩 + 3D 위치에서 유도한 관절 각도.

★ LOCKED: 모든 참조 각도는 **3D Joints.Position에서만** 유도한다. 사전계산 각도열
(col2-7, 예: AngleShoulderRight)은 절대 쓰지 않는다 — col4는 흉곽-상완 거상과 정의가
다르고 ~+90° 오프셋(상관 0.97이나 크기 틀림)이라 베이스라인을 조용히 오염시킨다.
검증: 유도 거상은 ~6-110°, col4는 113-180°. (DESIGN.md, dont-invent-beyond-data 참조.)

채널 신뢰도(실측):
 - elevation: ShoulderRight 20명 전원 100% tracked → 깨끗.
 - pro_sup: 위치기하(엄지/손끝 palm-normal)·쿼터니언 두 방법 일치(corr +0.56~+0.77) → 신뢰.
 - shoulder_rot: 두 방법 불일치 → 참조 안 만듦(정책 자유 DoF).
"""
import csv
import glob
import json
import os

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "kimhu", "V2")
FPS = 30.0  # Kinect V2 (실측 effective 29.91)

# 유도 거상 정합성 가드: 이 범위를 크게 벗어나면 col4를 잘못 집은 것.
ELEV_SANITY = (-10.0, 160.0)


def subject_files(task):
    """task in {'T1','T2'} → 정렬된 skeleton CSV 경로 리스트(피험자별 1개)."""
    pat = os.path.join(DATA_ROOT, f"* {task}", f"*_{task}_skeleton_tracking.csv")
    return sorted(glob.glob(pat))


def load_frames(path):
    """skeleton CSV(세미콜론 구분, col10=BodyInfoJson) → 프레임 dict 리스트."""
    frames = []
    with open(path) as fh:
        rdr = csv.reader(fh, delimiter=";")
        next(rdr)  # 헤더
        for row in rdr:
            if len(row) >= 10 and row[9].strip().startswith("{"):
                try:
                    frames.append(json.loads(row[9]))
                except json.JSONDecodeError:
                    pass
    return frames


def _pos(frames, name):
    return np.array(
        [[f["Joints"][name]["Position"][k] for k in "XYZ"] for f in frames]
    )


def _quat(frames, name):
    return np.array(
        [[f["JointOrientations"][name]["Orientation"][k] for k in "XYZW"] for f in frames]
    )


def _track(frames, name):
    return np.array([f["Joints"][name]["TrackingState"] for f in frames])


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _roll_about(axis, vec, ref):
    """vec를 axis 둘레로 굴린 각도(ref를 0으로). pro_sup·평면 회전용. unwrap된 deg."""
    u = _unit(axis)
    vp = _unit(vec - np.sum(vec * u, 1, keepdims=True) * u)
    rp = _unit(ref - np.sum(ref * u, 1, keepdims=True) * u)
    s = np.sum(np.cross(rp, vp) * u, 1)
    c = np.sum(rp * vp, 1)
    return np.degrees(np.unwrap(np.arctan2(s, c)))


def derive_channels(frames):
    """프레임 → 각 채널(deg) 시계열. 전부 3D 위치 기반.

    elevation   : 흉곽-상완 거상각(ShoulderR→ElbowR 벡터 vs 체간 아래). myoArm
                  shoulder_elv에 1:1 매핑(견갑 rhythm은 모델이 자동 부가).
    elbow_flex  : 팔꿈치 굴곡각.
    pro_sup     : 전완 회내/회외(손바닥 법선의 전완축 둘레 회전; 엄지+손끝).
    plane_az    : 거상 평면 방위각(0=관상/측면, 90=시상/정면).
    track_shoulder: ShoulderRight TrackingState(2=tracked).
    """
    sho = _pos(frames, "ShoulderRight")
    elb = _pos(frames, "ElbowRight")
    wri = _pos(frames, "WristRight")
    tip = _pos(frames, "HandTipRight")
    thb = _pos(frames, "ThumbRight")
    spB = _pos(frames, "SpineBase")
    spS = _pos(frames, "SpineShoulder")
    shoL = _pos(frames, "ShoulderLeft")

    ua = elb - sho          # 위팔
    fa = wri - elb          # 전완
    down = spB - spS        # 체간 아래

    elevation = np.degrees(
        np.arccos(np.clip(np.sum(_unit(ua) * _unit(down), 1), -1, 1))
    )

    v1 = sho - elb
    v2 = wri - elb
    elbow_flex = 180 - np.degrees(
        np.arccos(np.clip(np.sum(_unit(v1) * _unit(v2), 1), -1, 1))
    )

    pn = np.cross(tip - wri, thb - wri)   # 손바닥 법선
    pro_sup = _roll_about(fa, pn, down)

    # 평면 방위각: 위팔 수평성분을 (측면, 전방) 기저로. 0=관상(측면), +=전방(시상쪽).
    # ※ 팔이 수직에 가까우면 수평성분이 작아 방위각이 노이즈 → plane_mag로 가중해 쓸 것.
    d = _unit(down)
    lateral = _unit(sho - shoL)                       # 오른쪽(R-L)
    fwd = _unit(np.cross(d, lateral))                 # 전방축
    ua_h = ua - np.sum(ua * d, 1, keepdims=True) * d
    plane_mag = np.linalg.norm(ua_h, axis=1)          # 수평성분 크기(신뢰 가중)
    plane_az = np.degrees(np.arctan2(-np.sum(ua_h * fwd, 1),   # 전방을 +로
                                     np.sum(ua_h * lateral, 1)))

    lo, hi = elevation.min(), elevation.max()
    if not (ELEV_SANITY[0] <= lo and hi <= ELEV_SANITY[1]):
        raise ValueError(
            f"유도 거상 범위 [{lo:.0f},{hi:.0f}]가 비정상 — col2-7을 잘못 집었거나 "
            f"좌표 오류. 3D Position에서만 유도해야 함.")

    return {
        "elevation": elevation,
        "elbow_flex": elbow_flex,
        "pro_sup": pro_sup,
        "plane_az": plane_az,
        "plane_mag": plane_mag,
        "track_shoulder": _track(frames, "ShoulderRight"),
    }


def weighted_circmean(az_deg, weights):
    """크기 가중 원형평균(deg). 수평성분 작은(수직팔) 노이즈 프레임을 자연 하향가중."""
    ar = np.radians(az_deg)
    c = np.sum(weights * np.cos(ar))
    s = np.sum(weights * np.sin(ar))
    return float(np.degrees(np.arctan2(s, c)))
