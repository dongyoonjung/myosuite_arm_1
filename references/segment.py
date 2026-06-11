"""단조 거상 RISE 구간(저점→첫 정점) 분절.

★ LOCKED: 상승 구간만 쓴다. KIMHu 각 반복은 복합시퀀스(외전→팔꿈치 머리로→신전→손목…)라
정점 뒤 고원(plateau)에서 손목조작이 섞여 elbow/pro_sup 채널을 오염시킨다. 상승 구간 자체는
20명 전원 단조(반전 0)임을 실측 확인. (DESIGN.md 'D 참조 구성' 참조.)
"""
import numpy as np
from scipy.signal import find_peaks


def rise_segments(elevation, min_peak=55.0, min_rise=30.0, distance=60, prominence=25):
    """elevation(deg, 시계열) → [(lo, peak), ...] 상승 구간 인덱스 쌍.

    각 거상 반복의 정점을 찾고, 그 직전 저점까지 거슬러 올라가 상승만 잘라낸다.
    고원/하강은 포함하지 않는다.
    """
    pks, _ = find_peaks(elevation, height=min_peak, distance=distance,
                        prominence=prominence)
    segs = []
    for p in pks:
        lo = p
        thr = max(20.0, 0.3 * elevation[p])
        while lo > 0 and elevation[lo] > thr:
            lo -= 1
        if p - lo < 10:                       # 너무 짧음
            continue
        if elevation[p] - elevation[lo] < min_rise:
            continue
        segs.append((lo, p))
    return segs


def reversals(seg_curve, smooth=5):
    """상승 구간 단조성 점검: 부호 반전 횟수(0이 이상적).

    원시 30fps는 프레임 지터(~0.2°)로 미세 반전이 많으니, 짧은 이동평균으로
    평활한 뒤 '진짜' 비단조만 센다(거시적 단조성 점검).
    """
    x = seg_curve
    if smooth and len(x) > smooth:
        x = np.convolve(x, np.ones(smooth) / smooth, mode="valid")
    d = np.diff(x)
    s = np.sign(d)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0
