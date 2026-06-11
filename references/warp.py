"""런타임 참조 생성기 — 정규 템플릿을 latent(T·peak·skew·plane)로 워핑.

M1+ 환경이 매 reset마다 호출: sample_latent()로 궤적인자 뽑고 make_reference()로
해당 에피소드의 참조 궤적(각 채널, deg)을 만든다. 인자값은 정책 obs + 회귀기 known
공변량으로도 쓰인다(A1 LOCKED). 학습 인코더 불필요.

★ 반환은 해부학적 도(deg). myoArm qpos(rad)로의 매핑(부호/오프셋, thoracohumeral→
shoulder_elv 1:1, elv_angle 규약)은 환경 통합(M1) 단계에서 보정한다.
★ shoulder_rot은 참조 없음(자유 DoF) → 여기서 만들지 않는다.
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out")
_CACHE = {}


def load_template(task):
    if task not in _CACHE:
        d = dict(np.load(os.path.join(OUT, f"template_{task}.npz")))
        with open(os.path.join(OUT, f"latent_{task}.json")) as fh:
            d["_latent"] = json.load(fh)
        _CACHE[task] = d
    return _CACHE[task]


def sample_latent(task, rng, jitter=0.1):
    """경험적 표본 한 개를 뽑아(상관 보존) 소폭 jitter. dict(T,peak,skew,plane,elbow_peak)."""
    t = load_template(task)
    i = rng.integers(len(t["s_T"]))
    def j(arr):
        return float(arr[i] + rng.normal(0, jitter * (np.std(arr) + 1e-9)))
    return {"T": max(0.5, j(t["s_T"])), "peak": float(np.clip(j(t["s_peak"]), 30, 150)),
            "skew": j(t["s_skew"]), "plane": j(t["s_plane"]),
            "elbow_peak": float(np.clip(j(t["s_elbow_peak"]), 20, 150))}


def make_reference(task, latent, n_steps, dt):
    """latent → 에피소드 참조 궤적(deg). 상승 후 정점 유지(raise-and-hold).

    반환 dict (각 (n_steps,) deg):
      shoulder_elv, elv_angle, elbow_flexion, pro_sup
    """
    t = load_template(task)
    N = int(t["N"])
    ph_grid = np.linspace(0, 1, N)
    # 거상 형태: 평균 + skew·PC1 → 단조화 → [0,1] 정규화 → 정점진폭 스케일
    shape = t["elev_mean_norm"] + latent["skew"] * t["elev_pc1"]
    shape = np.maximum.accumulate(np.clip(shape, 0.0, 1.2))   # 단조 상승 보장
    shape = (shape - shape[0]) / (shape[-1] - shape[0] + 1e-9)
    elev_phase = float(latent["peak"]) * shape

    eb = t["elbow_shape"]
    elbow_phase = eb * (float(latent["elbow_peak"]) / (eb.max() + 1e-9))
    prosup_phase = t["prosup_shape"]
    # 평면 궤적(T1≈관상 유지 / T2=전방 스윕 0→~45°) + 피험자별 오프셋
    pshape = t["plane_shape"]
    plane_phase = pshape + (float(latent["plane"]) - float(pshape.mean()))

    # 시간축: 동작은 T초에 걸쳐, 이후 위상 1에서 자세 유지
    tt = np.arange(n_steps) * dt
    ph_t = np.clip(tt / max(1e-3, float(latent["T"])), 0.0, 1.0)

    def at(phase_curve):
        return np.interp(ph_t, ph_grid, phase_curve)

    return {
        "shoulder_elv": at(elev_phase),
        "elv_angle": at(plane_phase),          # T2=전방 스윕(자유드리프트는 정책 몫)
        "elbow_flexion": at(elbow_phase),
        "pro_sup": at(prosup_phase),
    }


def _selftest():
    rng = np.random.default_rng(0)
    for task in ("T1", "T2"):
        lat = sample_latent(task, rng)
        ref = make_reference(task, lat, n_steps=100, dt=0.02)
        elv = ref["shoulder_elv"]
        pl = ref["elv_angle"]
        assert np.all(np.diff(elv) >= -1e-6), "거상 참조가 단조가 아님"
        print(f"[{task}] latent T={lat['T']:.2f} peak={lat['peak']:.0f} "
              f"skew={lat['skew']:+.2f} plane={lat['plane']:.0f} | "
              f"ref elv {elv[0]:.0f}→{elv.max():.0f}° "
              f"plane {pl[0]:+.0f}→{pl[-1]:+.0f}° elbow {ref['elbow_flexion'].max():.0f}° "
              f"prosup Δ{ref['pro_sup'].max()-ref['pro_sup'].min():.0f}°")


if __name__ == "__main__":
    _selftest()
