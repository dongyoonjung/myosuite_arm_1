"""M0 메인: KIMHu T1/T2 → 정규 거상 템플릿 + latent(T·peak·skew·plane) 분포.

파이프라인(전부 LOCKED 결정 준수):
 1. 3D 위치 유도(kimhu_io) → 단조 상승 분절(segment) — col2-7 사전계산각 안 씀, 상승만.
 2. 반복마다 latent 인자 추출: T(이동시간), peak(거상정점), plane(평면방위), elbow_peak.
 3. 시간정규화(위상[0,1] 리샘플) + 진폭정규화(형태). 이상치 트림(MAD + 하드 플로어).
 4. 로버스트 평균 템플릿 + 형태 PC1(=skew). elbow/pro_sup 평균형태.
 5. references/out/ 에 npz(템플릿)+json(latent 분포) 저장 + 검증 플롯.

실행:  python -m references.build_templates
"""
import json
import os

import numpy as np

from references import kimhu_io as io
from references import segment as seg

N = 50                      # 위상 리샘플 점수
OUT = os.path.join(os.path.dirname(__file__), "out")
PEAK_FLOOR = 50.0           # 이보다 낮은 정점 = 불완전 거상(예: I03 66.6°는 통과, <50 컷)
T_FLOOR = 0.8               # 이보다 짧으면 flick(예: EXT19 0.66s 컷)

# 과제별 분절 파라미터(B 결정 b: T2는 전방 가슴앞 자세까지 포함해 평면 스윕 0→~45° 회복).
# T1=관상 상승만(평면 ~flat). T2=상승+전방스윕(지속된 가슴앞 정점을 큰 prominence로 포착).
TASK_SEG = {"T1": dict(distance=60, prominence=25),
            "T2": dict(distance=75, prominence=30)}


def _resample(x, n=N):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def _smooth(x, w=9):
    if len(x) <= w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _mad_keep(x, k=3.0):
    """MAD 기반 이상치 마스크(True=유지)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return np.abs(x - med) <= k * 1.4826 * mad


def gather(task):
    """task의 모든 피험자·반복에서 구간 채널·latent 수집. plane은 *궤적*으로(스윕 보존)."""
    elev_n, elbow_abs, prosup_abs, plane_shape = [], [], [], []  # 위상 리샘플 형태
    T, peak, plane, elbow_peak = [], [], [], []
    n_files = n_reps = n_rev = 0
    sp = TASK_SEG[task]
    for f in io.subject_files(task):
        frames = io.load_frames(f)
        if len(frames) < 100:
            continue
        n_files += 1
        ch = io.derive_channels(frames)
        elv = ch["elevation"]
        az_s = _smooth(ch["plane_az"])                   # 평면 궤적용 평활
        for lo, p in seg.rise_segments(elv, **sp):
            e = elv[lo:p + 1]
            n_rev += seg.reversals(e)
            pk = e.max() - e.min()
            dur = (p - lo) / io.FPS
            if e.max() < PEAK_FLOOR or dur < T_FLOOR:
                continue
            n_reps += 1
            T.append(dur)
            peak.append(e.max())
            # 평면 스칼라(가중 원형평균=대표값) + 궤적(스윕)
            plane.append(io.weighted_circmean(ch["plane_az"][lo:p + 1],
                                              ch["plane_mag"][lo:p + 1]))
            plane_shape.append(_resample(az_s[lo:p + 1]))
            eb = ch["elbow_flex"][lo:p + 1]
            elbow_peak.append(eb.max())
            en = (e - e.min()) / (pk + 1e-9)             # 진폭정규화 [0,1]
            elev_n.append(_resample(en))
            elbow_abs.append(_resample(eb))
            ps = ch["pro_sup"][lo:p + 1]
            prosup_abs.append(_resample(ps - ps[0]))     # 시작=0 기준 상대
    return {
        "elev_n": np.array(elev_n), "elbow_abs": np.array(elbow_abs),
        "prosup_abs": np.array(prosup_abs), "plane_shape": np.array(plane_shape),
        "T": np.array(T), "peak": np.array(peak),
        "plane": np.array(plane), "elbow_peak": np.array(elbow_peak),
        "n_files": n_files, "n_reps": n_reps, "n_rev": n_rev,
    }


def build(task):
    g = gather(task)
    # 이상치 트림: T·peak 동시 MAD 유지
    keep = _mad_keep(g["T"]) & _mad_keep(g["peak"])
    el = g["elev_n"][keep]
    mean_norm = np.mean(el, axis=0)                       # 로버스트(트림 후) 평균 형태
    # 형태 PC1(=skew): 진폭+시간 정규화 잔차의 1주성분
    X = el - mean_norm
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]
    if pc1[np.argmax(np.abs(pc1))] < 0:                   # 부호 결정적 고정
        pc1 = -pc1
    skew_scores = X @ pc1                                  # 반복별 skew

    tmpl = {
        "N": N,
        "elev_mean_norm": mean_norm,                      # [0,1] 평균 거상 형태
        "elev_pc1": pc1,                                  # skew 모드
        "elev_start": float(np.median(g["peak"][keep]) * 0),  # 시작 진폭(정규화 0)
        "elbow_shape": np.mean(g["elbow_abs"][keep], axis=0),    # deg 절대 평균
        "prosup_shape": np.mean(g["prosup_abs"][keep], axis=0),  # deg 상대 평균
        "plane_shape": np.mean(g["plane_shape"][keep], axis=0),  # deg 평면 궤적(T2=스윕)
        # 원시 표본(경험적 샘플링·상관 보존용)
        "s_T": g["T"][keep], "s_peak": g["peak"][keep],
        "s_skew": skew_scores, "s_plane": g["plane"][keep],
        "s_elbow_peak": g["elbow_peak"][keep],
    }
    np.savez(os.path.join(OUT, f"template_{task}.npz"), **tmpl)

    def dist(x):
        q = np.percentile(x, [5, 25, 50, 75, 95])
        return {"mean": float(np.mean(x)), "sd": float(np.std(x)),
                "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                "p75": float(q[3]), "p95": float(q[4]),
                "min": float(np.min(x)), "max": float(np.max(x))}
    latent = {"task": task, "plane_nominal": float(np.median(g["plane"][keep])),
              "n_reps_kept": int(keep.sum()), "n_reps_total": int(g["n_reps"]),
              "T": dist(g["T"][keep]), "peak": dist(g["peak"][keep]),
              "skew": dist(skew_scores), "plane": dist(g["plane"][keep]),
              "elbow_peak": dist(g["elbow_peak"][keep])}
    with open(os.path.join(OUT, f"latent_{task}.json"), "w") as fh:
        json.dump(latent, fh, indent=2)
    return g, tmpl, latent, keep


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ph = np.linspace(0, 1, N)
    for col, (task, (g, tmpl, latent, keep)) in enumerate(results.items()):
        ax = axes[0, col]
        for e in g["elev_n"][keep]:
            ax.plot(ph, e, color="0.8", lw=0.5)
        ax.plot(ph, tmpl["elev_mean_norm"], "b", lw=2, label="mean")
        ax.plot(ph, tmpl["elev_mean_norm"] + 0.3 * tmpl["elev_pc1"], "r--", lw=1,
                label="+0.3·PC1(skew)")
        ax.plot(ph, tmpl["elev_mean_norm"] - 0.3 * tmpl["elev_pc1"], "g--", lw=1,
                label="−0.3·PC1")
        ax.set_title(f"{task}: 거상 형태(진폭정규화), {keep.sum()}/{g['n_reps']}reps "
                     f"rev={g['n_rev']}")
        ax.set_xlabel("위상"); ax.legend(fontsize=8)
        ax = axes[1, col]
        ax.plot(ph, tmpl["elbow_shape"], "m", label="elbow_flex(deg)")
        ax.plot(ph, tmpl["prosup_shape"], "c", label="pro_sup(deg,rel)")
        ax.plot(ph, tmpl["plane_shape"], "orange", lw=2,
                label=f"plane(deg) {tmpl['plane_shape'][0]:+.0f}->{tmpl['plane_shape'][-1]:+.0f}")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_title(f"{task}: elbow/pro_sup/plane traj"); ax.set_xlabel("phase")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "m0_templates.png")
    fig.savefig(p, dpi=110)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for task in ("T1", "T2"):
        g, tmpl, latent, keep = build(task)
        results[task] = (g, tmpl, latent, keep)
        print(f"[{task}] files={g['n_files']} reps={g['n_reps']} "
              f"kept={int(keep.sum())} rise_reversals={g['n_rev']}")
        print(f"   T={latent['T']['p50']:.2f}s({latent['T']['p5']:.2f}-{latent['T']['p95']:.2f}) "
              f"peak={latent['peak']['p50']:.0f}°({latent['peak']['p5']:.0f}-{latent['peak']['p95']:.0f}) "
              f"plane={latent['plane_nominal']:.0f}° "
              f"elbow_peak={latent['elbow_peak']['p50']:.0f}°")
    print("plot:", _plot(results))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
