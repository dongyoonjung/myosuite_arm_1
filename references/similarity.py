"""processed(실 KIMHu 반복) ↔ generated(latent 워핑 참조) *시퀀스* 유사도.

평균/템플릿 비교(통계처리)가 아니라, **개별 반복 시퀀스끼리**의 형태+크기 일치를 본다.
측정자(M0 조사 워크플로 선정 — fastdtw/dtaidistance 미설치 → 순수 numpy):
  · 주 = DTW(동적시간왜곡) 거리. 길이/타이밍 차이는 정렬로 흡수, 크기차는 벌점.
        path_length·채널범위로 정규화 → 채널 간 비교가능. 0=동일, ↑=상이.
  · 보조 = NRMSE(공통길이 리샘플 후 RMS/범위) — 상수 오프셋·스케일 포착(DTW 일부 허용분).
  · 진단 = Pearson r — 순수 형태(오프셋/스케일 무감).
shoulder_rot은 generated 참조가 없으므로(자유 DoF) 비교 대상에서 제외.

실행:  python -m references.similarity   → references/out/similarity_proc_vs_gen.png + 표 출력
"""
import os

import numpy as np

from references import kimhu_io as io
from references import segment as seg
from references import build_templates as bt
from references import warp

OUT = os.path.join(os.path.dirname(__file__), "out")
DT = 0.02
CHANNELS = [("shoulder_elv", "elevation (deg)"), ("elv_angle", "plane (deg)")]


# ---- 측정자 -----------------------------------------------------------------
def dtw_align(a, b):
    """DTW: (정규화 거리, 정렬경로[(i,j)...]). 정규화=총비용/(경로길이·범위) → 채널비교가능."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            c = abs(ai - b[j - 1])                       # L1 샘플 비용(deg)
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        s = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if s == 0:
            i, j = i - 1, j - 1
        elif s == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    rng = max(a.max() - a.min(), b.max() - b.min(), 1e-9)
    return D[n, m] / (len(path) * rng), path


def dtw_norm(a, b):
    return dtw_align(a, b)[0]


def _resample_pair(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    L = max(len(a), len(b))
    g = np.linspace(0, 1, L)
    ar = np.interp(g, np.linspace(0, 1, len(a)), a)
    br = np.interp(g, np.linspace(0, 1, len(b)), b)
    return ar, br


def nrmse(a, b):
    """공통 위상격자 리샘플 후 RMS오차/범위(차원무). 상수 오프셋·스케일 포착."""
    ar, br = _resample_pair(a, b)
    rng = max(np.ptp(a), np.ptp(b), 1e-9)
    return float(np.sqrt(np.mean((ar - br) ** 2)) / rng)


def mae(a, b):
    """공통 위상격자 리샘플 후 평균 절대오차(deg) — 정규화 안 한 해석가능 잔차."""
    ar, br = _resample_pair(a, b)
    return float(np.mean(np.abs(ar - br)))


def pearson(a, b):
    """형태 상관(오프셋/스케일 무감)."""
    ar, br = _resample_pair(a, b)
    if np.std(ar) < 1e-9 or np.std(br) < 1e-9:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def metrics(a, b):
    return {"dtw": dtw_norm(a, b), "nrmse": nrmse(a, b), "mae": mae(a, b), "r": pearson(a, b)}


# ---- processed/generated 시퀀스 구성(viz와 단일 출처) -----------------------
def _rep_score(elv, az, lo, p):
    """상승반복 단조도 점수(낮을수록 단조): 거상 반전수 + 평면 backslide(평활 후 진행최대 대비 후퇴).

    ★용도 = *그림 표시용 대표 반복 선택*(argmin). 가장 단조로운 반복을 골라 Kinect 큰 흔들림
    (예: MAYR02 T2 1번째: 거상 85→60→85 딥, 평면 -38° 스파이크)이 그림을 가리지 않게 한다.
    ※ 검증(adversarial)으로 확인: 이 점수는 generated 적합도와 거의 무상관(되레 약한 음상관)이라
       '글리치 분류기'가 아니다 — 단지 시각적으로 깨끗한 예시를 고르는 랭커. 적합도 척도는 table()의
       *전체 반복 분포*가 정직한 헤드라인(대표 반복 한 개로 과장하지 않음).
    """
    e = elv[lo:p + 1]
    a = np.asarray(az[lo:p + 1], float)
    asm = np.convolve(a, np.ones(5) / 5, mode="valid") if len(a) > 5 else a
    backslide = float(np.max(np.maximum.accumulate(asm) - asm)) if len(asm) else 0.0
    return seg.reversals(e) * 5.0 + backslide


def proc_and_gen(task, subject="MAYR02", clean_plane=True, rep="clean"):
    """task의 한 피험자 상승반복 → (processed, generated, latent, (lo,p)).

    processed = 실 반복(상승 구간) 채널, generated = 그 반복의 latent로 워핑한 참조(같은 길이).
    plane은 기본적으로 저신뢰 방위각을 정리(reliable_plane)해서 비교한다.
    rep="clean"(기본)이면 _rep_score 최소(가장 단조·잡음 적은) 반복을 고른다(대표 시퀀스).
    rep=정수면 그 인덱스 반복을 쓴다.
    """
    f = [p for p in io.subject_files(task) if subject in p][0]
    frames = io.load_frames(f)
    ch = io.derive_channels(frames)
    elv = ch["elevation"]
    az = io.reliable_plane(ch["plane_az"], ch["plane_mag"]) if clean_plane else ch["plane_az"]
    segs = seg.rise_segments(elv, **bt.TASK_SEG[task])
    if rep == "clean":
        idx = int(np.argmin([_rep_score(elv, az, lo, p) for lo, p in segs]))
    else:
        idx = int(rep)
    lo, p = segs[idx]
    proc = {"shoulder_elv": elv[lo:p + 1], "elv_angle": az[lo:p + 1]}

    e = elv[lo:p + 1]
    pk = e.max() - e.min()
    en = bt._resample((e - e.min()) / (pk + 1e-9))
    tmpl = warp.load_template(task)
    skew = float((en - tmpl["elev_mean_norm"]) @ tmpl["elev_pc1"])
    latent = {"T": (p - lo) / io.FPS, "peak": float(e.max()), "skew": skew,
              "plane": io.weighted_circmean(ch["plane_az"][lo:p + 1], ch["plane_mag"][lo:p + 1]),
              "elbow_peak": float(ch["elbow_flex"][lo:p + 1].max())}
    n = max(2, int(round(latent["T"] / DT)) + 1)
    ref = warp.make_reference(task, latent, n_steps=n, dt=DT)
    gen = {"shoulder_elv": ref["shoulder_elv"], "elv_angle": ref["elv_angle"]}
    return proc, gen, latent, (lo, p)


def compare(task, subject="MAYR02"):
    proc, gen, latent, _ = proc_and_gen(task, subject)
    return {key: metrics(proc[key], gen[key]) for key, _ in CHANNELS}


# ---- 표 + 그림 --------------------------------------------------------------
def _dist(x):
    return np.percentile(x, [50, 25, 75, 90, 100])      # median, p25, p75, p90, max


def all_reps(task, clean_plane=True):
    """모든 피험자·모든 (유지)상승반복의 per-rep self-fit 지표. {channel: {metric: array}}.

    각 반복마다 latent를 재추출해 generated를 만들고 그 반복과 비교(= 4-latent+템플릿 표현충실도).
    build_templates와 같은 PEAK_FLOOR/T_FLOOR 필터로 '유지' 반복만 집계.
    """
    out = {key: {"dtw": [], "mae": [], "r": []} for key, _ in CHANNELS}
    tmpl = warp.load_template(task)
    for f in io.subject_files(task):
        frames = io.load_frames(f)
        if len(frames) < 100:
            continue
        ch = io.derive_channels(frames)
        elv = ch["elevation"]
        az = io.reliable_plane(ch["plane_az"], ch["plane_mag"]) if clean_plane else ch["plane_az"]
        for lo, p in seg.rise_segments(elv, **bt.TASK_SEG[task]):
            e = elv[lo:p + 1]
            if e.max() < bt.PEAK_FLOOR or (p - lo) / io.FPS < bt.T_FLOOR:
                continue
            pk = e.max() - e.min()
            en = bt._resample((e - e.min()) / (pk + 1e-9))
            latent = {"T": (p - lo) / io.FPS, "peak": float(e.max()),
                      "skew": float((en - tmpl["elev_mean_norm"]) @ tmpl["elev_pc1"]),
                      "plane": io.weighted_circmean(ch["plane_az"][lo:p + 1], ch["plane_mag"][lo:p + 1]),
                      "elbow_peak": float(ch["elbow_flex"][lo:p + 1].max())}
            n = max(2, int(round(latent["T"] / DT)) + 1)
            ref = warp.make_reference(task, latent, n_steps=n, dt=DT)
            proc = {"shoulder_elv": e, "elv_angle": az[lo:p + 1]}
            for key, _ in CHANNELS:
                m = metrics(proc[key], ref[key])
                out[key]["dtw"].append(m["dtw"])
                out[key]["mae"].append(m["mae"])
                out[key]["r"].append(m["r"])
    return {key: {kk: np.array(v) for kk, v in d.items()} for key, d in out.items()}


def table():
    print("processed↔generated 시퀀스 유사도 — self-fit 표현충실도 (전체 반복 분포)")
    print("  ※ self-fit: 각 반복서 latent(T·peak·skew·plane·elbow) 재추출 후 생성 → '4-latent + 단조")
    print("     템플릿'이 실 반복 형태를 얼마나 담는가의 척도(개별 시퀀스끼리, 궤적 평균 아님).")
    print("     런타임은 sampled latent로 생성; 미지 latent 일반화 검증은 M1/M2 게이트 몫.")
    print("  DTW=정규화 동시간왜곡(0=동일) | MAE=평균절대오차(deg, 해석가능) | r=형태상관")
    for task in ("T1", "T2"):
        ar = all_reps(task)
        n = len(ar["shoulder_elv"]["dtw"])
        print(f"\n[{task}] 전체 {n}반복   채널          DTW(med p25–p75 p90 max)        MAE°(med p90)   r(med)")
        for key, _ in CHANNELS:
            dq, mq = _dist(ar[key]["dtw"]), _dist(ar[key]["mae"])
            print(f"               {key:>11}    {dq[0]:.3f} {dq[1]:.3f}–{dq[2]:.3f} {dq[3]:.3f} {dq[4]:.3f}    "
                  f"{mq[0]:4.1f} {mq[3]:4.1f}     {np.median(ar[key]['r']):+.2f}")


def _plot(subject="MAYR02"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 행=채널(elevation, plane), 열=과제(T1, T2). DTW 정렬 connector 포함.
    fig, axes = plt.subplots(len(CHANNELS), 2, figsize=(13, 7))
    for col, task in enumerate(("T1", "T2")):
        proc, gen, latent, _ = proc_and_gen(task, subject)
        for row, (key, ylab) in enumerate(CHANNELS):
            ax = axes[row, col]
            a, b = np.asarray(proc[key], float), np.asarray(gen[key], float)
            pa = np.linspace(0, 1, len(a))
            pb = np.linspace(0, 1, len(b))
            dist, path = dtw_align(a, b)
            # 정렬 connector(샘플별 매칭) — 성기게
            for k in range(0, len(path), max(1, len(path) // 40)):
                i, j = path[k]
                ax.plot([pa[i], pb[j]], [a[i], b[j]], color="0.85", lw=0.6, zorder=1)
            ax.plot(pa, a, "b", lw=2.2, label="processed (real rep)", zorder=3)
            ax.plot(pb, b, "r--", lw=1.8, label="generated (latent warp)", zorder=3)
            m = metrics(a, b)
            ax.text(0.03, 0.97,
                    f"DTW={m['dtw']:.3f}\nMAE={m['mae']:.1f}deg\nr={m['r']:+.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
            ax.set_ylabel(ylab)
            ax.set_title(f"{task}  {key}")
            if row == len(CHANNELS) - 1:
                ax.set_xlabel("phase [0,1]  (gray = DTW alignment)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"processed vs generated sequence similarity  [{subject}, representative rep]  "
                 f"(DTW primary; lower = more similar)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.text(0.5, 0.005,
             "self-fit: latent re-derived from THIS rep, so this is representational fidelity of the "
             "4-latent+monotone template, not generalization. Representative (most-monotone) rep shown "
             "for clarity; honest headline = all-reps distribution in table().",
             ha="center", fontsize=7, color="0.4")
    path = os.path.join(OUT, "similarity_proc_vs_gen.png")
    fig.savefig(path, dpi=110)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    table()
    print("\nplot:", _plot())


if __name__ == "__main__":
    main()
