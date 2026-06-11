"""KIMHu per-repetition 상승 궤적 추출 → VAE 생성모델 학습셋.

각 반복: (elev, elbow, plane, prosup) 4채널을 위상 N=50으로 리샘플(deg). + T(이동시간).
기존 build_templates는 평균+PC1만 저장 → 여기선 *모든 반복 곡선*을 저장(생성모델용).

실행: python -m references.extract_traj   (data/kimhu/V2/ 필요)
산출: references/out/traj_{T1,T2}.npz (curves (n,50,4), T (n,), peak (n,))
"""
import os
import numpy as np

from references import kimhu_io as io
from references import segment as seg

N = 50
OUT = os.path.join(os.path.dirname(__file__), "out")
PEAK_FLOOR, T_FLOOR = 50.0, 0.8
TASK_SEG = {"T1": dict(distance=60, prominence=25),
            "T2": dict(distance=75, prominence=30)}
CHAN = ["elev", "elbow", "plane", "prosup"]


def _rs(x):
    return np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(x)), x)


def extract(task):
    sp = TASK_SEG[task]
    curves, Ts, peaks = [], [], []
    n_files = 0
    for f in io.subject_files(task):
        frames = io.load_frames(f)
        if len(frames) < 100:
            continue
        n_files += 1
        ch = io.derive_channels(frames)
        elv = ch["elevation"]
        plane = io.reliable_plane(ch["plane_az"], ch["plane_mag"])
        for lo, p in seg.rise_segments(elv, **sp):
            e = elv[lo:p + 1]
            dur = (p - lo) / io.FPS
            if e.max() < PEAK_FLOOR or dur < T_FLOOR:
                continue
            ps = ch["pro_sup"][lo:p + 1]
            cur = np.stack([_rs(e), _rs(ch["elbow_flex"][lo:p + 1]),
                            _rs(plane[lo:p + 1]), _rs(ps - ps[0])], axis=1)  # (N,4)
            curves.append(cur); Ts.append(dur); peaks.append(float(e.max()))
    return (np.array(curves, np.float32), np.array(Ts, np.float32),
            np.array(peaks, np.float32), n_files)


def main():
    os.makedirs(OUT, exist_ok=True)
    for task in ("T1", "T2"):
        cur, T, pk, nf = extract(task)
        np.savez(os.path.join(OUT, f"traj_{task}.npz"),
                 curves=cur, T=T, peak=pk, channels=np.array(CHAN))
        print(f"[{task}] files={nf} reps={len(cur)} curves{cur.shape} "
              f"T {T.min():.1f}-{T.max():.1f}s peak {pk.min():.0f}-{pk.max():.0f}°")


if __name__ == "__main__":
    main()
