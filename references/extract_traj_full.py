"""KIMHu *full sequence* per-rep 사이클 추출 (저점→정점→복귀 전체), 6채널.

기존 extract_traj는 상승(외전)만. 여기선 전체 사이클(거상→팔꿈치·신전→손목조작→복귀)을
포함하고 손목 굴곡·편위 2채널 추가 → 손목 해제 + 전체동작 모방용.
채널: elev, elbow, plane, prosup, wrist_flex, wrist_dev (위상 N_FULL=80 리샘플).

실행: python -m references.extract_traj_full   (data/kimhu/V2/ 필요)
산출: references/out/trajfull_{T1,T2}.npz
"""
import os
import numpy as np
from scipy.signal import find_peaks

from references import kimhu_io as io

N_FULL = 80
OUT = os.path.join(os.path.dirname(__file__), "out")
PEAK_FLOOR, T_FLOOR = 50.0, 0.8
TASK_SEG = {"T1": dict(distance=60, prominence=25), "T2": dict(distance=75, prominence=30)}
CHAN = ["elev", "elbow", "plane", "prosup", "wrist_flex", "wrist_dev"]


def _rs(x, n=N_FULL):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def _sm(x, w=5):
    if len(x) <= w:
        return x
    k = np.ones(w) / w
    return np.convolve(np.pad(x, w // 2, mode="edge"), k, mode="valid")[:len(x)]


def full_cycles(elev, distance, prominence, min_peak=55.0):
    """전체 사이클 [(lo, hi)] — 정점 전후 저점 쌍(거상→복귀)."""
    pks, _ = find_peaks(elev, height=min_peak, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-elev, distance=distance // 2)
    troughs = np.concatenate([[0], troughs, [len(elev) - 1]])
    cycles = []
    for p in pks:
        before = troughs[troughs < p]
        after = troughs[troughs > p]
        if len(before) and len(after):
            cycles.append((int(before[-1]), int(after[0])))
    return cycles


def extract(task):
    sp = TASK_SEG[task]
    curves, Ts, peaks = [], [], []
    nf = 0
    for f in io.subject_files(task):
        frames = io.load_frames(f)
        if len(frames) < 100:
            continue
        nf += 1
        ch = io.derive_channels(frames)
        elv = ch["elevation"]
        plane = io.reliable_plane(ch["plane_az"], ch["plane_mag"])
        for lo, hi in full_cycles(elv, sp["distance"], sp["prominence"]):
            seg = elv[lo:hi + 1]
            dur = (hi - lo) / io.FPS
            if seg.max() < PEAK_FLOOR or dur < T_FLOOR or (hi - lo) < 20:
                continue
            ps = ch["pro_sup"][lo:hi + 1]
            cur = np.stack([_rs(seg), _rs(ch["elbow_flex"][lo:hi + 1]),
                            _rs(plane[lo:hi + 1]), _rs(ps - ps[0]),
                            _rs(_sm(ch["wrist_flex"][lo:hi + 1])),
                            _rs(_sm(ch["wrist_dev"][lo:hi + 1]))], axis=1)  # (N,6)
            curves.append(cur); Ts.append(dur); peaks.append(float(seg.max()))
    return np.array(curves, np.float32), np.array(Ts, np.float32), np.array(peaks, np.float32), nf


def main():
    os.makedirs(OUT, exist_ok=True)
    for task in ("T1", "T2"):
        cur, T, pk, nf = extract(task)
        np.savez(os.path.join(OUT, f"trajfull_{task}.npz"),
                 curves=cur, T=T, peak=pk, channels=np.array(CHAN))
        wf = cur[:, :, 4]; wd = cur[:, :, 5]
        print(f"[{task}] files={nf} cycles={len(cur)} {cur.shape} T {T.min():.1f}-{T.max():.1f}s "
              f"peak {pk.min():.0f}-{pk.max():.0f}° | wrist_flex {wf.min():.0f}~{wf.max():.0f}° "
              f"wrist_dev {wd.min():.0f}~{wd.max():.0f}°")


if __name__ == "__main__":
    main()
