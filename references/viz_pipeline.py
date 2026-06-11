"""파이프라인 검증 시각화: 어깨 3축(거상·평면·축회전)을
   원본(raw, 전체 반복) → 가공(processed, 상승/전방 분절) → 생성(generated, latent 워핑)
   세 단계로 겹쳐 시계열 비교. T1·T2 각각.

실행:  python -m references.viz_pipeline   → references/out/shoulder_axes_pipeline.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from references import kimhu_io as io
from references import segment as seg
from references import build_templates as bt
from references import warp

OUT = os.path.join(os.path.dirname(__file__), "out")


def _sm(x, w=7):
    return np.convolve(x, np.ones(w) / w, mode="same") if len(x) > w else x


def _shoulder_rot_raw(frames):
    """raw shoulder_rot(축회전) — 쿼터니언 roll(검증불가=신뢰못함, 참조로 안 씀). 표시용."""
    sho = io._pos(frames, "ShoulderRight"); elb = io._pos(frames, "ElbowRight")
    spB = io._pos(frames, "SpineBase"); spS = io._pos(frames, "SpineShoulder")
    qS = io._quat(frames, "ShoulderRight")
    ua = elb - sho; down = spB - spS
    # 쿼터니언 → 회전행렬 local X
    def Rx(q):
        x, y, z, w = q / (np.linalg.norm(q) + 1e-9)
        return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
    lx = np.array([Rx(q) for q in qS])
    return io._roll_about(ua, lx, down)


def _rep_window(elv, lo, p):
    """반복 표시 구간: 분절 시작 lo 부터 정점 후 하강(40%) 까지."""
    end = p
    while end < len(elv) - 1 and elv[end] > 0.4 * elv[p]:
        end += 1
    return lo, end


def make(task, ax_elv, ax_pl, ax_rot, rng):
    f = [p for p in io.subject_files(task) if "MAYR02" in p][0]
    frames = io.load_frames(f)
    ch = io.derive_channels(frames)
    elv, az = ch["elevation"], ch["plane_az"]
    rot = _shoulder_rot_raw(frames)
    sp = bt.TASK_SEG[task]
    lo, p = seg.rise_segments(elv, **sp)[0]
    w0, w1 = _rep_window(elv, lo, p)
    t = (np.arange(w0, w1 + 1) - lo) / io.FPS                 # 분절 시작=0초
    ts = (np.arange(lo, p + 1) - lo) / io.FPS                 # 가공(분절) 시간축

    # 생성: 이 반복의 latent로 워핑(직접 비교)
    e_seg = elv[lo:p + 1]; pk = e_seg.max() - e_seg.min()
    en = bt._resample((e_seg - e_seg.min()) / (pk + 1e-9))
    tmpl = warp.load_template(task)
    skew = float((en - tmpl["elev_mean_norm"]) @ tmpl["elev_pc1"])
    latent = {"T": (p - lo) / io.FPS, "peak": float(e_seg.max()), "skew": skew,
              "plane": io.weighted_circmean(az[lo:p + 1], ch["plane_mag"][lo:p + 1]),
              "elbow_peak": float(ch["elbow_flex"][lo:p + 1].max())}
    dt = 0.02; n = int((w1 - lo) / io.FPS / dt)
    ref = warp.make_reference(task, latent, n_steps=n, dt=dt)
    tg = np.arange(n) * dt

    # --- elevation ---
    ax_elv.plot(t, _sm(elv[w0:w1 + 1]), color="0.6", label="raw (full rep)")
    ax_elv.plot(ts, _sm(elv[lo:p + 1]), "b", lw=2.5, label="processed (segment)")
    ax_elv.plot(tg, ref["shoulder_elv"], "r--", lw=1.8, label="generated (latent warp)")
    ax_elv.set_ylabel("shoulder_elv (deg)"); ax_elv.set_title(f"{task}  elevation")
    ax_elv.legend(fontsize=7)
    # --- plane ---
    ax_pl.plot(t, _sm(az[w0:w1 + 1]), color="0.6")
    ax_pl.plot(ts, _sm(az[lo:p + 1]), "b", lw=2.5)
    ax_pl.plot(tg, ref["elv_angle"], "r--", lw=1.8)
    ax_pl.axhline(0, color="0.8", lw=0.5)
    ax_pl.set_ylabel("elv_angle plane (deg, + ant.)")
    ax_pl.set_title(f"{task}  plane of elevation (T2 = forward sweep)")
    # --- axial rotation ---
    ax_rot.plot(t, _sm(rot[w0:w1 + 1]), color="0.6")
    ax_rot.plot(ts, _sm(rot[lo:p + 1]), "b", lw=2.5)
    ax_rot.axhline(0, color="0.8", lw=0.5)
    ax_rot.text(0.02, 0.92, "generated = NONE (free DoF, no reference)\nraw also untrusted (quaternion failed cross-check)",
                transform=ax_rot.transAxes, fontsize=7, va="top", color="darkred")
    ax_rot.set_ylabel("shoulder_rot axial (deg)"); ax_rot.set_title(f"{task}  axial rotation")
    ax_rot.set_xlabel("time (s, segment start = 0)")


def main():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex="col")
    for col, task in enumerate(("T1", "T2")):
        make(task, axes[0, col], axes[1, col], axes[2, col], rng)
    fig.suptitle("Shoulder 3 axes: raw -> processed (segment) -> generated (latent warp)  [MAYR02]",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT, "shoulder_axes_pipeline.png")
    fig.savefig(path, dpi=110)
    print("saved:", path)


if __name__ == "__main__":
    main()
