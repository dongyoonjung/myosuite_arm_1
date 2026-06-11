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
from references import similarity as sim

OUT = os.path.join(os.path.dirname(__file__), "out")


def _sm(x, w=7):
    """이동평균 평활. 경계는 edge-pad로 보존.
    (mode="same"는 배열 밖을 0으로 패딩해 양끝이 0으로 끌려가는 가짜 급강하를 만든다.)
    """
    x = np.asarray(x, float)
    if len(x) <= w:
        return x
    pad = w // 2
    y = np.convolve(np.pad(x, pad, mode="edge"), np.ones(w) / w, mode="valid")
    if len(y) == len(x):
        return y
    return np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(y)), y)


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


def make(task, ax_elv, ax_pl, ax_rot, subject="MAYR02"):
    f = [p for p in io.subject_files(task) if subject in p][0]
    frames = io.load_frames(f)
    ch = io.derive_channels(frames)
    elv = ch["elevation"]
    az = io.reliable_plane(ch["plane_az"], ch["plane_mag"])   # 저신뢰(수직팔) 방위각 정리
    rot = _shoulder_rot_raw(frames)

    # processed/generated 시퀀스는 similarity와 단일 출처에서 구성(plane은 정리된 az 사용)
    proc, gen, latent, (lo, p) = sim.proc_and_gen(task, subject)
    w0, w1 = _rep_window(elv, lo, p)
    t = (np.arange(w0, w1 + 1) - lo) / io.FPS                 # raw full-rep 시간축
    ts = (np.arange(lo, p + 1) - lo) / io.FPS                 # processed(분절) 시간축
    tg = np.arange(len(gen["shoulder_elv"])) * sim.DT         # generated 시간축(상승 T)

    d_elv = sim.metrics(proc["shoulder_elv"], gen["shoulder_elv"])
    d_pl = sim.metrics(proc["elv_angle"], gen["elv_angle"])

    # --- elevation ---
    ax_elv.plot(t, _sm(elv[w0:w1 + 1]), color="0.6", label="raw (full rep)")
    ax_elv.plot(ts, _sm(proc["shoulder_elv"]), "b", lw=2.5, label="processed (segment)")
    ax_elv.plot(tg, gen["shoulder_elv"], "r--", lw=1.8, label="generated (latent warp)")
    ax_elv.set_ylabel("shoulder_elv (deg)")
    ax_elv.set_title(f"{task}  elevation   (DTW={d_elv['dtw']:.3f}, r={d_elv['r']:+.2f})")
    ax_elv.legend(fontsize=7)
    ax_elv.text(0.02, 0.04,
                "gray = full rep (rise+descent; descent is discarded)\n"
                "blue/red over rise T only -> both end at peak\n"
                "env appends hold@peak (raise-and-hold), not shown here",
                transform=ax_elv.transAxes, fontsize=6.5, va="bottom", color="0.35")
    # --- plane ---
    ax_pl.plot(t, _sm(az[w0:w1 + 1]), color="0.6")
    ax_pl.plot(ts, _sm(proc["elv_angle"]), "b", lw=2.5)
    ax_pl.plot(tg, gen["elv_angle"], "r--", lw=1.8)
    ax_pl.axhline(0, color="0.8", lw=0.5)
    ax_pl.set_ylabel("elv_angle plane (deg, + ant.)")
    ax_pl.set_title(f"{task}  plane of elevation   (DTW={d_pl['dtw']:.3f}, r={d_pl['r']:+.2f})")
    ax_pl.text(0.02, 0.04,
               "azimuth magnitude-masked (plane_mag<0.13 = arm near-vertical -> az undefined)\n"
               "representative (cleanest-rise) rep shown; T2 = forward sweep 0 -> ~+45 deg",
               transform=ax_pl.transAxes, fontsize=6.5, va="bottom", color="0.35")
    # --- axial rotation ---
    ax_rot.plot(t, _sm(rot[w0:w1 + 1]), color="0.6", label="raw (quaternion, untrusted)")
    ax_rot.plot(ts, _sm(rot[lo:p + 1]), "b", lw=2.5)
    ax_rot.axhline(0, color="0.8", lw=0.5)
    ax_rot.text(0.02, 0.94,
                "generated = NONE by design: shoulder_rot is a FREE DoF, no KIMHu reference.\n"
                "Humeral axial twist is not isolable from Kinect on this protocol -- the only\n"
                "quaternion that tracks is confounded ~0.84 with elbow_flexion (it carries the\n"
                "forearm through space, not the isolated twist). Building a reference would\n"
                "inject elbow signal into the rot target = inventing beyond data.\n"
                "In-sim physics supplies obligatory external rotation; policy learns it free.",
                transform=ax_rot.transAxes, fontsize=6.0, va="top", color="darkred")
    ax_rot.set_ylabel("shoulder_rot axial (deg)"); ax_rot.set_title(f"{task}  axial rotation")
    ax_rot.set_xlabel("time (s, segment start = 0)")


def main():
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex="col")
    for col, task in enumerate(("T1", "T2")):
        make(task, axes[0, col], axes[1, col], axes[2, col])
    fig.suptitle("Shoulder 3 axes: raw -> processed (segment) -> generated (latent warp)  [MAYR02]",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT, "shoulder_axes_pipeline.png")
    fig.savefig(path, dpi=110)
    print("saved:", path)


if __name__ == "__main__":
    main()
