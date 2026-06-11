"""VAE 참조 생성 검증: 실제 KIMHu reps vs VAE 샘플 vs 기존 warp — elev·plane 곡선 + 분포."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from references import vae_ref, warp

OUT = os.path.join(os.path.dirname(__file__), "..", "references", "out")
N = 50
ph = np.linspace(0, 1, N)
rng = np.random.default_rng(0)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for r, task in enumerate(["T1", "T2"]):
    real = np.load(os.path.join(OUT, f"traj_{task}.npz"))["curves"]   # (n,50,4)
    # 실제
    ax = axes[r, 0]
    for c in real[rng.choice(len(real), 30, replace=False)]:
        ax.plot(ph, c[:, 0], color="0.5", lw=0.6, alpha=0.6)
    ax.set_title(f"{task}: REAL KIMHu reps (elev)"); ax.set_ylabel("elevation (deg)")
    # VAE 샘플
    ax = axes[r, 1]
    vpeaks, vplanes = [], []
    for _ in range(30):
        lat = vae_ref.sample_latent(task, rng)
        ref = vae_ref.make_reference(task, lat, N, 1.0 / N)   # phase축 그대로
        cur = lat["_curve"]
        ax.plot(ph, cur[:, 0], lw=0.7, alpha=0.7)
        vpeaks.append(cur[:, 0].max()); vplanes.append(cur[:, 2].mean())
    ax.set_title(f"{task}: VAE samples (elev)")
    # 기존 warp
    ax = axes[r, 2]
    for _ in range(30):
        lat = warp.sample_latent(task, rng)
        ref = warp.make_reference(task, lat, N, 1.0 / N)
        ax.plot(ph, ref["shoulder_elv"], lw=0.7, alpha=0.7)
    ax.set_title(f"{task}: warp (mean+1PC) (elev)")
    for ax in axes[r]:
        ax.set_xlabel("phase"); ax.grid(alpha=0.3); ax.set_ylim(0, 130)
    # 분포 비교(정점)
    rp = real[:, :, 0].max(1)
    print(f"[{task}] 정점 분포 — REAL {rp.mean():.0f}±{rp.std():.0f}° | "
          f"VAE {np.mean(vpeaks):.0f}±{np.std(vpeaks):.0f}°")

fig.suptitle("Reference generation: REAL KIMHu vs VAE generative vs warp(mean+1PC). "
             "VAE captures fuller shape diversity from data.", fontsize=12)
fig.tight_layout()
p = os.path.join(os.path.dirname(__file__), "..", "results", "report", "fig7_vae_reference.png")
os.makedirs(os.path.dirname(p), exist_ok=True)
fig.savefig(p, dpi=110)
print("saved:", p)
