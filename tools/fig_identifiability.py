"""식별성 결과 그림 — 채널별 s_F 복원 R² (ACT만 / KIN만 / ACT+KIN).
DESIGN 명제 "ACT 주채널·KIN 보너스"의 시각 증거."""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import regress_nn as R
from custom_envs import muscle_groups as MG

CH = [c for c, _ in MG.PERTURB_CHANNELS]
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "sim", "train.npz")


def r2_per_channel(feats, epochs=150):
    model, (Xv, yv), _ = R.train(DATA, epochs=epochs, feats=feats, reweight=True)
    model.eval()
    with torch.no_grad():
        pv = model(Xv).numpy()
    yv = yv.numpy()
    out = []
    for i in range(10):  # s_F
        t, p = yv[:, i], pv[:, i]
        r2 = 1 - ((t - p) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9)
        out.append(max(-0.05, r2))
    return np.array(out)


def main():
    sets = {"ACT only (privileged)": "act", "KIN only (kinematics)": "kin",
            "ACT + KIN (full)": "actkin"}
    res = {name: r2_per_channel(f) for name, f in sets.items()}
    x = np.arange(10); w = 0.26
    fig, ax = plt.subplots(figsize=(14, 5.5))
    colors = {"ACT only (privileged)": "#2b8cbe", "KIN only (kinematics)": "#fdae61",
              "ACT + KIN (full)": "#1a9850"}
    for j, (name, r2) in enumerate(res.items()):
        ax.bar(x + (j - 1) * w, r2, w, label=name, color=colors[name])
    ax.axhline(0.6, color="r", ls=":", alpha=0.6, label="strong-ID threshold (R²=0.6)")
    ax.set_xticks(x); ax.set_xticklabels(CH, rotation=30, ha="right")
    ax.set_ylabel("R²  (s_F = Fmax scale recovery)")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Per-channel identifiability: ACT is primary channel, KIN is bonus\n"
                 "(compensable muscles invisible in KIN but recovered by ACT; "
                 "cuff/abductors also show in KIN)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "report", "fig5_identifiability.png")
    fig.savefig(out, dpi=120)
    print("saved:", out)
    for name, r2 in res.items():
        print(f"  {name}: mean R² {r2.mean():.2f}")


if __name__ == "__main__":
    main()
