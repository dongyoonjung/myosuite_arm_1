"""KIN-only(VAE참조+운동노이즈) 식별성 그림 — 채널별 s_F·s_L R²."""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_envs import muscle_groups as MG

CH = [c for c, _ in MG.PERTURB_CHANNELS]


def parse(path):
    sf, sl = {}, {}; sec = None
    for line in open(path):
        if "s_F: Fmax" in line: sec = "sf"
        elif "=== s_L" in line: sec = "sl"
        m = re.match(r"^(\w+)\s+[\d.]+\s+(-?[\d.]+)", line)
        if m and m.group(1) in CH:
            (sf if sec == "sf" else sl)[m.group(1)] = float(m.group(2))
    return sf, sl


sf, sl = parse("/tmp/kinvae_G.log")
x = np.arange(len(CH)); w = 0.4
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.bar(x - w / 2, [sf.get(c, 0) for c in CH], w, label="s_F (Fmax scale)", color="#2166ac")
ax.bar(x + w / 2, [sl.get(c, 0) for c in CH], w, label="s_L (Lopt scale)", color="#f4a582")
ax.axhline(0.6, color="r", ls=":", alpha=0.6, label="strong-ID (R²=0.6)")
ax.set_xticks(x); ax.set_xticklabels(CH, rotation=30, ha="right")
ax.set_ylabel("R²"); ax.set_ylim(0, 1.0)
ax.set_title("KINEMATICS-ONLY identifiability (VAE references + motor noise, NO EMG)\n"
             f"s_F mean R²={np.mean([sf.get(c,0) for c in CH]):.2f}, "
             f"s_L mean R²={np.mean([sl.get(c,0) for c in CH]):.2f}  — time-series TCN, 20k rollouts")
ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = os.path.join(os.path.dirname(__file__), "..", "results", "report", "fig8_kinonly_vae.png")
fig.savefig(p, dpi=120); print("saved:", p)
