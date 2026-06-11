"""손목 해제 + full-seq(T1) KIN-only 식별성 그림 — 손목근 포함 12채널."""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_envs.muscle_groups import PERTURB_CHANNELS_WRIST

CH = [c for c, _ in PERTURB_CHANNELS_WRIST]


def parse(path):
    sf, sl = {}, {}; sec = None
    for line in open(path):
        if "s_F: Fmax" in line: sec = "sf"
        elif "=== s_L" in line: sec = "sl"
        m = re.match(r"^(\w+)\s+[\d.]+\s+(-?[\d.]+)", line)
        if m and m.group(1) in CH:
            (sf if sec == "sf" else sl)[m.group(1)] = float(m.group(2))
    return sf, sl


sf, sl = parse("/tmp/wristG.log")
x = np.arange(len(CH)); w = 0.4
fig, ax = plt.subplots(figsize=(15, 5.5))
cols = ["#2166ac"] * 10 + ["#d6604d"] * 2          # 손목근 2개 강조
ax.bar(x - w / 2, [sf.get(c, 0) for c in CH], w, color=cols, label="s_F (Fmax)")
ax.bar(x + w / 2, [sl.get(c, 0) for c in CH], w, color="#f4a582", alpha=0.8, label="s_L (Lopt)")
ax.axhline(0.6, color="r", ls=":", alpha=0.6, label="strong-ID (R²=0.6)")
ax.set_xticks(x); ax.set_xticklabels(CH, rotation=30, ha="right")
ax.set_ylabel("R²"); ax.set_ylim(0, 1.0)
ax.set_title("WRIST-UNLOCKED + full-sequence (T1), KINEMATICS-ONLY identifiability\n"
             f"s_F mean R²={np.mean([sf.get(c,0) for c in CH]):.2f}  "
             "(red = newly-unlocked wrist muscles: identifiable! but shoulder/elbow partly reduced)")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = os.path.join(os.path.dirname(__file__), "..", "results", "report", "fig10_wrist.png")
fig.savefig(p, dpi=120); print("saved:", p)
