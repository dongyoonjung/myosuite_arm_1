"""그림6: 채널별 s_F 복원 R² — 스냅샷 MLP 대 시계열 TCN."""
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
SNAP_SF = {"DELT1": .94, "DELT2": .96, "DELT3": .93, "SUPSP": .94, "PECM1": .83,
           "CORB": .92, "BIClong": .91, "TRIlong": .94, "A_lowcuff": .97, "B_latadd": .96}


def parse(path):
    r = {}; sec = None
    for line in open(path):
        if "s_F: Fmax" in line:
            sec = "sF"
        elif "=== s_L" in line:
            sec = "sL"
        m = re.match(r"^(\w+)\s+[\d.]+\s+(-?[\d.]+)", line)
        if m and m.group(1) in CH and sec == "sF":
            r[m.group(1)] = float(m.group(2))
    return r


seq = parse("/tmp/seqG_all.log")
x = np.arange(len(CH)); w = 0.4
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.bar(x - w / 2, [SNAP_SF[c] for c in CH], w, label="Snapshot MLP (mean/max summary)", color="#7fbf7b")
ax.bar(x + w / 2, [seq.get(c, 0) for c in CH], w, label="Sequence TCN (time series)", color="#2166ac")
ax.axhline(0.6, color="r", ls=":", alpha=0.6, label="strong-ID threshold (R²=0.6)")
ax.set_xticks(x); ax.set_xticklabels(CH, rotation=30, ha="right")
ax.set_ylabel("R²  (s_F = Fmax scale recovery)"); ax.set_ylim(0, 1.0)
ax.set_title("Per-channel identifiability: snapshot summary vs full time-series regression")
ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "results", "report", "fig6_seq_vs_snapshot.png")
fig.savefig(out, dpi=120)
print("saved:", out, "| seq mean", round(float(np.mean([seq.get(c,0) for c in CH])), 3))
