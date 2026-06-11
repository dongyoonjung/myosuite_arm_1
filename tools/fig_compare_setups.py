"""세 설정 식별성 비교: rise-only(손목잠금) vs full-seq+손목(T1) vs full-seq+손목(T1+T2)."""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_envs.muscle_groups import PERTURB_CHANNELS_WRIST

ALL = [c for c, _ in PERTURB_CHANNELS_WRIST]   # 12 (어깨10 + 손목2)
REP = os.path.join(os.path.dirname(__file__), "..", "results", "report")


def parse(path):
    r = {}; sec = None
    if not os.path.exists(path):
        return r
    for line in open(path):
        if "s_F: Fmax" in line: sec = 1
        elif "=== s_L" in line: sec = 0
        m = re.match(r"^(\w+)\s+[\d.]+\s+(-?[\d.]+)", line)
        if m and sec:
            r[m.group(1)] = float(m.group(2))
    return r


runs = {
    "rise-only T1+T2 (wrist locked)": parse(os.path.join(REP, "kinonly_vae_identifiability.txt")),
    "full-seq + wrist, T1": parse(os.path.join(REP, "wrist_identifiability.txt")),
    "full-seq + wrist, T1+T2": parse("/tmp/wmG.log"),
}
x = np.arange(len(ALL)); w = 0.26
fig, ax = plt.subplots(figsize=(16, 6))
colors = ["#7fbf7b", "#fdae61", "#2166ac"]
for j, (name, r) in enumerate(runs.items()):
    vals = [r.get(c, np.nan) for c in ALL]
    ax.bar(x + (j - 1) * w, vals, w, label=name, color=colors[j])
ax.axhline(0.6, color="r", ls=":", alpha=0.6, label="strong-ID (R²=0.6)")
ax.axvspan(9.5, 11.5, color="0.9", alpha=0.5)   # 손목근 영역 강조
ax.text(10.5, 0.95, "wrist\nmuscles", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ALL, rotation=30, ha="right")
ax.set_ylabel("R²  (s_F Fmax recovery, kinematics-only)"); ax.set_ylim(0, 1.0)
ax.set_title("Identifiability across setups: rise-only(no wrist) best for shoulder/elbow; "
             "wrist-unlock+full-seq adds wrist muscles but dilutes shoulder/elbow (rise phase is most informative)")
ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = os.path.join(REP, "fig11_compare_setups.png")
fig.savefig(p, dpi=120); print("saved:", p)
for name, r in runs.items():
    sh = np.nanmean([r.get(c, np.nan) for c in ALL[:10]])
    wr = np.nanmean([r.get(c, np.nan) for c in ALL[10:]])
    print(f"  {name}: 어깨/팔꿈치 평균 {sh:.2f} | 손목근 평균 {wr:.2f}")
