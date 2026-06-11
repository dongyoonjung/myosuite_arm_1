"""참조 생성법 비교: REAL vs warp(평균+1PC) vs VAE vs fPCA(ProMP).
주변분포 일치(KS·sd), 다양성(1-NN), 매끄러움 + 포락선 그림."""
import os
import sys

import numpy as np
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from references import warp, vae_ref, fpca_ref

OUT = os.path.join(os.path.dirname(__file__), "..", "references", "out")
REP = os.path.join(os.path.dirname(__file__), "..", "results", "report")
N = 50
rng = np.random.default_rng(0)


def phase_curve(mod, task, n=400):
    """각 생성법에서 위상 50점 (n,50,4) deg 곡선."""
    cs = []
    for _ in range(n):
        lat = mod.sample_latent(task, rng)
        if "_curve" in lat:
            cs.append(lat["_curve"])
        else:                                            # warp: make_reference로 위상곡선
            r = mod.make_reference(task, lat, N, lat["T"] / N)
            cs.append(np.degrees(np.stack([r["shoulder_elv"], r["elbow_flexion"],
                                           r["elv_angle"], r["pro_sup"]], 1)))
    return np.array(cs, np.float32)


def feats(c):
    return {"peak": c[:, :, 0].max(1), "start": c[:, 0, 0], "elbow_pk": c[:, :, 1].max(1),
            "plane_mean": c[:, :, 2].mean(1), "plane_rng": c[:, :, 2].max(1) - c[:, :, 2].min(1)}


def main():
    for task in ("T1", "T2"):
        real = np.load(os.path.join(OUT, f"traj_{task}.npz"))["curves"].astype(np.float32)
        gens = {"warp": phase_curve(warp, task), "VAE": phase_curve(vae_ref, task),
                "fPCA": phase_curve(fpca_ref, task)}
        fr = feats(real)
        print(f"\n===== {task} (real n={len(real)}) — 특징 mean±sd (KS vs real) =====")
        keys = ["peak", "start", "elbow_pk", "plane_mean", "plane_rng"]
        print(f"{'특징':<11}{'REAL':>13}" + "".join(f"{g:>20}" for g in gens))
        for k in keys:
            row = f"{k:<11}{fr[k].mean():>6.0f}±{fr[k].std():<5.0f}"
            for g, c in gens.items():
                fk = feats(c)[k]; ks, _ = ks_2samp(fr[k], fk)
                row += f"{fk.mean():>8.0f}±{fk.std():<4.0f}KS{ks:>4.2f}"
            print(row)
        # 다양성: 1-NN(gen→real) 정규화
        cmu = real.reshape(-1, 4).mean(0); csd = real.reshape(-1, 4).std(0) + 1e-6
        R = ((real - cmu) / csd).reshape(len(real), -1)
        def nn(c):
            G = ((c - cmu) / csd).reshape(len(c), -1)
            return np.mean([np.linalg.norm(R - g, axis=1).min() for g in G])
        rr = np.mean([np.partition(np.linalg.norm(R - r, axis=1), 1)[1] for r in R])
        print(f"1-NN(gen→real): " + " ".join(f"{g} {nn(c):.1f}" for g, c in gens.items())
              + f" | real→real {rr:.1f}")

    # 포락선 그림(T1·T2 elev): real vs VAE vs fPCA
    ph = np.linspace(0, 1, N)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, task in zip(axes, ["T1", "T2"]):
        real = np.load(os.path.join(OUT, f"traj_{task}.npz"))["curves"][:, :, 0]
        v = phase_curve(vae_ref, task)[:, :, 0]; f = phase_curve(fpca_ref, task)[:, :, 0]
        for data, col, lab in [(real, "0.4", "REAL"), (v, "tab:orange", "VAE"), (f, "tab:blue", "fPCA")]:
            mn, sd = data.mean(0), data.std(0)
            ax.plot(ph, mn, color=col, lw=2, label=lab); ax.fill_between(ph, mn - sd, mn + sd, color=col, alpha=0.15)
        ax.set_title(f"{task}: elevation mean±std"); ax.set_xlabel("phase"); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Reference generators vs REAL: fPCA matches real dispersion, VAE under-disperses", fontsize=12)
    fig.tight_layout(); p = os.path.join(REP, "fig9_refgen_compare.png"); fig.savefig(p, dpi=110)
    print("\nsaved:", p)


if __name__ == "__main__":
    main()
