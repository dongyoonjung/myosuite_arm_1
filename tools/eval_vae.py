"""VAE 참조 생성모델 품질 평가 — 얼마나 그럴듯한 kin 궤적을 생성하나.

평가: (1) 재구성 충실도(deg) (2) 주변분포 일치(real vs VAE, KS) (3) 다양성(1-NN, 붕괴여부)
      (4) 채널간 상관 보존 (5) 형태 포락선(mean±std) (6) 매끄러움.
산출: results/report/eval_vae_curves.png, eval_vae_dist.png + 콘솔 표.
"""
import os
import sys

import numpy as np
import torch
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from references import vae_ref

OUT = os.path.join(os.path.dirname(__file__), "..", "references", "out")
REP = os.path.join(os.path.dirname(__file__), "..", "results", "report")
os.makedirs(REP, exist_ok=True)
N = 50
CH = ["elev", "elbow", "plane", "prosup"]
rng = np.random.default_rng(0)


def feats(curves):
    """곡선 (n,N,4) → 스칼라 특징 dict."""
    return {"start_elev": curves[:, 0, 0], "peak_elev": curves[:, :, 0].max(1),
            "elbow_peak": curves[:, :, 1].max(1), "plane_mean": curves[:, :, 2].mean(1),
            "plane_range": curves[:, :, 2].max(1) - curves[:, :, 2].min(1),
            "prosup_range": curves[:, :, 3].max(1) - curves[:, :, 3].min(1)}


def smoothness(curves):
    """2차 차분 절대평균(작을수록 매끄러움), elev 채널."""
    return np.abs(np.diff(curves[:, :, 0], n=2, axis=1)).mean()


def evaluate(task, n_gen=500):
    real = np.load(os.path.join(OUT, f"traj_{task}.npz"))["curves"].astype(np.float32)
    m, cmu, csd, Ts, _ = vae_ref._load(task)
    # VAE 샘플
    gen = np.array([vae_ref._decode_curve(task, rng.standard_normal(vae_ref.ZDIM).astype(np.float32))
                    for _ in range(n_gen)], np.float32)
    print(f"\n===== {task} (real n={len(real)}, VAE n={n_gen}) =====")

    # (1) 재구성 충실도: real → encode → decode
    Xn = ((real - cmu) / csd).reshape(len(real), -1)
    with torch.no_grad():
        mu, lv = m.encode(torch.tensor(Xn))
        rec = m.decode(mu).numpy().reshape(-1, N, 4) * csd + cmu
    rmse = np.sqrt(((rec - real) ** 2).reshape(-1, 4).mean(0))
    print("재구성 RMSE(deg):", {c: round(float(r), 1) for c, r in zip(CH, rmse)})

    # (2) 주변분포 일치 (real vs VAE): mean±std + KS
    fr, fg = feats(real), feats(gen)
    print(f"{'특징':<13}{'REAL mean±sd':>16}{'VAE mean±sd':>16}{'KS':>7}{'p':>7}")
    for k in fr:
        ks, p = ks_2samp(fr[k], fg[k])
        print(f"{k:<13}{fr[k].mean():>8.0f}±{fr[k].std():<6.0f}{fg[k].mean():>8.0f}±{fg[k].std():<6.0f}"
              f"{ks:>7.2f}{p:>7.2f}")

    # (3) 다양성: 1-NN 거리(정규화 곡선). VAE→real, real→real(기준)
    def flat(c): return ((c - cmu) / csd).reshape(len(c), -1)
    R, G = flat(real), flat(gen)
    def nn_dist(A, B, self_=False):
        ds = []
        for i, a in enumerate(A):
            d = np.linalg.norm(B - a, axis=1)
            if self_: d[i] = np.inf
            ds.append(d.min())
        return np.array(ds)
    d_gr = nn_dist(G, R); d_rr = nn_dist(R, R, self_=True)
    print(f"1-NN 거리(정규화): VAE→real {d_gr.mean():.2f}±{d_gr.std():.2f} | "
          f"real→real {d_rr.mean():.2f}±{d_rr.std():.2f}  "
          f"(VAE가 real끼리 간격과 비슷=현실적·비복사)")
    print(f"   복사 의심(거의 0거리) 비율: {(d_gr < 0.1 * d_rr.mean()).mean():.2f}")

    # (4) 상관 보존
    def corr(f, a, b): return np.corrcoef(f[a], f[b])[0, 1]
    print(f"상관 corr(peak,elbow_peak): REAL {corr(fr,'peak_elev','elbow_peak'):+.2f} "
          f"VAE {corr(fg,'peak_elev','elbow_peak'):+.2f} | "
          f"corr(start,peak): REAL {corr(fr,'start_elev','peak_elev'):+.2f} "
          f"VAE {corr(fg,'start_elev','peak_elev'):+.2f}")

    # (5) 매끄러움
    print(f"매끄러움(2차차분 |Δ²elev|): REAL {smoothness(real):.2f} VAE {smoothness(gen):.2f}")

    # (6) 잠재 활성 차원(KL/dim)
    kld = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean(0).detach().numpy()
    print(f"잠재 KL/dim: {np.round(kld,2)}  활성차원(KL>0.05): {int((kld>0.05).sum())}/{vae_ref.ZDIM}")
    return real, gen, fr, fg


def main():
    res = {t: evaluate(t) for t in ("T1", "T2")}
    # Fig: 채널 포락선 real vs VAE
    ph = np.linspace(0, 1, N)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for r, t in enumerate(["T1", "T2"]):
        real, gen, _, _ = res[t]
        for c in range(4):
            ax = axes[r, c]
            for data, col, lab in [(real, "0.4", "REAL"), (gen, "tab:blue", "VAE")]:
                mn, sd = data[:, :, c].mean(0), data[:, :, c].std(0)
                ax.plot(ph, mn, color=col, lw=2, label=lab)
                ax.fill_between(ph, mn - sd, mn + sd, color=col, alpha=0.2)
            ax.set_title(f"{t}: {CH[c]} (mean±std)"); ax.grid(alpha=0.3)
            if r == 0 and c == 0: ax.legend()
            if r == 1: ax.set_xlabel("phase")
    fig.suptitle("VAE-generated vs REAL KIMHu kinematic curves (mean±std envelope per channel)", fontsize=13)
    fig.tight_layout(); p1 = os.path.join(REP, "eval_vae_curves.png"); fig.savefig(p1, dpi=110); plt.close(fig)

    # Fig: 주변분포 + 산점
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, t in enumerate(["T1", "T2"]):
        _, _, fr, fg = res[t]
        for c, k in enumerate(["peak_elev", "elbow_peak", "plane_mean"]):
            ax = axes[r, c]
            bins = np.linspace(min(fr[k].min(), fg[k].min()), max(fr[k].max(), fg[k].max()), 25)
            ax.hist(fr[k], bins, alpha=0.5, label="REAL", color="0.4", density=True)
            ax.hist(fg[k], bins, alpha=0.5, label="VAE", color="tab:blue", density=True)
            ax.set_title(f"{t}: {k}"); ax.grid(alpha=0.3)
            if r == 0 and c == 0: ax.legend()
    fig.suptitle("Marginal distributions: REAL vs VAE-generated", fontsize=13)
    fig.tight_layout(); p2 = os.path.join(REP, "eval_vae_dist.png"); fig.savefig(p2, dpi=110); plt.close(fig)
    print("\nsaved:", p1, p2)


if __name__ == "__main__":
    main()
