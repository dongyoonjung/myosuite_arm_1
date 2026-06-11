"""G — 식별 회귀기. M4 특징(ACT 주채널+KIN 보조+known latent) → 근육 파라미터(s_F,s_L).

DESIGN: MLP-FUSED(팔꿈치 계승). 성공 기준 = 채널별 식별성(R²/MAE)을 정직하게 보고
  — 천장이 낮으면(약식별 cuff 등) 그게 측정 결과(끌어올릴 손잡이 아님).
약화정도별 재가중(중증=큰 이탈, 이질적 난이도) 옵션.

사용:
  .venv/bin/python regress_nn.py --data data/sim/train.npz [--val data/sim/val.npz]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from custom_envs import muscle_groups as MG

CH = [c for c, _ in MG.PERTURB_CHANNELS]


class FusedMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(256, 256)):
        super().__init__()
        layers = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 특징 레이아웃: act_mean(63) act_max(63) kin(11) covariate(6) = 143
FEAT_SLICES = {"act": slice(0, 126), "kin": slice(126, 137), "cov": slice(137, 143)}


def _select(X, feats):
    """feats: 'all' | 'act' | 'kin' | 'actkin' | 'act+cov' 등(+로 조합)."""
    if feats == "all":
        return X
    cols = []
    for part in feats.split("+"):
        if part == "actkin":
            cols += [FEAT_SLICES["act"], FEAT_SLICES["kin"]]
        else:
            cols.append(FEAT_SLICES[part])
    return np.concatenate([X[:, s] for s in cols], axis=1)


def load(path, feats="all"):
    d = np.load(path, allow_pickle=True)
    X = _select(d["X"].astype(np.float32), feats)
    y = np.concatenate([d["y_sF"], d["y_sL"]], 1).astype(np.float32)   # (N,20)
    return X, y, d["feature_names"], d


def train(data, val=None, epochs=200, bs=512, lr=1e-3, reweight=True,
          device="cpu", seed=0, feats="all"):
    torch.manual_seed(seed)
    X, y, feat_names, _ = load(data, feats)
    # 표준화
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    n = len(Xn); idx = np.random.default_rng(seed).permutation(n)
    if val:
        Xtr, ytr = Xn, y
        Xv, yv, _, _ = load(val, feats); Xv = (Xv - mu) / sd
    else:
        cut = int(0.85 * n)
        tr, va = idx[:cut], idx[cut:]
        Xtr, ytr, Xv, yv = Xn[tr], y[tr], Xn[va], y[va]

    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr)
    Xv = torch.tensor(Xv); yv = torch.tensor(yv)
    # 재가중: s_F가 낮을수록(중증) 가중↑ (per-sample, 표적채널 기준)
    if reweight:
        sev = (1.0 - ytr[:, :10]).clamp(0, 1).max(1).values    # 최대 약화 정도
        w = 1.0 + 2.0 * sev
    else:
        w = torch.ones(len(ytr))

    model = FusedMLP(Xtr.shape[1], ytr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.SmoothL1Loss(reduction="none")

    best = 1e9; best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            pred = model(Xtr[b])
            l = (lossf(pred, ytr[b]).mean(1) * w[b]).mean()
            opt.zero_grad(); l.backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(Xv)
                mae = (pv - yv).abs().mean().item()
            if mae < best:
                best = mae; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep+1}: val MAE {mae:.4f}")
    model.load_state_dict(best_state)
    return model, (Xv, yv), (mu, sd)


def report(model, Xv, yv):
    model.eval()
    with torch.no_grad():
        pv = model(Xv).numpy()
    yv = yv.numpy()
    print("\n=== 채널별 식별성 (s_F: Fmax scale 복원) ===")
    print(f"{'채널':<12}{'MAE':>8}{'R²':>8}  해석")
    for i, name in enumerate(CH):
        t, p = yv[:, i], pv[:, i]
        mae = np.abs(t - p).mean()
        ss_res = ((t - p) ** 2).sum(); ss_tot = ((t - t.mean()) ** 2).sum() + 1e-9
        r2 = 1 - ss_res / ss_tot
        verdict = "강식별" if r2 > 0.6 else ("부분" if r2 > 0.3 else "약식별")
        print(f"{name:<12}{mae:>8.3f}{r2:>8.2f}  {verdict}")
    print("\n=== s_L (Lopt scale) 복원 ===")
    for i, name in enumerate(CH):
        t, p = yv[:, 10 + i], pv[:, 10 + i]
        mae = np.abs(t - p).mean()
        ss_res = ((t - p) ** 2).sum(); ss_tot = ((t - t.mean()) ** 2).sum() + 1e-9
        r2 = 1 - ss_res / ss_tot
        print(f"{name:<12}{mae:>8.3f}{r2:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--no-reweight", action="store_true")
    ap.add_argument("--features", default="all")   # all|act|kin|actkin|cov|act+cov
    ap.add_argument("--out", default="models/regressor_G.pt")
    a = ap.parse_args()
    print(f"=== G 회귀: features={a.features} ===")
    model, (Xv, yv), (mu, sd) = train(a.data, val=a.val, epochs=a.epochs,
                                      reweight=not a.no_reweight, feats=a.features)
    report(model, Xv, yv)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, a.out)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
