"""G(seq) — 시계열 회귀기. 매 스텝 (ACT63+pos5+vel5+err4)=77채널 시퀀스 → 근육 파라미터 20.

elbow 갱신 결론: 시퀀스가 요약통계보다 훨씬 많은 식별 정보를 포착(상승 동역학·떨림·
정착 타이밍·이탈 시점). 1D-TCN(dilated temporal conv)+masked pooling으로 정식화.

모델: [stride-2 다운샘플 conv] → dilated conv ×3 (GroupNorm·ReLU) → masked avg+max pool
      → known 공변량(6) concat → MLP head → 20(s_F·s_L).

사용:
  OMP_NUM_THREADS=16 .venv/bin/python regress_seq.py --data data/seq/train.npz \
      [--channels all|act|kin] [--epochs 80] [--stride 2]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from custom_envs import muscle_groups as MG

CH = [c for c, _ in MG.PERTURB_CHANNELS]
# 채널 블록: act63 pos5 vel5 err4
BLK = {"act": slice(0, 63), "pos": slice(63, 68), "vel": slice(68, 73), "err": slice(73, 77)}
# 표면 EMG 접근 가능근(actuator idx) — 심부근(회전근개 SUPSP/INFSP/SUBSC/TMIN, CORB,
# TRImed, SUP, BRA) 제외. 현실적 제한 관측 시나리오.
SURF_ACT = [0, 1, 2, 8, 9, 10, 11, 12, 13, 15, 16, 20, 21, 23]   # DELT/PECM/LAT/TRIlong·lat/BIC/BRD


def select_channels(seq, channels):
    """channels: 'all'|'act'|'kin'|'surf'(표면EMG)|'surf+kin'(현실적 제한관측) 등."""
    if channels == "all":
        return seq
    parts = []
    for p in channels.split("+"):
        if p == "kin":
            parts.append(seq[:, :, 63:77])
        elif p == "surf":
            parts.append(seq[:, :, SURF_ACT])
        else:
            parts.append(seq[:, :, BLK[p]])
    return np.concatenate(parts, axis=2)


class TCN(nn.Module):
    """dilated 1D-TCN + masked global pooling + 공변량 융합 head."""
    def __init__(self, c_in, n_cov, n_out=20, width=64, head=128):
        super().__init__()
        def blk(ci, co, d, k=5, stride=1):
            pad = (k - 1) * d // 2
            return nn.Sequential(nn.Conv1d(ci, co, k, stride=stride, padding=pad, dilation=d),
                                 nn.GroupNorm(8, co), nn.ReLU())
        self.down = blk(c_in, width, 1, k=7, stride=2)      # 시간 다운샘플(연산↓)
        self.c1 = blk(width, width, 2)
        self.c2 = blk(width, width * 2, 4)
        self.c3 = blk(width * 2, width * 2, 8)
        self.head = nn.Sequential(
            nn.Linear(width * 2 * 2 + n_cov, head), nn.LayerNorm(head), nn.ReLU(),
            nn.Linear(head, head), nn.LayerNorm(head), nn.ReLU(),
            nn.Linear(head, n_out))

    def forward(self, x, mask, cov):
        # x:(B,C,T)  mask:(B,T) 1=valid
        h = self.down(x)
        m = mask[:, ::2][:, :h.shape[2]]                    # stride-2에 맞춰 마스크 축소
        h = self.c1(h); h = self.c2(h); h = self.c3(h)
        m = m[:, :h.shape[2]].unsqueeze(1)                  # (B,1,T')
        msum = m.sum(2).clamp(min=1)
        avg = (h * m).sum(2) / msum                         # masked mean
        mx = (h.masked_fill(m == 0, -1e9)).max(2).values    # masked max
        z = torch.cat([avg, mx, cov], dim=1)
        return self.head(z)


def load(path, channels="all", stride=2, drop_err=False):
    d = np.load(path, allow_pickle=True)
    seq = d["seq"].astype(np.float32)                       # (N,T,C)
    names = [str(n) for n in d["channel_names"]] if "channel_names" in d else None
    if stride > 1:
        seq = seq[:, ::stride, :]
    seq = select_channels(seq, channels)
    if drop_err and names is not None and channels == "all" and seq.shape[2] == len(names):
        keep = [i for i, n in enumerate(names) if not n.startswith("err_")]   # 추적오차 제외(참조 누수 차단)
        seq = seq[:, :, keep]
        print(f"  [drop_err] err 채널 제외 → KIN {seq.shape[2]}채널({[names[i] for i in keep]})")
    length = np.ceil(d["length"].astype(np.float32) / stride).astype(np.int32)
    cov = d["cov"].astype(np.float32)
    y = np.concatenate([d["y_sF"], d["y_sL"]], 1).astype(np.float32)
    return seq, length, cov, y


def standardize(seq, length, stats=None):
    """유효 스텝만으로 채널별 표준화 + 공변량 표준화. stats 주어지면 재사용."""
    N, T, C = seq.shape
    mask = (np.arange(T)[None, :] < length[:, None])
    if stats is None:
        flat = seq[mask]                                    # (유효스텝합, C)
        mu = flat.mean(0); sd = flat.std(0) + 1e-6
    else:
        mu, sd = stats
    seqn = (seq - mu) / sd
    seqn[~mask] = 0.0                                       # 패딩 0
    return seqn, mask.astype(np.float32), (mu, sd)


def train(data, channels="all", stride=2, epochs=80, bs=256, lr=1e-3,
          reweight=True, seed=0, threads=16, drop_err=False):
    torch.manual_seed(seed); torch.set_num_threads(threads)
    seq, length, cov, y = load(data, channels, stride, drop_err=drop_err)
    seqn, mask, seq_stats = standardize(seq, length)
    cmu, csd = cov.mean(0), cov.std(0) + 1e-6
    covn = (cov - cmu) / csd

    n = len(seqn); idx = np.random.default_rng(seed).permutation(n)
    cut = int(0.85 * n); tr, va = idx[:cut], idx[cut:]
    to = lambda a: torch.tensor(a)
    Xtr = to(seqn[tr].transpose(0, 2, 1)); Xv = to(seqn[va].transpose(0, 2, 1))  # (B,C,T)
    Mtr, Mv = to(mask[tr]), to(mask[va])
    Ctr, Cv = to(covn[tr]), to(covn[va])
    ytr, yv = to(y[tr]), to(y[va])

    w = (1.0 + 2.0 * (1.0 - ytr[:, :10]).clamp(0, 1).max(1).values) if reweight else torch.ones(len(ytr))

    model = TCN(Xtr.shape[1], Ctr.shape[1], n_out=ytr.shape[1])   # n_out=2×섭동채널(10→20, 손목12→24)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.SmoothL1Loss(reduction="none")
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            pred = model(Xtr[b], Mtr[b], Ctr[b])
            l = (lossf(pred, ytr[b]).mean(1) * w[b]).mean()
            opt.zero_grad(); l.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                mae = (model(Xv, Mv, Cv) - yv).abs().mean().item()
            if mae < best:
                best = mae; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep+1}: val MAE {mae:.4f}")
    model.load_state_dict(best_state)
    return model, (Xv, Mv, Cv, yv), (seq_stats, (cmu, csd))


def report(model, val):
    Xv, Mv, Cv, yv = val
    model.eval()
    with torch.no_grad():
        pv = model(Xv, Mv, Cv).numpy()
    yv = yv.numpy()
    nch = yv.shape[1] // 2                          # 섭동 채널수(10 또는 손목12)
    from custom_envs.muscle_groups import PERTURB_CHANNELS, PERTURB_CHANNELS_WRIST
    names = [c for c, _ in (PERTURB_CHANNELS_WRIST if nch == 12 else PERTURB_CHANNELS)]
    def r2(t, p):
        return 1 - ((t - p) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9)
    print("\n=== 채널별 식별성 (s_F: Fmax scale) — 시계열 회귀 ===")
    print(f"{'채널':<12}{'MAE':>8}{'R²':>8}  해석")
    r2s = []
    for i, n in enumerate(names):
        m = np.abs(yv[:, i] - pv[:, i]).mean(); rr = r2(yv[:, i], pv[:, i]); r2s.append(rr)
        v = "강식별" if rr > 0.6 else ("부분" if rr > 0.3 else "약식별")
        print(f"{n:<12}{m:>8.3f}{rr:>8.2f}  {v}")
    print(f"  s_F 평균 R² {np.mean(r2s):.3f}")
    print("=== s_L (Lopt scale) ===")
    r2l = [r2(yv[:, nch + i], pv[:, nch + i]) for i in range(nch)]
    for i, n in enumerate(names):
        print(f"{n:<12}{np.abs(yv[:,nch+i]-pv[:,nch+i]).mean():>8.3f}{r2l[i]:>8.2f}")
    print(f"  s_L 평균 R² {np.mean(r2l):.3f}")
    return np.mean(r2s), np.mean(r2l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--channels", default="all")     # all|act|kin|act+err
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="models/regressor_seqG.pt")
    ap.add_argument("--no-err", action="store_true")   # 추적오차 채널 제외(참조 누수 차단)
    a = ap.parse_args()
    print(f"=== G(seq) TCN: channels={a.channels} stride={a.stride} no_err={a.no_err} ===")
    model, val, stats = train(a.data, channels=a.channels, stride=a.stride,
                              epochs=a.epochs, threads=a.threads, drop_err=a.no_err)
    report(model, val)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": model.state_dict()}, a.out)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
