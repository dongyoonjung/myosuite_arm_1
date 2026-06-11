"""페어(같은 피험자 T1·T2) → 공유 파라미터 회귀.
공유 TCN 인코더로 각 시행 임베딩 → 어텐션 풀링(순열불변) → head → θ.
입력 = 관측 운동학(관절각+속도; err 제외). 라벨 = 피험자당 1개 θ.

사용: python regress_pair.py --data data/pair/train.npz --epochs 120 --out models/regressor_pair.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn


def load(path, stride=2):
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["channel_names"]]
    keep = [i for i, n in enumerate(names) if not n.startswith("err_")]   # 참조 누수(err) 제외
    def prep(seq):
        seq = seq.astype(np.float32)[:, ::stride, :][:, :, keep]
        return seq
    s1, s2 = prep(d["seq1"]), prep(d["seq2"])
    l1 = np.ceil(d["len1"] / stride).astype(np.int32)
    l2 = np.ceil(d["len2"] / stride).astype(np.int32)
    c1, c2 = d["cov1"].astype(np.float32), d["cov2"].astype(np.float32)
    y = np.concatenate([d["y_sF"], d["y_sL"]], 1).astype(np.float32)
    print(f"  KIN {s1.shape[2]}채널(err 제외): {[names[i] for i in keep]}")
    return s1, s2, l1, l2, c1, c2, y


def stdz(seqs, lens):
    """두 시행 유효스텝 합산으로 채널 표준화 stats 산출."""
    flats = []
    for seq, ln in zip(seqs, lens):
        T = seq.shape[1]; m = np.arange(T)[None, :] < ln[:, None]; flats.append(seq[m])
    flat = np.concatenate(flats, 0)
    return flat.mean(0), flat.std(0) + 1e-6


def apply_std(seq, ln, mu, sd):
    T = seq.shape[1]; m = (np.arange(T)[None, :] < ln[:, None])
    out = (seq - mu) / sd; out[~m] = 0.0
    return out, m.astype(np.float32)


class Encoder(nn.Module):
    """dilated TCN + masked global pooling → 시행 임베딩(2*width)."""
    def __init__(self, c_in, width=64):
        super().__init__()
        def blk(ci, co, d, k=5, s=1):
            return nn.Sequential(nn.Conv1d(ci, co, k, stride=s, padding=(k - 1) * d // 2, dilation=d),
                                 nn.GroupNorm(8, co), nn.ReLU())
        self.down = blk(c_in, width, 1, k=7, s=2)
        self.c1 = blk(width, width, 2); self.c2 = blk(width, width * 2, 4)
        self.c3 = blk(width * 2, width * 2, 8)
        self.out_dim = width * 2 * 2

    def forward(self, x, mask):
        h = self.down(x); m = mask[:, ::2][:, :h.shape[2]]
        h = self.c1(h); h = self.c2(h); h = self.c3(h)
        m = m[:, :h.shape[2]].unsqueeze(1); msum = m.sum(2).clamp(min=1)
        avg = (h * m).sum(2) / msum
        mx = (h.masked_fill(m == 0, -1e9)).max(2).values
        return torch.cat([avg, mx], 1)


class PairNet(nn.Module):
    """공유 인코더 + per-시행 공변량 융합 + 어텐션 풀링 + head."""
    def __init__(self, c_in, n_cov, n_out=20, width=64, head=128, att=64):
        super().__init__()
        self.enc = Encoder(c_in, width)
        e = self.enc.out_dim + n_cov
        self.att_W = nn.Linear(e, att); self.att_w = nn.Linear(att, 1)
        self.head = nn.Sequential(
            nn.Linear(e, head), nn.LayerNorm(head), nn.ReLU(),
            nn.Linear(head, head), nn.LayerNorm(head), nn.ReLU(), nn.Linear(head, n_out))

    def embed(self, x, m, c):
        return torch.cat([self.enc(x, m), c], 1)            # (B, e)

    def forward(self, x1, m1, c1, x2, m2, c2):
        e1, e2 = self.embed(x1, m1, c1), self.embed(x2, m2, c2)
        E = torch.stack([e1, e2], 1)                        # (B,2,e)
        score = self.att_w(torch.tanh(self.att_W(E)))       # (B,2,1)
        alpha = torch.softmax(score, 1)
        g = (alpha * E).sum(1)                              # (B,e) 순열불변 가중합
        return self.head(g)


def train(data, stride=2, epochs=120, bs=256, lr=1e-3, seed=0, threads=16):
    torch.manual_seed(seed); torch.set_num_threads(threads)
    s1, s2, l1, l2, c1, c2, y = load(data, stride)
    mu, sd = stdz([s1, s2], [l1, l2])
    s1n, m1 = apply_std(s1, l1, mu, sd); s2n, m2 = apply_std(s2, l2, mu, sd)
    cstack = np.concatenate([c1, c2], 0); cmu, csd = cstack.mean(0), cstack.std(0) + 1e-6
    c1n, c2n = (c1 - cmu) / csd, (c2 - cmu) / csd

    n = len(y); idx = np.random.default_rng(seed).permutation(n); cut = int(0.85 * n)
    tr, va = idx[:cut], idx[cut:]
    T = lambda a: torch.tensor(a)
    def split(a, sel, tp=False):
        a = a[sel]; return T(a.transpose(0, 2, 1) if tp else a)
    Xtr1, Xv1 = split(s1n, tr, 1), split(s1n, va, 1); Xtr2, Xv2 = split(s2n, tr, 1), split(s2n, va, 1)
    Mtr1, Mv1 = T(m1[tr]), T(m1[va]); Mtr2, Mv2 = T(m2[tr]), T(m2[va])
    Ctr1, Cv1 = T(c1n[tr]), T(c1n[va]); Ctr2, Cv2 = T(c2n[tr]), T(c2n[va])
    ytr, yv = T(y[tr]), T(y[va])
    nch = y.shape[1] // 2
    w = 1.0 + 2.0 * (1.0 - ytr[:, :nch]).clamp(0, 1).max(1).values

    model = PairNet(Xtr1.shape[1], Ctr1.shape[1], n_out=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    huber = nn.SmoothL1Loss(reduction="none")
    best = 1e9; best_state = None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            pred = model(Xtr1[b], Mtr1[b], Ctr1[b], Xtr2[b], Mtr2[b], Ctr2[b])
            loss = (huber(pred, ytr[b]).mean(1) * w[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv1, Mv1, Cv1, Xv2, Mv2, Cv2)
            vl = huber(pv, yv).mean().item()
        if vl < best: best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0: print(f"  ep{ep+1} val {vl:.4f}")
    model.load_state_dict(best_state)
    return model, (Xv1, Mv1, Cv1, Xv2, Mv2, Cv2, yv)


def report(model, val):
    Xv1, Mv1, Cv1, Xv2, Mv2, Cv2, yv = val
    model.eval()
    with torch.no_grad():
        pv = model(Xv1, Mv1, Cv1, Xv2, Mv2, Cv2).numpy()
    yv = yv.numpy(); nch = yv.shape[1] // 2
    from custom_envs.muscle_groups import PERTURB_CHANNELS, PERTURB_CHANNELS_WRIST
    names = [c for c, _ in (PERTURB_CHANNELS_WRIST if nch == 12 else PERTURB_CHANNELS)]
    r2 = lambda t, p: 1 - ((t - p) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9)
    print("\n=== 페어(T1+T2 공유) 식별성 (s_F) — 관측 운동학만, err 제외 ===")
    print(f"{'채널':<12}{'MAE':>8}{'R²':>8}  해석")
    r2s = []
    for i, nm in enumerate(names):
        m = np.abs(yv[:, i] - pv[:, i]).mean(); rr = r2(yv[:, i], pv[:, i]); r2s.append(rr)
        v = "강식별" if rr > 0.6 else ("부분" if rr > 0.3 else "약식별")
        print(f"{nm:<12}{m:>8.3f}{rr:>8.2f}  {v}")
    print(f"  s_F 평균 R² {np.mean(r2s):.3f}")
    r2l = [r2(yv[:, nch + i], pv[:, nch + i]) for i in range(nch)]
    print(f"  s_L 평균 R² {np.mean(r2l):.3f}")
    return np.mean(r2s), np.mean(r2l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="models/regressor_pair.pt")
    a = ap.parse_args()
    print(f"=== 페어 회귀(T1+T2 공유 θ): epochs={a.epochs} ===")
    model, val = train(a.data, stride=a.stride, epochs=a.epochs, threads=a.threads)
    report(model, val)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state": model.state_dict()}, a.out)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
