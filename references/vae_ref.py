"""KIMHu 궤적 VAE 생성모델 — 참조(tracking 대상) 생성기.

기존 warp.py(평균 템플릿 + 1 PC + 4-latent 워핑)의 대체: KIMHu 모든 반복의 다채널
상승 궤적(elev/elbow/plane/prosup)을 VAE로 학습 → z~N(0,I)에서 *다양하고 현실적인*
전체 궤적을 샘플. 변동이 저차원·1모드가 아니라 데이터가 말하는 전 분포.

drop-in 인터페이스(warp와 동일): sample_latent(task,rng) / make_reference(task,latent,n_steps,dt).
  - sample_latent: z 샘플 → 곡선 디코드 → (T·peak·skew·plane·elbow_peak) 유도 + 곡선 stash.
  - make_reference: stash된 곡선을 단조화·시간워핑 → 4채널(deg) 궤적.

학습: python -m references.vae_ref   (references/out/traj_{T1,T2}.npz 필요)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

OUT = os.path.join(os.path.dirname(__file__), "out")
N = 50
CHAN = ["elev", "elbow", "plane", "prosup"]
ZDIM = 6
_CACHE = {}


class VAE(nn.Module):
    def __init__(self, d_in=N * 4, z=ZDIM, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, z); self.lv = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, d_in))

    def encode(self, x):
        h = self.enc(x); return self.mu(h), self.lv(h)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv


def train_task(task, epochs=1500, beta=0.5, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    d = np.load(os.path.join(OUT, f"traj_{task}.npz"), allow_pickle=True)
    cur = d["curves"].astype(np.float32)               # (n,N,4) deg
    T = d["T"].astype(np.float32)
    n = len(cur)
    cmu = cur.reshape(-1, 4).mean(0); csd = cur.reshape(-1, 4).std(0) + 1e-6
    Xn = ((cur - cmu) / csd).reshape(n, -1)            # 정규화 flatten (n,200)
    X = torch.tensor(Xn)
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        model.train()
        recon, mu, lv = model(X)
        rl = ((recon - X) ** 2).mean()
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        loss = rl + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 500 == 0:
            print(f"  [{task}] ep{ep+1} recon {rl.item():.3f} kl {kl.item():.3f}")
    torch.save({"state": model.state_dict(), "cmu": cmu, "csd": csd,
                "T": T, "peak": d["peak"]}, os.path.join(OUT, f"vae_{task}.pt"))
    print(f"[{task}] VAE 학습완료 n={n}")


def _load(task):
    if task not in _CACHE:
        ck = torch.load(os.path.join(OUT, f"vae_{task}.pt"), weights_only=False)
        m = VAE(); m.load_state_dict(ck["state"]); m.eval()
        _CACHE[task] = (m, ck["cmu"], ck["csd"], ck["T"], ck["peak"])
    return _CACHE[task]


def _decode_curve(task, z):
    m, cmu, csd, _, _ = _load(task)
    with torch.no_grad():
        out = m.decode(torch.tensor(z, dtype=torch.float32)[None])[0].numpy()
    cur = out.reshape(N, 4) * csd + cmu                # 역정규화 (N,4) deg
    cur[:, 0] = np.maximum.accumulate(cur[:, 0])       # 거상 단조화
    return cur


def _latent_from(task, z, T):
    cur = _decode_curve(task, z)
    elev = cur[:, 0]
    half = elev[0] + 0.5 * (elev[-1] - elev[0])
    skew = float(np.searchsorted(elev, half) / N - 0.5)     # 형태 비대칭(반높이 위상)
    return {"T": max(0.5, float(T)), "peak": float(elev.max()),
            "skew": skew, "plane": float(cur[:, 2].mean()),
            "elbow_peak": float(cur[:, 1].max()), "_curve": cur, "_z": z}


def sample_latent(task, rng, jitter=0.0):
    """z 샘플 → 곡선 디코드 → 표준 latent 키 유도(+곡선 stash)."""
    Ts = _load(task)[3]
    z = rng.standard_normal(ZDIM).astype(np.float32)
    T = float(rng.choice(Ts) * (1 + jitter * rng.standard_normal()))
    return _latent_from(task, z, T)


def median_latent(task):
    """z=0(VAE 평균 궤적) + 중앙 T — env fixed/eval 모드용."""
    Ts = _load(task)[3]
    return _latent_from(task, np.zeros(ZDIM, np.float32), float(np.median(Ts)))


def make_reference(task, latent, n_steps, dt):
    """stash된 곡선을 시간워핑 → 4채널(deg) 궤적(raise-and-hold)."""
    cur = latent.get("_curve")
    if cur is None:                                    # latent만 주어진 경우 z=0 곡선
        cur = _decode_curve(task, np.zeros(ZDIM, np.float32))
    ph = np.linspace(0, 1, N)
    tt = np.arange(n_steps) * dt
    ph_t = np.clip(tt / max(1e-3, float(latent["T"])), 0.0, 1.0)
    def at(c):
        return np.interp(ph_t, ph, c)
    return {"shoulder_elv": at(cur[:, 0]), "elbow_flexion": at(cur[:, 1]),
            "elv_angle": at(cur[:, 2]), "pro_sup": at(cur[:, 3])}


def _selftest():
    rng = np.random.default_rng(0)
    for task in ("T1", "T2"):
        lat = sample_latent(task, rng)
        ref = make_reference(task, lat, 175, 0.02)
        e = ref["shoulder_elv"]
        assert np.all(np.diff(e) >= -1e-5), "거상 비단조"
        print(f"[{task}] z-sample: elev {e[0]:.0f}→{e.max():.0f}° T={lat['T']:.2f} "
              f"plane {ref['elv_angle'].mean():+.0f}° elbow→{ref['elbow_flexion'].max():.0f}°")


if __name__ == "__main__":
    for task in ("T1", "T2"):
        train_task(task)
    _selftest()
