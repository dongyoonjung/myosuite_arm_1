"""T1 full-sequence 6채널 fPCA 참조 생성기 (손목 해제 모드).

trajfull_T1(저점→정점→복귀, 6채널: elev·elbow·plane·prosup·wrist_flex·wrist_dev)에
fPCA + 점수공간 KDE. 손목까지 포함한 전체동작 참조를 생성. (T1만 — T2 손목은 불신.)
env 참조키 매핑: shoulder_elv←elev, elbow_flexion←elbow, elv_angle←plane,
  pro_sup←prosup, flexion←wrist_flex, deviation←wrist_dev.

학습: python -m references.fpca_full_ref
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out")
N = 80
NCH = 6
VAR_KEEP = 0.95
KDE_H = 0.35
_CACHE = {}
# trajfull 채널순 → env 참조키
KEY = ["shoulder_elv", "elbow_flexion", "elv_angle", "pro_sup", "flexion", "deviation"]


def train_task(task="T1"):
    d = np.load(os.path.join(OUT, f"trajfull_{task}.npz"), allow_pickle=True)
    cur = d["curves"].astype(np.float64)                  # (n,80,6)
    n = len(cur)
    cmu = cur.reshape(-1, NCH).mean(0); csd = cur.reshape(-1, NCH).std(0) + 1e-6
    X = ((cur - cmu) / csd).reshape(n, -1)
    mean = X.mean(0); Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (n - 1)
    K = int(np.searchsorted(np.cumsum(var) / var.sum(), VAR_KEEP) + 1)
    comp = Vt[:K]; lam = var[:K]; sstd = (Xc @ comp.T) / np.sqrt(lam)
    np.savez(os.path.join(OUT, f"fpcafull_{task}.npz"), mean=mean, comp=comp, lam=lam,
             sstd=sstd, cmu=cmu, csd=csd, T=d["T"].astype(np.float64))
    print(f"[{task}] full-seq fPCA K={K}/{n} (95%분산), 6채널(손목포함)")


def _load(task):
    if task not in _CACHE:
        d = np.load(os.path.join(OUT, f"fpcafull_{task}.npz"))
        _CACHE[task] = {k: d[k] for k in d.files}
    return _CACHE[task]


def _latent_from(task, z_std, T):
    p = _load(task)
    x = p["mean"] + (z_std * np.sqrt(p["lam"])) @ p["comp"]
    cur = (x.reshape(N, NCH) * p["csd"] + p["cmu"]).astype(np.float32)   # 단조화 안 함(복귀 포함)
    elev = cur[:, 0]
    return {"T": max(0.5, float(T)), "peak": float(elev.max()),
            "skew": 0.0, "plane": float(cur[:, 2].mean()),
            "elbow_peak": float(cur[:, 1].max()), "_curve": cur, "_z": z_std}


def sample_latent(task, rng, jitter=0.0):
    p = _load(task); sstd = p["sstd"]; Ts = p["T"]
    z = sstd[rng.integers(len(sstd))] + rng.normal(0, KDE_H, size=sstd.shape[1])
    T = float(rng.choice(Ts) * (1 + jitter * rng.standard_normal()))
    return _latent_from(task, z, T)


def median_latent(task="T1"):
    p = _load(task)
    return _latent_from(task, np.zeros(len(p["lam"])), float(np.median(p["T"])))


def make_reference(task, latent, n_steps, dt):
    cur = latent.get("_curve")
    if cur is None:
        cur = _latent_from(task, np.zeros(len(_load(task)["lam"])), 2.0)["_curve"]
    ph = np.linspace(0, 1, N)
    ph_t = np.clip(np.arange(n_steps) * dt / max(1e-3, float(latent["T"])), 0, 1)
    at = lambda c: np.interp(ph_t, ph, c)
    return {KEY[k]: at(cur[:, k]) for k in range(NCH)}


def _selftest():
    rng = np.random.default_rng(0)
    lat = sample_latent("T1", rng)
    ref = make_reference("T1", lat, 200, 0.02)
    print("키:", list(ref.keys()))
    for k in ("shoulder_elv", "flexion", "deviation"):
        v = ref[k]; print(f"  {k:13s} {v.min():.0f}~{v.max():.0f}°")


if __name__ == "__main__":
    for t in ("T1", "T2"):
        train_task(t)
    _selftest()
