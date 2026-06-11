"""KIMHu 궤적 참조 생성 — functional PCA + 점수공간 KDE (ProMP 계열).

VAE 대비 우위: 경험적 평균+공분산을 *구조적으로* 그대로 포착(under-dispersion 없음),
채널간 상관 보존, 소량 데이터(~200 reps)에 강건(DL 과적합/잠재붕괴 없음).
방법: 정규화 곡선(50×4=200)에 PCA → 상위 K모드(95%분산). 생성은 점수공간에서
경험적 점수 부트스트랩 + 소폭 jitter(=KDE) → 실제 분포 지지·다양성·비복사.

drop-in 인터페이스(warp/vae_ref와 동일): sample_latent / make_reference / median_latent.
학습: python -m references.fpca_ref
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out")
N = 50
VAR_KEEP = 0.95
KDE_H = 0.35            # 점수공간 KDE 대역폭(표준화 점수 단위)
_CACHE = {}


def train_task(task):
    d = np.load(os.path.join(OUT, f"traj_{task}.npz"), allow_pickle=True)
    cur = d["curves"].astype(np.float64)                  # (n,50,4)
    n = len(cur)
    cmu = cur.reshape(-1, 4).mean(0); csd = cur.reshape(-1, 4).std(0) + 1e-6
    X = ((cur - cmu) / csd).reshape(n, -1)                # (n,200) 정규화 flatten
    mean = X.mean(0); Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (n - 1)
    K = int(np.searchsorted(np.cumsum(var) / var.sum(), VAR_KEEP) + 1)
    comp = Vt[:K]; lam = var[:K]                          # (K,200),(K,)
    scores = Xc @ comp.T                                  # (n,K)
    sstd = scores / np.sqrt(lam)                          # 표준화 점수(~단위분산)
    np.savez(os.path.join(OUT, f"fpca_{task}.npz"), mean=mean, comp=comp, lam=lam,
             sstd=sstd, cmu=cmu, csd=csd, T=d["T"].astype(np.float64))
    ev = var.sum()
    print(f"[{task}] fPCA K={K}/{n} (95%분산) reps={n}  설명분산 {np.cumsum(var)[K-1]/ev*100:.0f}%")


def _load(task):
    if task not in _CACHE:
        d = np.load(os.path.join(OUT, f"fpca_{task}.npz"))
        _CACHE[task] = {k: d[k] for k in d.files}
    return _CACHE[task]


def _decode(task, z_std):
    p = _load(task)
    x = p["mean"] + (z_std * np.sqrt(p["lam"])) @ p["comp"]    # (200,)
    cur = (x.reshape(N, 4) * p["csd"] + p["cmu"]).astype(np.float32)
    cur[:, 0] = np.maximum.accumulate(cur[:, 0])              # 거상 단조화
    return cur


def _latent_from(task, z_std, T):
    cur = _decode(task, z_std)
    elev = cur[:, 0]; half = elev[0] + 0.5 * (elev[-1] - elev[0])
    skew = float(np.searchsorted(elev, half) / N - 0.5)
    return {"T": max(0.5, float(T)), "peak": float(elev.max()), "skew": skew,
            "plane": float(cur[:, 2].mean()), "elbow_peak": float(cur[:, 1].max()),
            "_curve": cur, "_z": z_std}


def sample_latent(task, rng, jitter=0.0):
    """점수공간 KDE: 실제 rep 점수 부트스트랩 + jitter → 분포 지지·다양성·비복사."""
    p = _load(task); sstd = p["sstd"]; Ts = p["T"]
    base = sstd[rng.integers(len(sstd))]                      # 실제 rep의 점수
    z = base + rng.normal(0, KDE_H, size=base.shape)          # KDE jitter
    T = float(rng.choice(Ts) * (1 + jitter * rng.standard_normal()))
    return _latent_from(task, z, T)


def median_latent(task):
    p = _load(task)
    return _latent_from(task, np.zeros(len(p["lam"])), float(np.median(p["T"])))


def make_reference(task, latent, n_steps, dt):
    cur = latent.get("_curve")
    if cur is None:
        cur = _decode(task, np.zeros(len(_load(task)["lam"])))
    ph = np.linspace(0, 1, N)
    ph_t = np.clip(np.arange(n_steps) * dt / max(1e-3, float(latent["T"])), 0, 1)
    at = lambda c: np.interp(ph_t, ph, c)
    return {"shoulder_elv": at(cur[:, 0]), "elbow_flexion": at(cur[:, 1]),
            "elv_angle": at(cur[:, 2]), "pro_sup": at(cur[:, 3])}


def _selftest():
    rng = np.random.default_rng(0)
    for task in ("T1", "T2"):
        peaks = [sample_latent(task, rng)["peak"] for _ in range(300)]
        print(f"[{task}] fPCA 샘플 정점 {np.mean(peaks):.0f}±{np.std(peaks):.0f}° "
              f"(VAE는 폭 좁았음; 실제 sd에 근접 기대)")


if __name__ == "__main__":
    for t in ("T1", "T2"):
        train_task(t)
    _selftest()
