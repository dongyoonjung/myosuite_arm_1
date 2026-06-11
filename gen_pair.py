"""피험자 단위 페어 데이터: θ(s_F,s_L) 고정 → T1·T2 각각 독립 latent 롤아웃 → 공유 라벨.
KIN-only(pos+vel+err 저장; err는 회귀기에서 제외). 같은 피험자=같은 θ 강제.

사용: python gen_pair.py --model models/ppo_arm_M3v.zip --vecnorm ..._vec.pkl --n 8000 --out data/pair/train.npz
"""
import argparse
import os

import numpy as np

from custom_envs.arm_perturb_v0 import ArmPerturbEnv
from gen_seq_data import _step_feat, _posdof


def sample_theta(rng, nch, p_healthy=0.15, ks=(1, 2, 3), kw=(0.5, 0.3, 0.2)):
    """희소-k 약화 θ 샘플(에피소드=피험자 단위)."""
    sF, sL = np.ones(nch), np.ones(nch)
    if rng.random() < p_healthy:
        return sF, sL
    k = int(rng.choice(ks, p=np.array(kw) / np.sum(kw)))
    ch = rng.choice(nch, size=k, replace=False)
    sF[ch] = rng.uniform(0.05, 1.0, k)
    sL[ch] = rng.uniform(0.70, 1.20, k)
    return sF, sL


def rollout(venv, raw, model, task):
    """task를 강제해 1회 롤아웃 → (시계열, 길이, 공변량). θ는 호출 전 set_perturbation으로 고정."""
    raw.tasks = [task]; raw.task = task           # 과제 강제(reset이 재배정 안 함)
    obs = venv.reset()
    lat = raw.latent
    cov = [lat["T"], lat["peak"], lat["skew"], lat["plane"], lat["elbow_peak"],
           1.0 if task == "T2" else 0.0]
    steps, done = [], False
    while not done:
        steps.append(_step_feat(raw))             # step 전 기록(리셋오염 방지)
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, _ = venv.step(a); done = d[0]
    return np.array(steps, np.float32), len(steps), cov


def generate(model_path, vec_path, n=8000, seed=0, ref_gen="fpca",
             motor_noise=0.10, kin_noise=0.0087):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def mk():
        return ArmPerturbEnv(task="mix", latent_mode="sample", perturb=True, rsi=False,
                             seed=seed, motor_noise=motor_noise, ref_gen=ref_gen)
    venv = DummyVecEnv([mk]); venv = VecNormalize.load(vec_path, venv)
    venv.training = False; venv.norm_reward = False
    raw = venv.venv.envs[0]
    model = PPO.load(model_path, device="cpu")
    rng = np.random.default_rng(seed + 777)
    Tmax = raw.n_steps; nch = raw.n_perturb
    Cact = 63; n_pos = len(_posdof(raw)); n_err = len(raw.track_dof)
    Cfull = Cact + 2 * n_pos + n_err; keep = slice(Cact, Cfull)     # KIN만(act 드롭)
    nkin = Cfull - Cact

    def pad(arr, L):
        a = arr[:, keep]; out = np.zeros((Tmax, nkin), np.float32); out[:min(L, Tmax)] = a[:Tmax]; return out

    s1, s2, l1, l2, c1, c2, ysF, ysL = [], [], [], [], [], [], [], []
    for ep in range(n):
        sF, sL = sample_theta(rng, nch)                 # 피험자 θ
        raw.set_perturbation(sF, sL)                    # 두 시행에 동일 θ 고정
        a1, L1, cov1 = rollout(venv, raw, model, "T1")  # 독립 latent
        a2, L2, cov2 = rollout(venv, raw, model, "T2")  # 독립 latent, 같은 θ
        if kin_noise > 0:
            a1[:, Cact:] += rng.normal(0, kin_noise, a1[:, Cact:].shape)
            a2[:, Cact:] += rng.normal(0, kin_noise, a2[:, Cact:].shape)
        s1.append(pad(a1, L1)); s2.append(pad(a2, L2))
        l1.append(min(L1, Tmax)); l2.append(min(L2, Tmax))
        c1.append(cov1); c2.append(cov2); ysF.append(sF); ysL.append(sL)
        if (ep + 1) % 500 == 0:
            print(f"  {ep+1}/{n}")
    raw.set_perturbation(None)
    pd = _posdof(raw)
    names = [f"pos_{x}" for x in pd] + [f"vel_{x}" for x in pd] + [f"err_{x}" for x in raw.track_dof]
    return (np.array(s1), np.array(s2), np.array(l1, np.int32), np.array(l2, np.int32),
            np.array(c1, np.float32), np.array(c2, np.float32),
            np.array(ysF, np.float32), np.array(ysL, np.float32), np.array(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-gen", default="fpca"); ap.add_argument("--motor-noise", type=float, default=0.10)
    ap.add_argument("--kin-noise", type=float, default=0.0087)
    ap.add_argument("--out", default="data/pair/train.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    s1, s2, l1, l2, c1, c2, ysF, ysL, names = generate(
        a.model, a.vecnorm, n=a.n, seed=a.seed, ref_gen=a.ref_gen,
        motor_noise=a.motor_noise, kin_noise=a.kin_noise)
    np.savez_compressed(a.out, seq1=s1, seq2=s2, len1=l1, len2=l2, cov1=c1, cov2=c2,
                        y_sF=ysF, y_sL=ysL, channel_names=names)
    print(f"saved {a.out}: pair {s1.shape} x2, label {ysF.shape}, 채널 {len(names)}")


if __name__ == "__main__":
    main()
