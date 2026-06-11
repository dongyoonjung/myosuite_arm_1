"""M4(seq) — 시계열 회귀용 데이터. 매 스텝 (ACT 63 + 관절각5 + 속도5 + 추종오차4)=77채널.

elbow 결론 갱신: 요약통계(mean/max)보다 *시퀀스*가 훨씬 많은 식별 정보를 담는다
(상승 동역학·떨림·정착 타이밍·이탈 시점 등). 그래서 시계열로 정식화.

저장(샤드): seq (N,Tmax,77) float32, length (N,), cov (N,6), y_sF (N,10), y_sL (N,10).
병합: tools/concat_seq.py

사용(병렬 샤드):
  python gen_seq_data.py --model models/ppo_arm_M3b.zip --vecnorm ..._vec.pkl \
      --task mix --n 1000 --seed i --out data/seq/shard_i.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_envs.arm_perturb_v0 import ArmPerturbEnv
from custom_envs import muscle_groups as MG

TRACK = MG.TRACK_DOF                       # [shoulder_elv, elv_angle, elbow_flexion, pro_sup]
C_ACT, C_POS, C_VEL, C_ERR = 63, 5, 5, 4   # 채널 블록
C = C_ACT + C_POS + C_VEL + C_ERR          # = 77
POS_DOF = ["shoulder_elv", "elv_angle", "shoulder_rot", "elbow_flexion", "pro_sup"]


def _posdof(raw):
    return list(raw.track_dof) + ["shoulder_rot"]


def _step_feat(raw):
    """현 시점 특징(act63 + pos + vel + err). track_dof 기반(손목 모드 자동 확장).
    step 전 호출(리셋오염 방지)."""
    i = min(raw.t_idx, raw.n_steps - 1)
    act = raw.data.act.copy()                                  # 63 privileged
    pos = np.array([raw.data.qpos[raw.track_qadr[n]] if n in raw.track_qadr
                    else raw.data.qpos[raw.rot_qadr] for n in _posdof(raw)])
    vel = np.array([raw.data.qvel[raw.track_dadr[n]] if n in raw.track_dadr
                    else raw.data.qvel[raw.rot_dadr] for n in _posdof(raw)])
    err = np.array([raw.data.qpos[raw.track_qadr[n]] - raw.ref[n][i] for n in raw.track_dof])
    return np.concatenate([act, pos, vel, err]).astype(np.float32)


def generate(model_path, vec_path, task="mix", n=1000, seed=0,
             curriculum_k=(0, 1, 2, 3), curriculum_w=(0.15, 0.45, 0.25, 0.15),
             motor_noise=0.0, emg_noise=0.0, kin_noise=0.0, ref_gen="warp", kin_only=True,
             wrist=False, horizon_s=3.5):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ks = [k for k in curriculum_k if k >= 1] or [1]
    kw = [w for k, w in zip(curriculum_k, curriculum_w) if k >= 1]
    p_healthy = sum(w for k, w in zip(curriculum_k, curriculum_w) if k == 0)

    def mk():
        return ArmPerturbEnv(task=task, latent_mode="sample", perturb=True, rsi=False,
                             seed=seed, curriculum_k=tuple(ks), curriculum_k_weights=tuple(kw),
                             motor_noise=motor_noise, ref_gen=ref_gen, wrist=wrist,
                             horizon_s=horizon_s)
    venv = DummyVecEnv([mk])
    venv = VecNormalize.load(vec_path, venv); venv.training = False; venv.norm_reward = False
    raw = venv.venv.envs[0]
    model = PPO.load(model_path, device="cpu")
    rng = np.random.default_rng(seed + 12345)
    Tmax = raw.n_steps
    # 채널 레이아웃을 env의 track_dof에서 동적 계산
    n_pos = len(_posdof(raw)); n_err = len(raw.track_dof)
    Cact = 63; Cfull = Cact + 2 * n_pos + n_err
    keep = slice(Cact, Cfull) if kin_only else slice(0, Cfull)
    nch = (Cfull - Cact) if kin_only else Cfull

    seqs, lens, covs, ysF, ysL = [], [], [], [], []
    for ep in range(n):
        if rng.random() < p_healthy:
            raw.set_perturbation(np.ones(raw.n_perturb), np.ones(raw.n_perturb))
        else:
            raw.set_perturbation(None)
        obs = venv.reset()
        sF = raw.s_F.copy(); sL = raw.s_L.copy(); lat = raw.latent
        cov = [lat["T"], lat["peak"], lat["skew"], lat["plane"], lat["elbow_peak"],
               1.0 if raw.task == "T2" else 0.0]
        steps = []
        done = False
        while not done:
            steps.append(_step_feat(raw))           # step 전 기록
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, _ = venv.step(a); done = d[0]
        arr = np.array(steps, np.float32)            # (L, 77)
        if emg_noise > 0 and not kin_only:           # EMG 측정노이즈(가법, ACT 유지 시만)
            arr[:, :C_ACT] += rng.normal(0, emg_noise, arr[:, :C_ACT].shape)
        if kin_noise > 0:                            # 운동학 센서노이즈(rad)
            arr[:, C_ACT:] += rng.normal(0, kin_noise, arr[:, C_ACT:].shape)
        arr = arr[:, keep]                           # KIN-only 시 ACT 드롭
        L = len(arr)
        pad = np.zeros((Tmax, nch), np.float32)
        pad[:L] = arr[:Tmax]
        seqs.append(pad); lens.append(min(L, Tmax))
        covs.append(cov); ysF.append(sF); ysL.append(sL)
        if (ep + 1) % 250 == 0:
            print(f"  {ep+1}/{n}")
    raw.set_perturbation(None)
    pd = _posdof(raw)
    kin_names = ([f"pos_{n}" for n in pd] + [f"vel_{n}" for n in pd]
                 + [f"err_{n}" for n in raw.track_dof])
    names = kin_names if kin_only else [f"act_{i}" for i in range(63)] + kin_names
    return (np.array(seqs, np.float32), np.array(lens, np.int32),
            np.array(covs, np.float32), np.array(ysF, np.float32), np.array(ysL, np.float32),
            np.array(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--task", default="mix"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motor-noise", type=float, default=0.0)   # 운동노이즈 σ_sd
    ap.add_argument("--emg-noise", type=float, default=0.0)     # EMG 측정노이즈
    ap.add_argument("--kin-noise", type=float, default=0.0)     # 운동학 센서노이즈(rad)
    ap.add_argument("--ref-gen", default="warp")                # warp | vae | fpca
    ap.add_argument("--kin-only", type=int, default=1)          # 1=KIN만 저장(ACT 드롭)
    ap.add_argument("--wrist", action="store_true")             # 손목 해제(T1 full-seq)
    ap.add_argument("--horizon", type=float, default=3.5)
    ap.add_argument("--out", default="data/seq/shard.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    seq, ln, cov, ysF, ysL, names = generate(
        a.model, a.vecnorm, task=a.task, n=a.n, seed=a.seed,
        motor_noise=a.motor_noise, emg_noise=a.emg_noise, kin_noise=a.kin_noise,
        ref_gen=a.ref_gen, kin_only=bool(a.kin_only), wrist=a.wrist, horizon_s=a.horizon)
    blocks = [len(names)]
    np.savez_compressed(a.out, seq=seq, length=ln, cov=cov, y_sF=ysF, y_sL=ysL,
                        channel_names=np.array(names), chan_blocks=np.array(blocks))
    print(f"saved {a.out}: seq{seq.shape} len[{ln.min()},{ln.max()}] "
          f"약화비율 {(ysF.min(1) < 0.999).mean():.2f}")


if __name__ == "__main__":
    main()
