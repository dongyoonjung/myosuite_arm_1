"""M4 — 섭동 정책 스윕 롤아웃 → 회귀기(G) 학습 특징·라벨 생성.

각 에피소드: env가 (latent, 섭동 s_F/s_L)를 샘플 → M3 정책 롤아웃 → 특징 추출.
  특징 X = privileged 전체근육(63) 활성 요약(주채널) + 운동학 요약(보조)
           + known 공변량(latent 5 + task 1).
  라벨 y = s_F(10) + s_L(10)  (per-muscle Fmax/Lopt scale = 추정 표적).
DESIGN G: "특징=ACT 주채널+KIN 보조+known latent 공변량".

사용:
  .venv/bin/python gen_data.py --model models/ppo_arm_M3b.zip \
      --vecnorm models/ppo_arm_M3b_vec.pkl --n 20000 --out data/sim/train.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_envs.arm_perturb_v0 import ArmPerturbEnv
from custom_envs import muscle_groups as MG

NU = 63  # 전체 근육


def _feature_names():
    names = []
    import mujoco
    e = ArmPerturbEnv(task="T1", perturb=False)
    for stat in ("mean", "max"):
        for a in range(NU):
            names.append(f"act_{stat}_{mujoco.mj_id2name(e.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)}")
    names += ["elv_peak", "elv_final", "elv_rms_ref", "elb_peak", "elb_final",
              "plane_final", "plane_range", "rot_final", "rot_range", "prosup_range",
              "term_frac"]
    names += ["lat_T", "lat_peak", "lat_skew", "lat_plane", "lat_elbpeak", "task_T2"]
    return names


def _extract(traj, ref_elv, latent, task, rise, terminated_frac):
    act = traj["act"]                       # (T, 63)
    elv = traj["elv"]; elb = traj["elb"]; plane = traj["plane"]
    rot = traj["rot"]; pros = traj["pros"]
    feats = [act.mean(0), act.max(0)]       # ACT 주채널 요약
    n = min(len(elv), len(ref_elv))
    kin = [elv.max(), elv[-1], float(np.sqrt(np.mean((elv[:n] - ref_elv[:n]) ** 2))),
           elb.max(), elb[-1], plane[-1], plane.max() - plane.min(),
           rot[-1], rot.max() - rot.min(), pros.max() - pros.min(), terminated_frac]
    cov = [latent["T"], latent["peak"], latent["skew"], latent["plane"],
           latent["elbow_peak"], 1.0 if task == "T2" else 0.0]
    return np.concatenate([feats[0], feats[1], kin, cov]).astype(np.float32)


def generate(model_path, vecnorm_path, task="mix", n=20000, seed=0,
             curriculum_k=(0, 1, 2, 3), curriculum_w=(0.15, 0.45, 0.25, 0.15)):
    """n 에피소드 롤아웃 → (X, y_sF, y_sL, latent). k=0 포함(건강 샘플)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # k=0 지원: curriculum에 0 포함 시 건강(섭동 없음) 에피소드
    ks = [k for k in curriculum_k if k >= 1] or [1]
    kw = [w for k, w in zip(curriculum_k, curriculum_w) if k >= 1]
    p_healthy = sum(w for k, w in zip(curriculum_k, curriculum_w) if k == 0)

    def mk():
        e = ArmPerturbEnv(task=task, latent_mode="sample", perturb=True, rsi=False,
                          seed=seed, curriculum_k=tuple(ks),
                          curriculum_k_weights=tuple(kw))
        return e
    venv = DummyVecEnv([mk])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False; venv.norm_reward = False
    raw = venv.venv.envs[0]
    model = PPO.load(model_path, device="cpu")
    rng = np.random.default_rng(seed + 777)

    X, ysF, ysL, lat_rec = [], [], [], []
    for i in range(n):
        # 건강 샘플 비율만큼 섭동 끄기
        if rng.random() < p_healthy:
            raw.set_perturbation(np.ones(MG.N_PERTURB_CH), np.ones(MG.N_PERTURB_CH))
        else:
            raw.set_perturbation(None)        # env 랜덤 섭동(k≥1)
        obs = venv.reset()
        sF = raw.s_F.copy(); sL = raw.s_L.copy(); latent = dict(raw.latent)
        task_i = raw.task
        ref_elv = np.degrees(raw.ref["shoulder_elv"]).copy()
        elv, elb, plane, rot, pros, act = [], [], [], [], [], []
        done = False; steps = 0
        while not done:
            elv.append(np.degrees(raw._q("shoulder_elv")))
            elb.append(np.degrees(raw._q("elbow_flexion")))
            plane.append(np.degrees(raw._q("elv_angle")))
            rot.append(np.degrees(raw.data.qpos[raw.rot_qadr]))
            pros.append(np.degrees(raw._q("pro_sup")))
            act.append(raw.data.act.copy())
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, _ = venv.step(a); done = d[0]; steps += 1
        traj = dict(act=np.array(act), elv=np.array(elv), elb=np.array(elb),
                    plane=np.array(plane), rot=np.array(rot), pros=np.array(pros))
        term_frac = 1.0 - steps / raw.n_steps      # 조기종료 비율(=KIN 발산 신호)
        X.append(_extract(traj, ref_elv, latent, task_i, raw.rise_steps, term_frac))
        ysF.append(sF); ysL.append(sL)
        lat_rec.append([latent["T"], latent["peak"], latent["skew"],
                        latent["plane"], latent["elbow_peak"]])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n}")
    raw.set_perturbation(None)
    return (np.array(X), np.array(ysF, np.float32), np.array(ysL, np.float32),
            np.array(lat_rec, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--task", default="mix")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sim/train.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    X, ysF, ysL, lat = generate(a.model, a.vecnorm, task=a.task, n=a.n, seed=a.seed)
    np.savez_compressed(a.out, X=X, y_sF=ysF, y_sL=ysL, latent=lat,
                        feature_names=np.array(_feature_names()),
                        channel_names=np.array([c for c, _ in MG.PERTURB_CHANNELS]))
    print(f"saved {a.out}: X{X.shape} y_sF{ysF.shape} y_sL{ysL.shape}")
    print(f"  약화샘플 비율: {(ysF.min(1) < 0.999).mean():.2f}")


if __name__ == "__main__":
    main()
