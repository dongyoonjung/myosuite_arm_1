"""M1 환경 스모크 테스트 + 물리적 실현가능성(팔을 95°로 들 수 있나)."""
import numpy as np
import sys, os
sys.path.insert(0, "/home/aaron/projects/myoarm_1")
from custom_envs.arm_perturb_v0 import ArmPerturbEnv
import mujoco

env = ArmPerturbEnv(task="T1", latent_mode="fixed", perturb=False, seed=0)
print(f"obs_dim={env.observation_space.shape} act_dim={env.action_space.shape}")
print(f"dt={env.dt} n_steps={env.n_steps} rise_steps={env.rise_steps}")
print(f"model neq={env.model.neq} nu={env.model.nu} (action {env.n_act})")

# 1) 랜덤 정책 한 에피소드
env.set_eval(False)
obs, _ = env.reset(seed=1)
tot = 0.0; n = 0
for _ in range(env.n_steps):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    tot += r; n += 1
    if term or trunc:
        break
print(f"\n[랜덤] steps={n} return={tot:.1f} obs_finite={np.all(np.isfinite(obs))}")

# 2) 실현가능성: 거상근 강하게 켜고 팔이 올라가나 (eval, rest 시작)
env.set_eval(True)
obs, _ = env.reset(seed=2)
# DELT1,2,3 + SUPSP 강하게, 나머지 중립
names = env.action_names if hasattr(env, "action_names") else None
from custom_envs import muscle_groups as MG
a = -np.ones(env.n_act)  # 전부 0 활성
lift = ["DELT1", "DELT2", "DELT3", "SUPSP", "PECM1"]
for nm in lift:
    a[MG.ACTION_MUSCLES.index(nm)] = 1.0   # 최대
# 팔꿈치 약간(BIClong)으로 elbow 굽힘
a[MG.ACTION_MUSCLES.index("BIClong")] = 0.2
elvs = []
for t in range(env.n_steps):
    obs, r, term, trunc, info = env.step(a)
    elvs.append(np.degrees(env._q("shoulder_elv")))
    if term or trunc:
        break
elvs = np.array(elvs)
print(f"\n[거상근 최대] shoulder_elv: 시작 {elvs[0]:.1f}° → 최대 {elvs.max():.1f}° "
      f"끝 {elvs[-1]:.1f}° (목표 ~95°)")
print(f"   도달 가능?(>85°) {elvs.max() > 85}")

# 3) 참조 자체 자세를 그대로 주입했을 때 reward(추적완벽 상한)
env.set_eval(True)
obs, _ = env.reset(seed=3)
perfect_r = []
for t in range(env.n_steps):
    # 참조 자세로 강제 세팅 후 reward만 평가(정책 상한 감 잡기)
    for nm in MG.TRACK_DOF:
        env.data.qpos[env.track_qadr[nm]] = env.ref[nm][min(env.t_idx, env.n_steps-1)]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    a0 = np.zeros(env.n_act)
    r, info = env._reward(a0)
    perfect_r.append(r)
    env.t_idx += 1
print(f"\n[완벽추적 reward] step당 평균 {np.mean(perfect_r):.3f} "
      f"(task항 상한≈0.75; quality·effort로 약간 낮음)")

# 4) 섭동(M3) 동작 확인
env2 = ArmPerturbEnv(task="T1", perturb=True, seed=7)
env2.set_eval(False)
env2.reset(seed=7)
print(f"\n[섭동] s_F={np.round(env2.s_F,2)}")
print(f"       s_L={np.round(env2.s_L,2)}")
n_weak = int((env2.s_F < 0.999).sum())
print(f"       약화 채널 수 k={n_weak} (커리큘럼 {env2.curriculum_k})")

print("\nSMOKE OK")
