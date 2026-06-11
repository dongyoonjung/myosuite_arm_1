"""실현가능성 정밀 진단: 어떤 근육 조합이 팔을 95°로 드나 + reward 항 분해."""
import numpy as np
import sys
sys.path.insert(0, "/home/aaron/projects/myoarm_1")
from custom_envs.arm_perturb_v0 import ArmPerturbEnv
from custom_envs import muscle_groups as MG
import mujoco

env = ArmPerturbEnv(task="T1", latent_mode="fixed", perturb=False, seed=0)

def run_with(active, plane_deg=0.0, hold=True, steps=None):
    """active: {name: ctrl(0..1)}. eval 시작(rest), 지정 근육만 활성, 거상 추적."""
    env.set_eval(True)
    env.reset(seed=2)
    # 평면 강제(elv_angle)
    env.data.qpos[env.track_qadr["elv_angle"]] = np.radians(plane_deg)
    mujoco.mj_forward(env.model, env.data)
    a = -np.ones(env.n_act)
    for nm, c in active.items():
        a[MG.ACTION_MUSCLES.index(nm)] = 2 * c - 1
    elv = []
    S = steps or env.n_steps
    for _ in range(S):
        env.step(a)
        elv.append(np.degrees(env._q("shoulder_elv")))
    return np.array(elv)

print("=== 단일/조합 근육으로 거상 도달 (coronal plane=0) ===")
tests = {
    "DELT2 only": {"DELT2": 1.0},
    "DELT1+2+3": {"DELT1": 1.0, "DELT2": 1.0, "DELT3": 1.0},
    "DELT1+2+3+SUPSP": {"DELT1": 1.0, "DELT2": 1.0, "DELT3": 1.0, "SUPSP": 1.0},
    "DELT123+SUPSP+INFSP": {"DELT1": 1, "DELT2": 1, "DELT3": 1, "SUPSP": 1, "INFSP": 1},
    "ALL 26 full": {n: 1.0 for n in MG.ACTION_MUSCLES},
    "+PECM1(내전)": {"DELT1": 1, "DELT2": 1, "DELT3": 1, "SUPSP": 1, "PECM1": 1.0},
}
for name, act in tests.items():
    elv = run_with(act, plane_deg=0)
    print(f"  {name:24s}: 최대 {elv.max():5.1f}°  끝 {elv[-1]:5.1f}°")

print("\n=== 평면(elv_angle)별 DELT1+2+3+SUPSP 최대거상 ===")
for pl in (0, 15, 30, 45, 60):
    elv = run_with({"DELT1": 1, "DELT2": 1, "DELT3": 1, "SUPSP": 1}, plane_deg=pl)
    print(f"  plane={pl:2d}°: 최대 {elv.max():5.1f}°")

print("\n=== shoulder_rot 자유효과: 거상 중 외회전 허용 시 ===")
# DELT + 외회전 보조(INFSP/TMIN는 외회전근). 압핀 없으니 정책이 학습할 것 — 여기선 수동확인
elv = run_with({"DELT1": 1, "DELT2": 1, "DELT3": 1, "SUPSP": 1, "INFSP": 0.5, "TMIN": 0.5},
               plane_deg=30)
print(f"  DELT+SUPSP+외회전(INFSP/TMIN) @plane30: 최대 {elv.max():.1f}°")

# === reward 항 분해(완벽추적 시) ===
print("\n=== 완벽추적 reward 항 분해 ===")
env.set_eval(True); env.reset(seed=3)
comps = {"task": [], "quality": [], "effort": [], "reward": []}
for t in range(env.n_steps):
    i = min(env.t_idx, env.n_steps - 1)
    for nm in MG.TRACK_DOF:
        env.data.qpos[env.track_qadr[nm]] = env.ref[nm][i]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    r, info = env._reward(np.zeros(env.n_act))
    comps["task"].append(info["task"]); comps["quality"].append(info["quality"])
    comps["effort"].append(info["effort"]); comps["reward"].append(r)
    env.t_idx += 1
for k, v in comps.items():
    v = np.array(v)
    print(f"  {k:8s}: mean {v.mean():+.3f}  min {v.min():+.3f}  max {v.max():+.3f}")
# task 항이 낮으면 어느 DoF가 범인인지
env.set_eval(True); env.reset(seed=3)
i = env.n_steps // 2
for nm in MG.TRACK_DOF:
    env.data.qpos[env.track_qadr[nm]] = env.ref[nm][i]
env.data.qvel[:] = 0; mujoco.mj_forward(env.model, env.data)
env.t_idx = i
print("  중간프레임 각 DoF err(deg) & band_gauss:")
for nm in MG.TRACK_DOF:
    err = env.data.qpos[env.track_qadr[nm]] - env.ref[nm][i]
    bg = env._band_gauss(err, MG.TRACK_BAND[nm], MG.TRACK_KOUT[nm])
    print(f"     {nm:14s} err={np.degrees(err):+.2f}° band_gauss={bg:.3f} w={MG.TRACK_W[nm]}")
