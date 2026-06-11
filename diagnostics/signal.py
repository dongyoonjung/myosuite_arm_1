"""M3 식별 신호 진단 — 근육 약화가 ACT(활성 보상) 또는 KIN(운동학 이탈)로 드러나나.

DESIGN 핵심: "추적 실패/이탈 = 식별 신호." 약화 채널마다 건강 대비:
  KIN 신호 = shoulder_elv 정점 미달(undershoot) + 궤적 이탈(deg)
  ACT 신호 = privileged 전체근육 활성의 변화(보상 패턴 norm)
성공 기준 = "두 채널 중 하나로 복원"(ACT 주채널, KIN 보너스).

건강 정책(M2)으로 돌리면 미보상 KIN 신호(약화→그냥 못 미침) 미리보기.
M3 학습 정책으로 돌리면 경증=ACT 보상·중증=KIN 이탈로 갈림.

사용:
  .venv/bin/python -m diagnostics.signal --model models/ppo_arm_M2b.zip \
      --vecnorm models/ppo_arm_M2b_vec.pkl --task T1 --sf 0.1
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_envs.arm_perturb_v0 import ArmPerturbEnv
from custom_envs import muscle_groups as MG


def _rollout_once(model, venv, raw, sF, sL):
    """고정 섭동 1회 롤아웃 → kin/act 시계열(step 전 기록=리셋오염 방지)."""
    raw.set_perturbation(sF, sL)
    obs = venv.reset()
    elv, elb, plane, rot, act = [], [], [], [], []
    done = False
    while not done:
        elv.append(np.degrees(raw._q("shoulder_elv")))
        elb.append(np.degrees(raw._q("elbow_flexion")))
        plane.append(np.degrees(raw._q("elv_angle")))
        rot.append(np.degrees(raw.data.qpos[raw.rot_qadr]))
        act.append(raw.data.act.copy())            # 전체 63근 활성(privileged)
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, _ = venv.step(a); done = d[0]
    return dict(elv=np.array(elv), elb=np.array(elb), plane=np.array(plane),
               rot=np.array(rot), act=np.array(act), rise=raw.rise_steps)


def scan(model_path, vecnorm_path, task="T1", s_F_weak=0.1, s_L_weak=1.0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def mk():
        e = ArmPerturbEnv(task=task, latent_mode="fixed", perturb=False, rsi=False)
        e.set_eval(True)        # 고정 중앙 latent; 섭동은 set_perturbation으로 강제
        return e
    venv = DummyVecEnv([mk])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False; venv.norm_reward = False
    raw = venv.venv.envs[0]
    model = PPO.load(model_path, device="cpu")

    def hold_mean(r):
        """hold 평균 활성; 조기종료(hold 없음)면 후반부 사용."""
        a = r["act"]
        seg = a[r["rise"]:] if len(a) > r["rise"] + 1 else a[max(0, len(a) // 2):]
        return seg.mean(0) if len(seg) else a.mean(0)

    ones = np.ones(MG.N_PERTURB_CH)
    base = _rollout_once(model, venv, raw, ones, ones)       # 건강
    base_act = hold_mean(base)
    base_peak = base["elv"].max()

    print(f"건강: 정점 {base_peak:.1f}°  elbow {base['elb'].max():.0f}°  "
          f"rot {base['rot'][-1]:.0f}°")
    print(f"\n{'채널':<12}{'KIN(정점Δ)':>11}{'KIN(궤적RMS)':>13}{'rotΔ':>8}{'ACT변화':>9}{'  신호'}")
    rows = []
    for ch, (name, muses) in enumerate(MG.PERTURB_CHANNELS):
        sF = ones.copy(); sF[ch] = s_F_weak
        sL = ones.copy(); sL[ch] = s_L_weak
        r = _rollout_once(model, venv, raw, sF, sL)
        # KIN 신호
        n = min(len(r["elv"]), len(base["elv"]))
        kin_peak = base_peak - r["elv"].max()                # undershoot(+면 못미침)
        kin_rms = float(np.sqrt(np.mean((r["elv"][:n] - base["elv"][:n]) ** 2)))
        rot_dev = abs(r["rot"][-1] - base["rot"][-1])
        # ACT 신호(privileged 전체근육 hold 평균 변화)
        wact = hold_mean(r)
        diverged = len(r["elv"]) < base["rise"]           # 조기종료=발산
        act_change = float(np.abs(wact - base_act).sum())
        act_max_ch = int(np.argmax(np.abs(wact - base_act)))
        act_max_nm = (__import__("mujoco").mj_id2name(
            raw.model, __import__("mujoco").mjtObj.mjOBJ_ACTUATOR, act_max_ch))
        sig = []
        if abs(kin_peak) > 8 or kin_rms > 8 or rot_dev > 10 or diverged:
            sig.append("KIN")
        if act_change > 0.3:
            sig.append("ACT")
        tag = ("+".join(sig) or "—") + ("!" if diverged else "")
        rows.append((name, kin_peak, kin_rms, rot_dev, act_change, tag, act_max_nm))
        print(f"{name:<12}{kin_peak:>10.1f}°{kin_rms:>12.1f}°{rot_dev:>7.0f}°"
              f"{act_change:>9.2f}   {tag:<8}(↑{act_max_nm})")
    raw.set_perturbation(None)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--task", default="T1")
    ap.add_argument("--sf", type=float, default=0.1)     # 약화 정도(s_F)
    ap.add_argument("--sl", type=float, default=1.0)
    a = ap.parse_args()
    print(f"=== M3 식별 신호 스캔 (s_F={a.sf}) — {os.path.basename(a.model)} ===")
    scan(a.model, a.vecnorm, task=a.task, s_F_weak=a.sf, s_L_weak=a.sl)


if __name__ == "__main__":
    main()
