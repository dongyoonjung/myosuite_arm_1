"""학습 정책 1 에피소드를 OSMesa로 렌더 → mp4. 시각 확인용.

사용(헤드리스):
  MUJOCO_GL=osmesa .venv/bin/python tools/render_episode.py \
      --model models/ppo_arm_T1.zip --vecnorm models/ppo_arm_T1_vec.pkl \
      --task T1 --out results/report/M1_T1.mp4
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_envs.arm_perturb_v0 import ArmPerturbEnv


def render_episode(model_path, vec_path, task="T1", out="episode.mp4",
                   W=640, H=480, fps=50, az=90.0, el=-12.0, dist=1.5,
                   lookat=(-0.25, 0.10, 1.20), perturb_sF=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def mk():
        e = ArmPerturbEnv(task=task, latent_mode="fixed", perturb=False, rsi=False)
        e.set_eval(True)
        return e
    venv = DummyVecEnv([mk])
    venv = VecNormalize.load(vec_path, venv); venv.training = False; venv.norm_reward = False
    raw = venv.venv.envs[0]
    if perturb_sF is not None:
        raw.set_perturbation(perturb_sF, np.ones_like(perturb_sF))
    model = PPO.load(model_path, device="cpu")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    renderer = mujoco.Renderer(raw.model, H, W)

    obs = venv.reset()
    frames = []
    peak = 0.0; last_elv = 0.0
    done = False
    while not done:
        # step 전(자동리셋 오염 방지) 렌더·각도 기록
        renderer.update_scene(raw.data, camera=cam)
        frames.append(renderer.render().copy())
        last_elv = np.degrees(raw._q("shoulder_elv")); peak = max(peak, last_elv)
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, _ = venv.step(a); done = d[0]
    # 마지막 정상 프레임(정점유지)을 0.6초 반복 — 리셋 자세 렌더 금지
    frames += [frames[-1]] * int(fps * 0.6)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, quality=8)
    print(f"saved {out}: {len(frames)} frames @ {fps}fps  정점 {peak:.0f}° 최종 {last_elv:.0f}°  "
          f"mean px {np.mean(frames[len(frames)//2]):.0f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--task", default="T1")
    ap.add_argument("--out", default="results/report/episode.mp4")
    ap.add_argument("--az", type=float, default=90.0)
    ap.add_argument("--el", type=float, default=-12.0)
    ap.add_argument("--dist", type=float, default=1.5)
    a = ap.parse_args()
    render_episode(a.model, a.vecnorm, task=a.task, out=a.out,
                   az=a.az, el=a.el, dist=a.dist)
