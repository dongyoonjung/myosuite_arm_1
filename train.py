"""myoArm PPO 학습 (M1–M3 공용). 곱셈형 soft-tracking + anti-tremor 레시피.

예:
  # M1 건강 베이스라인(T1 고정 latent)
  OMP_NUM_THREADS=1 .venv/bin/python train.py --task T1 --timesteps 8000000 \
      --n-envs 20 --out ppo_arm_T1
  # 이어서 학습
  ... --resume models/ppo_arm_T1.zip --vecnorm models/ppo_arm_T1_vec.pkl

산출: models/<out>.zip + models/<out>_vec.pkl (+ checkpoints/, logs/)
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = os.path.join(os.path.dirname(__file__), "models")
LOGS = os.path.join(os.path.dirname(__file__), "logs")
CKPT = os.path.join(MODELS, "checkpoints")


def make_env_fn(task, latent_mode, perturb, rsi, seed, act_lowpass=0.0,
                effort_pow=2, reward_cfg=None, frame_skip=10,
                curriculum_k=(1, 2, 3), curriculum_w=(0.5, 0.3, 0.2), motor_noise=0.0,
                ref_gen="warp", wrist=False, horizon_s=3.5):
    def _f():
        from custom_envs.arm_perturb_v0 import ArmPerturbEnv
        e = ArmPerturbEnv(task=task, latent_mode=latent_mode, perturb=perturb,
                          rsi=rsi, seed=seed, act_lowpass=act_lowpass,
                          effort_pow=effort_pow, reward_cfg=reward_cfg,
                          frame_skip=frame_skip, curriculum_k=curriculum_k,
                          curriculum_k_weights=curriculum_w, motor_noise=motor_noise,
                          ref_gen=ref_gen, wrist=wrist, horizon_s=horizon_s)
        return e
    return _f


class GateCallback:
    """학습 중 주기적으로 M1 게이트 평가 → 콘솔/텐서보드 로깅."""
    def __init__(self, task, train_vecnorm, every=250_000, n_eps=3):
        self.task = task; self.vn = train_vecnorm
        self.every = every; self.n_eps = n_eps; self.last = 0

    def __call__(self, locals_, globals_):
        return True


def build_gate_callback(task, model_getter, train_vn, every=250_000, n_eps=3,
                        frame_skip=10, sample_latent=False, ref_gen="warp",
                        wrist=False, horizon_s=3.5):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from custom_envs.arm_perturb_v0 import ArmPerturbEnv
    from diagnostics import gate as G
    eval_task = "T1" if task == "mix" else task   # mix는 T1으로 모니터(분포 일부)

    class _GateCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last = 0

        def _make_eval(self):
            def mk():
                e = ArmPerturbEnv(task=eval_task, latent_mode="fixed", perturb=False,
                                  rsi=False, frame_skip=frame_skip, ref_gen=ref_gen,
                                  wrist=wrist, horizon_s=horizon_s)
                e.set_eval(True, sample_latent=sample_latent)
                return e
            venv = DummyVecEnv([mk])
            vn = VecNormalize(venv, training=False, norm_reward=False)
            # 학습 통계 공유
            vn.obs_rms = self.model.get_vec_normalize_env().obs_rms
            return vn, venv.envs[0]

        def _run(self):
            vn, raw = self._make_eval()
            eps = []
            for _ in range(self.n_eps_):
                obs = vn.reset(); done = False
                elv, ref_elv, t = [], [], []
                while not done:   # step 전 기록(post-reset 오염 방지, gate.py와 동일)
                    i = min(raw.t_idx, raw.n_steps - 1)
                    elv.append(np.degrees(raw._q("shoulder_elv")))
                    ref_elv.append(np.degrees(raw.ref["shoulder_elv"][i]))
                    t.append(raw.t_idx * raw.dt)
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, _, d, _ = vn.step(a); done = d[0]
                eps.append(dict(elv=np.array(elv), ref_elv=np.array(ref_elv),
                                elb=np.array([0]), plane=np.array([0]),
                                prosup=np.array([0]), rot=np.array([0]),
                                t=np.array(t), rise_steps=raw.rise_steps, dt=raw.dt))
            res = G.evaluate(eps, relative=sample_latent)
            g1 = res["G1_peak"]["value"]; rms = res["G5_rms"]["value"]
            trem = res["G4_tremor"]["value"]; rev = res["G2_monotonic"]["max"]
            self.logger.record("gate/g1", g1)
            self.logger.record("gate/rms_deg", rms)
            self.logger.record("gate/tremor", trem)
            self.logger.record("gate/reversals", rev)
            self.logger.record("gate/pass", int(res["PASS"]))
            g1lbl = "peakErr" if sample_latent else "peak"
            print(f"  [gate @ {self.num_timesteps}] {g1lbl}={g1:.1f}° rms={rms:.1f}° "
                  f"tremor={trem:.2f} rev={rev} speed={res['G3_speed']['value']:.2f}s "
                  f"PASS={res['PASS']}")

        n_eps_ = n_eps

        def _on_step(self):
            if self.num_timesteps - self.last >= every:
                self.last = self.num_timesteps
                try:
                    self._run()
                except Exception as e:
                    print("  [gate] 평가 실패:", e)
            return True

    return _GateCB()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1")
    ap.add_argument("--latent-mode", default="fixed", choices=["fixed", "sample"])
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--timesteps", type=int, default=8_000_000)
    ap.add_argument("--n-envs", type=int, default=20)
    ap.add_argument("--out", default="ppo_arm_T1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-every", type=int, default=300_000)
    # anti-tremor 스택(DESIGN escalation #2)
    ap.add_argument("--act-lowpass", type=float, default=0.0)   # 누설적분 α(0=off)
    ap.add_argument("--effort-pow", type=int, default=2)         # 3=cubic
    ap.add_argument("--w-vel", type=float, default=0.0)          # 속도추적(escalation #4)
    ap.add_argument("--c-dctrl", type=float, default=0.5)        # 행동평활
    ap.add_argument("--c-settle", type=float, default=1.0)       # hold 정지
    ap.add_argument("--w-effort", type=float, default=0.05)
    ap.add_argument("--caps", action="store_true")             # CAPS 공간평활 PPO
    ap.add_argument("--caps-lambda", type=float, default=0.5)
    ap.add_argument("--caps-sigma", type=float, default=0.05)
    ap.add_argument("--frame-skip", type=int, default=10)       # 10=50Hz, 20=25Hz
    ap.add_argument("--curriculum-k", default="1,2,3")          # M3 손상근 수 후보
    ap.add_argument("--curriculum-w", default="0.5,0.3,0.2")    # 가중
    ap.add_argument("--motor-noise", type=float, default=0.0)   # 신호의존 운동노이즈 σ_sd
    ap.add_argument("--ref-gen", default="warp")                # warp | vae | fpca
    ap.add_argument("--wrist", action="store_true")             # 손목 해제(T1 full-seq, 32근)
    ap.add_argument("--horizon", type=float, default=3.5)       # 에피소드 길이(full-seq는 4.0+)
    a = ap.parse_args()
    cur_k = tuple(int(x) for x in a.curriculum_k.split(","))
    cur_w = tuple(float(x) for x in a.curriculum_w.split(","))
    reward_cfg = dict(w_vel=a.w_vel, c_dctrl=a.c_dctrl, c_settle=a.c_settle,
                      w_effort=a.w_effort)

    os.makedirs(MODELS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    env_fns = [make_env_fn(a.task, a.latent_mode, a.perturb, True, a.seed + i,
                           act_lowpass=a.act_lowpass, effort_pow=a.effort_pow,
                           reward_cfg=reward_cfg, frame_skip=a.frame_skip,
                           curriculum_k=cur_k, curriculum_w=cur_w, motor_noise=a.motor_noise,
                           ref_gen=a.ref_gen, wrist=a.wrist, horizon_s=a.horizon)
               for i in range(a.n_envs)]
    venv = SubprocVecEnv(env_fns, start_method="spawn")
    if a.vecnorm and os.path.exists(a.vecnorm):
        venv = VecNormalize.load(a.vecnorm, venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    Algo = PPO
    extra = {}
    if a.caps:
        from caps_ppo import CAPSPPO
        Algo = CAPSPPO
        extra = dict(caps_lambda=a.caps_lambda, caps_sigma=a.caps_sigma)
        print(f"CAPS-PPO: lambda={a.caps_lambda} sigma={a.caps_sigma}")
    if a.resume and os.path.exists(a.resume):
        print("resume:", a.resume)
        model = Algo.load(a.resume, env=venv, device="cpu", **extra)
    else:
        model = Algo("MlpPolicy", venv, device="cpu", verbose=1,
                     learning_rate=a.lr, n_steps=a.n_steps, batch_size=a.batch_size,
                     n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                     ent_coef=a.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
                     policy_kwargs=policy_kwargs, tensorboard_log=LOGS, seed=a.seed,
                     **extra)

    ckpt_cb = CheckpointCallback(save_freq=max(1, 500_000 // a.n_envs),
                                 save_path=CKPT, name_prefix=a.out,
                                 save_vecnormalize=True)
    gate_cb = build_gate_callback(a.task, lambda: model, venv,
                                  every=a.gate_every, n_eps=4, frame_skip=a.frame_skip,
                                  sample_latent=(a.latent_mode == "sample"), ref_gen=a.ref_gen,
                                  wrist=a.wrist, horizon_s=a.horizon)

    model.learn(total_timesteps=a.timesteps, callback=[ckpt_cb, gate_cb],
                tb_log_name=a.out, progress_bar=False)

    mp = os.path.join(MODELS, f"{a.out}.zip")
    vp = os.path.join(MODELS, f"{a.out}_vec.pkl")
    model.save(mp); venv.save(vp)
    print("saved:", mp, vp)


if __name__ == "__main__":
    main()
