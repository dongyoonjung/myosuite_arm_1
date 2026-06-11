"""보고서용 궤적 그림 생성 — 학습이 제대로 일어났음을 시각 입증.

Fig1 M2b 추적 갤러리(T1/T2, 여러 latent) : 정책이 다양한 참조를 따라감
Fig2 latent 활용 산점도(참조정점 vs 달성)  : 정책이 latent을 실제로 씀(상관)
Fig3 M3 섭동 신호(건강/경증/중증 궤적+활성) : 약화→KIN이탈/ACT보상
Fig4 학습곡선(로그 파싱)                    : 게이트 지표 수렴
산출: results/report/*.png
"""
import os
import re
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_envs.arm_perturb_v0 import ArmPerturbEnv
from custom_envs import muscle_groups as MG

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "report")
os.makedirs(OUT, exist_ok=True)


def load_policy(model, vec, task="T1", perturb=False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def mk():
        e = ArmPerturbEnv(task=task, latent_mode="fixed", perturb=perturb, rsi=False)
        e.set_eval(True, sample_latent=True)
        return e
    venv = DummyVecEnv([mk])
    venv = VecNormalize.load(vec, venv); venv.training = False; venv.norm_reward = False
    return PPO.load(model, device="cpu"), venv, venv.venv.envs[0]


def rollout(model, venv, raw, sample_latent=True, sF=None, sL=None):
    raw.set_eval(True, sample_latent=sample_latent)
    if sF is not None:
        raw.set_perturbation(sF, sL)
    else:
        raw.set_perturbation(None)
    obs = venv.reset()
    rec = {k: [] for k in ("elv", "refelv", "elb", "refelb", "plane", "refplane", "rot", "t")}
    acts = []
    done = False
    while not done:
        i = min(raw.t_idx, raw.n_steps - 1)
        rec["elv"].append(np.degrees(raw._q("shoulder_elv")))
        rec["refelv"].append(np.degrees(raw.ref["shoulder_elv"][i]))
        rec["elb"].append(np.degrees(raw._q("elbow_flexion")))
        rec["refelb"].append(np.degrees(raw.ref["elbow_flexion"][i]))
        rec["plane"].append(np.degrees(raw._q("elv_angle")))
        rec["refplane"].append(np.degrees(raw.ref["elv_angle"][i]))
        rec["rot"].append(np.degrees(raw.data.qpos[raw.rot_qadr]))
        rec["t"].append(raw.t_idx * raw.dt)
        acts.append(raw.data.act[raw.action_idx].copy())
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, _ = venv.step(a); done = d[0]
    rec = {k: np.array(v) for k, v in rec.items()}
    rec["act"] = np.array(acts)
    rec["latent"] = dict(raw.latent)
    return rec


def fig1_gallery(M2b):
    model, venv, raw = M2b
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, task in enumerate(["T1", "T2"]):
        raw.task = task
        eps = [rollout(model, venv, raw, sample_latent=True) for _ in range(7)]
        ax = axes[col, 0]
        for e in eps:
            ax.plot(e["t"], e["elv"], lw=1.6, alpha=0.85)
            ax.plot(e["t"], e["refelv"], "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"{task}: shoulder_elv  policy(color) vs ref(dashed)")
        ax.set_ylabel("elevation (deg)"); ax.grid(alpha=0.3)
        ax = axes[col, 1]
        for e in eps:
            ax.plot(e["t"], e["elb"], lw=1.4, alpha=0.8)
            ax.plot(e["t"], e["refelb"], "k--", lw=0.7, alpha=0.4)
        ax.set_title(f"{task}: elbow_flexion vs ref"); ax.grid(alpha=0.3)
        ax = axes[col, 2]
        for e in eps:
            ax.plot(e["t"], e["plane"], lw=1.4, alpha=0.8)
            ax.plot(e["t"], e["refplane"], "k--", lw=0.7, alpha=0.4)
        ax.set_title(f"{task}: elv_angle/plane vs ref" +
                     ("  (T2 forward sweep 0->+45)" if task == "T2" else "  (T1 coronal ~0)"))
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("M2b single policy tracks varied latents & both tasks (T1 coronal / T2 forward)",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_tracking_gallery.png"); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig2_latent_scatter(M2b):
    model, venv, raw = M2b
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["T1", "T2"]):
        raw.task = task
        rp, ap, rt, at = [], [], [], []
        for _ in range(40):
            e = rollout(model, venv, raw, sample_latent=True)
            rp.append(e["refelv"].max()); ap.append(e["elv"].max())
            rt.append(e["latent"]["T"]); at.append(e["t"][-1])
        rp, ap = np.array(rp), np.array(ap)
        r = np.corrcoef(rp, ap)[0, 1]
        ax.scatter(rp, ap, s=22, alpha=0.7)
        lim = [min(rp.min(), ap.min()) - 3, max(rp.max(), ap.max()) + 3]
        ax.plot(lim, lim, "k--", alpha=0.5, label="ideal y=x")
        ax.set_xlabel("reference peak (deg)"); ax.set_ylabel("achieved peak (deg)")
        ax.set_title(f"{task}: peak tracking across latents  (corr={r:.2f})")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Policy conditions on latent: achieved peak follows sampled reference peak", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_latent_usage.png"); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig3_signal(M3, task="T1"):
    model, venv, raw = M3
    raw.task = task
    # 대표 채널: DELT2(주외전), SUPSP(극상), TRIlong(이관절), A_lowcuff(회전근개)
    chans = [("DELT2", 1), ("SUPSP", 3), ("TRIlong", 7), ("A_lowcuff", 8)]
    ones = np.ones(MG.N_PERTURB_CH)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (cname, ci) in enumerate(chans):
        # 같은 latent 비교 위해 sample_latent False(고정 중앙값)
        base = rollout(model, venv, raw, sample_latent=False, sF=ones, sL=ones)
        sevF = ones.copy(); sevF[ci] = 0.1
        mildF = ones.copy(); mildF[ci] = 0.5
        sev = rollout(model, venv, raw, sample_latent=False, sF=sevF, sL=ones)
        mild = rollout(model, venv, raw, sample_latent=False, sF=mildF, sL=ones)
        ax = axes[0, col]
        ax.plot(base["t"], base["refelv"], "k--", lw=1, label="reference")
        ax.plot(base["t"], base["elv"], "g", lw=2, label="healthy s_F=1")
        ax.plot(mild["t"], mild["elv"], "orange", lw=2, label="mild s_F=0.5")
        ax.plot(sev["t"], sev["elv"], "r", lw=2, label="severe s_F=0.1")
        ax.set_title(f"weaken {cname}: elevation (KIN)")
        ax.set_ylabel("elevation (deg)"); ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
        # 활성: 약화근(첫 근육) + 최대 보상근
        ax = axes[1, col]
        m0 = MG.PERTURB_CHANNELS[ci][1][0]
        mi = MG.ACTION_MUSCLES.index(m0)
        ax.plot(base["t"], base["act"][:, mi], "g", lw=1.5, label=f"{m0} healthy")
        ax.plot(sev["t"], sev["act"][:, mi], "r", lw=1.5, label=f"{m0} severe")
        # 최대 보상근(severe에서 base 대비 가장 증가)
        n = min(len(sev["act"]), len(base["act"]))
        dlt = (sev["act"][:n].mean(0) - base["act"][:n].mean(0))
        comp = int(np.argmax(dlt))
        cname2 = MG.ACTION_MUSCLES[comp]
        ax.plot(base["t"], base["act"][:, comp], "b--", lw=1.2, alpha=0.6, label=f"{cname2} healthy")
        ax.plot(sev["t"], sev["act"][:, comp], "b", lw=1.5, label=f"{cname2} severe(↑comp)")
        ax.set_title(f"{cname}: activations (ACT signal)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("activation"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("M3 identification signal: mild weakness -> ACT compensation (elevation recovers); "
                 "severe / key muscle -> KIN deviation (undershoot)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_perturbation_signal.png"); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig4_curves():
    logs = {"M1": "train_T1_final.log", "M2a": "train_M2a.log",
            "M2b": "train_M2b.log", "M3a": "train_M3a.log", "M3b": "train_M3b.log"}
    base = os.path.join(os.path.dirname(__file__), "..", "logs")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    # peakErr(상대) 또는 peak(절대)를 구분 — 절대는 |peak-96|(중앙참조 대비 오차)로 환산해 통일
    pat = re.compile(r"gate @ (\d+)\] (peakErr|peak)=([\d.]+).* rms=([\d.]+).* tremor=([\d.]+)")
    for name, fn in logs.items():
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            continue
        steps, g1, rms, trem = [], [], [], []
        for line in open(p):
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                val = float(m.group(3))
                g1.append(val if m.group(2) == "peakErr" else abs(val - 96.0))
                rms.append(float(m.group(4))); trem.append(float(m.group(5)))
        if not steps:
            continue
        steps = np.array(steps) / 1e6
        axes[0].plot(steps, g1, "o-", label=name, alpha=0.8)
        axes[1].plot(steps, rms, "o-", label=name, alpha=0.8)
        axes[2].plot(steps, trem, "o-", label=name, alpha=0.8)
    axes[0].set_title("peak error vs reference (deg)"); axes[0].axhline(12, color="r", ls=":", alpha=0.5)
    axes[1].set_title("tracking RMS (deg)"); axes[1].axhline(12, color="r", ls=":", alpha=0.5)
    axes[2].set_title("tremor band-ratio"); axes[2].axhline(0.15, color="r", ls=":", alpha=0.5)
    for ax in axes:
        ax.set_xlabel("timesteps (M)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Learning curves (in-training gate). Red dotted = pass threshold. "
                 "Note: gates evaluate healthy tracking; M3 perturbation handled separately.", fontsize=11)
    fig.tight_layout()
    pth = os.path.join(OUT, "fig4_learning_curves.png"); fig.savefig(pth, dpi=110); plt.close(fig)
    return pth


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2b", default="models/ppo_arm_M2b.zip")
    ap.add_argument("--m2b-vec", default="models/ppo_arm_M2b_vec.pkl")
    ap.add_argument("--m3", default="models/ppo_arm_M3a.zip")
    ap.add_argument("--m3-vec", default="models/ppo_arm_M3a_vec.pkl")
    a = ap.parse_args()
    print("fig4:", fig4_curves())
    M2b = load_policy(a.m2b, a.m2b_vec, task="T1")
    print("fig1:", fig1_gallery(M2b))
    print("fig2:", fig2_latent_scatter(M2b))
    M3 = load_policy(a.m3, a.m3_vec, task="T1", perturb=False)
    print("fig3:", fig3_signal(M3))
    print("done")
