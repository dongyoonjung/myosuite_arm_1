"""M1 건강 베이스라인 검증 게이트 (★임계경로).

명목파라미터(s_F=s_L=1) 정책이 인간 템플릿을 재현하는지 5기준:
  G1 정점 인간 91±11°(=[80,102]) 내
  G2 단조 상승(반전 0)
  G3 인간대역 속도(상승시간 T∈[1.0,4.0]s)
  G4 4–12Hz 떨림 없음(hold 속도 스펙트럼 대역파워 비율 < 0.15)
  G5 추종 RMS 허용내(shoulder_elv RMS < 12°)

통과 못 하면 "이탈=신호" 전제 무효 → 다음 단계 금지(DESIGN.md).

사용:
  .venv/bin/python -m diagnostics.gate --model models/ppo_arm_T1.zip \
      --vecnorm models/vec_T1.pkl [--episodes 10] [--plot]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_envs.arm_perturb_v0 import ArmPerturbEnv

# 인간 기준(latent_T1.json 실측)
PEAK_LO, PEAK_HI = 80.0, 102.0          # 91±11
T_RISE_LO, T_RISE_HI = 1.0, 4.0         # 인간 T p5~p95 여유
TREMOR_BAND = (4.0, 12.0)
TREMOR_MAX_RATIO = 0.15
TREMOR_MAX_AMP_DEG = 0.5     # 4–12Hz 각도성분 RMS(deg). 생리적 떨림 < ~0.5°
RMS_MAX_DEG = 12.0


def _reversals(x, deadband=1.5):
    """상승 구간의 *의미 있는* 후퇴 횟수(running-max 대비 deadband° 이상 하강).

    제어 미세 jitter(<0.5°)는 무시 — "단조 상승"은 거시적 후퇴 부재를 뜻함.
    인간 템플릿은 구성상 단조라 0; 정책의 0.1° 떨림이 반전으로 오집계되던 것 교정.
    """
    if len(x) < 3:
        return 0
    revs = 0
    running_max = x[0]
    dropping = False
    for v in x[1:]:
        if v > running_max:
            running_max = v
            dropping = False
        elif v < running_max - deadband and not dropping:
            revs += 1
            dropping = True
        elif dropping and v > running_max - 0.5 * deadband:
            dropping = False
    return revs


def rollout(model_path, vecnorm_path, task="T1", n_eps=10, deterministic=True,
            sample_latent=False, frame_skip=10):
    """학습 정책을 평가모드(rest 시작·명목 param)로 롤아웃.

    sample_latent=False(M1): 고정 중앙값 latent. True(M2): latent 분포 전반.
    frame_skip: 학습과 동일해야 함(25Hz=20).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make():
        e = ArmPerturbEnv(task=task, latent_mode="fixed", perturb=False, rsi=False,
                          frame_skip=frame_skip)
        e.set_eval(True, sample_latent=sample_latent)
        return e
    venv = DummyVecEnv([make])
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    raw = venv.venv.envs[0] if hasattr(venv, "venv") else venv.envs[0]
    dt = raw.dt
    eps = []
    for ep in range(n_eps):
        obs = venv.reset()
        elv, ref_elv, elb, plane, prosup, rot, t = [], [], [], [], [], [], []
        done = False
        # ★ step 전(action 결정 시점)의 상태를 기록 — DummyVecEnv가 done 시 자동
        #   리셋하므로 step 후 기록하면 마지막에 rest(≈0) 샘플이 섞여 떨림이 오염됨.
        while not done:
            i = min(raw.t_idx, raw.n_steps - 1)
            elv.append(np.degrees(raw._q("shoulder_elv")))
            ref_elv.append(np.degrees(raw.ref["shoulder_elv"][i]))
            elb.append(np.degrees(raw._q("elbow_flexion")))
            plane.append(np.degrees(raw._q("elv_angle")))
            prosup.append(np.degrees(raw._q("pro_sup")))
            rot.append(np.degrees(raw.data.qpos[raw.rot_qadr]))
            t.append(raw.t_idx * dt)
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, _ = venv.step(action)
            done = dones[0]
        eps.append(dict(elv=np.array(elv), ref_elv=np.array(ref_elv),
                        elb=np.array(elb), plane=np.array(plane),
                        prosup=np.array(prosup), rot=np.array(rot),
                        t=np.array(t), rise_steps=raw.rise_steps, dt=dt))
    return eps


PEAK_MATCH_TOL = 12.0    # M2 상대모드: 달성정점이 참조정점에서 이내(deg)


def evaluate(eps, relative=False):
    """에피소드들 → 게이트 5기준 판정(보수적: worst-case).

    relative=False(M1 고정 latent): G1=절대 인간밴드[80,102].
    relative=True(M2 latent 분포): G1=참조정점 매칭(|달성−참조|<12°) — 참조 정점이
      77~107° 변동하므로 절대밴드 부적합. 정책이 *샘플된 참조*를 추종하는지 본다.
    """
    res = {}
    peaks, revs, t_rises, tremor_ratios, tremor_amps, rms_list = [], [], [], [], [], []
    peak_errs = []
    for e in eps:
        elv = e["elv"]; dt = e["dt"]; rise = e["rise_steps"]
        peak = float(np.max(elv))
        peaks.append(peak)
        peak_errs.append(abs(peak - float(np.max(e["ref_elv"]))))
        # 상승 구간(시작~정점 도달 인덱스)
        ip = int(np.argmax(elv))
        rise_seg = elv[:max(ip + 1, 2)]
        revs.append(_reversals(rise_seg))
        # 상승시간: 정점의 10%→90% 통과 시간
        lo_th, hi_th = 0.1 * peak, 0.9 * peak
        above_lo = np.where(elv >= lo_th)[0]
        above_hi = np.where(elv >= hi_th)[0]
        if len(above_lo) and len(above_hi):
            t_rises.append((above_hi[0] - above_lo[0]) * dt)
        else:
            t_rises.append(np.nan)
        # 떨림: hold 구간 각도 스펙트럼 — 대역파워 비율 + 절대진폭(deg)
        if len(elv) > rise + 8:
            hold = elv[rise:]
            ac = hold - hold.mean()
            if len(ac) >= 8:
                spec = np.abs(np.fft.rfft(ac)) ** 2
                freqs = np.fft.rfftfreq(len(ac), dt)
                band = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
                tremor_ratios.append(float(spec[band].sum() / (spec.sum() + 1e-9)))
                # Parseval: 대역 RMS(deg) = sqrt(2·Σ|X|²_band)/N
                amp = np.sqrt(2.0 * spec[band].sum()) / len(ac)
                tremor_amps.append(float(amp))
        # 추종 RMS
        rms_list.append(float(np.sqrt(np.mean((e["elv"] - e["ref_elv"]) ** 2))))

    if relative:
        pe = float(np.median(peak_errs))
        res["G1_peak"] = dict(value=pe, hi=float(np.max(peak_errs)),
                              ok=pe < PEAK_MATCH_TOL and np.percentile(peak_errs, 80) < 1.5 * PEAK_MATCH_TOL)
    else:
        peak_med = float(np.median(peaks))
        res["G1_peak"] = dict(value=peak_med, lo=min(peaks), hi=max(peaks),
                              ok=PEAK_LO <= np.median(peaks) <= PEAK_HI
                              and np.mean([(PEAK_LO <= p <= PEAK_HI) for p in peaks]) >= 0.6)
    res["G2_monotonic"] = dict(value=float(np.mean(revs)), max=int(np.max(revs)),
                               ok=np.max(revs) <= 1)   # 평활 후 ≤1 허용
    tr = np.nanmedian(t_rises)
    # 상대모드(분포)는 속도 밴드를 넓힘(T가 1~5.5s 변동)
    lo_s, hi_s = (0.5, 5.0) if relative else (T_RISE_LO, T_RISE_HI)
    res["G3_speed"] = dict(value=float(tr), ok=bool(lo_s <= tr <= hi_s))
    tremor = float(np.median(tremor_ratios)) if tremor_ratios else 0.0
    amp = float(np.median(tremor_amps)) if tremor_amps else 0.0
    # 통과: 대역비율 낮거나 OR 절대진폭이 생리적 수준(<0.5°)
    res["G4_tremor"] = dict(value=tremor, amp_deg=amp,
                            ok=(tremor < TREMOR_MAX_RATIO) or (amp < TREMOR_MAX_AMP_DEG))
    rms = float(np.median(rms_list))
    res["G5_rms"] = dict(value=rms, hi=float(np.max(rms_list)),
                         ok=rms < RMS_MAX_DEG)
    res["PASS"] = all(v["ok"] for k, v in res.items() if k.startswith("G"))
    return res


def print_report(res, relative=False):
    print("\n=== 검증 게이트", "(M2 분포·상대)" if relative else "(M1 고정)", "===")
    labels = {"G1_peak": (f"정점 매칭 |달성−참조|<{PEAK_MATCH_TOL:.0f}°" if relative
                          else f"정점 인간 [{PEAK_LO:.0f},{PEAK_HI:.0f}]°내"),
              "G2_monotonic": "단조 상승(반전≤1)",
              "G3_speed": "상승시간 인간대역",
              "G4_tremor": f"4–12Hz 떨림 <{TREMOR_MAX_RATIO}",
              "G5_rms": f"추종 RMS <{RMS_MAX_DEG}°"}
    for k in ["G1_peak", "G2_monotonic", "G3_speed", "G4_tremor", "G5_rms"]:
        v = res[k]
        mark = "✓" if v["ok"] else "✗"
        extra = " ".join(f"{kk}={vv:.2f}" if isinstance(vv, float) else f"{kk}={vv}"
                         for kk, vv in v.items() if kk != "ok")
        print(f"  [{mark}] {labels[k]:28s} {extra}")
    print(f"  >>> {'PASS ✅' if res['PASS'] else 'FAIL ❌'}")
    return res["PASS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--task", default="T1")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--sample-latent", action="store_true")   # M2: 분포 전반 평가
    ap.add_argument("--frame-skip", type=int, default=10)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    eps = rollout(a.model, a.vecnorm, task=a.task, n_eps=a.episodes,
                  sample_latent=a.sample_latent, frame_skip=a.frame_skip)
    res = evaluate(eps, relative=a.sample_latent)
    ok = print_report(res, relative=a.sample_latent)
    if a.plot:
        _plot(eps, a.model)
    sys.exit(0 if ok else 1)


def _plot(eps, model_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for e in eps:
        axes[0, 0].plot(e["t"], e["elv"], "b", alpha=0.4)
        axes[0, 0].plot(e["t"], e["ref_elv"], "k--", alpha=0.3)
        axes[0, 1].plot(e["t"], e["elb"], alpha=0.4)
        axes[1, 0].plot(e["t"], e["plane"], alpha=0.4)
        axes[1, 1].plot(e["t"], e["rot"], alpha=0.4)
    axes[0, 0].set_title("shoulder_elv (파랑) vs ref (검정점선)")
    axes[0, 1].set_title("elbow_flexion")
    axes[1, 0].set_title("elv_angle(plane)")
    axes[1, 1].set_title("shoulder_rot (자유)")
    for ax in axes.flat:
        ax.set_xlabel("t(s)"); ax.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(model_path),
                       os.path.basename(model_path).replace(".zip", "_gate.png"))
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("plot:", out)


if __name__ == "__main__":
    main()
