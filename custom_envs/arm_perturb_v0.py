"""myoArm 어깨거상 soft-tracking 환경 (M1–M4 공용).

설계(DESIGN.md):
  reward = TASK_track · QUALITY + w_e·EFFORT + penalties
    TASK_track = Σ wᵢ · band_gauss(qᵢ−q_refᵢ)   (양면 허용폭: 밴드 안 평평 / 밖 가파름)
    QUALITY    = exp(−(jerk + dctrl + settle))
    EFFORT     = −mean(act²)
  행동 26근(어깨15+팔꿈치9+PT·PQ), 원위 22관절 잠금(equality), 견갑 rhythm 자동.
  참조 = references.warp(라텐트 워핑, raise-and-hold). shoulder_rot=자유(참조 없음).
  섭동 C(M3) = gainprm/biasprm[:,2]*=s_F, [:,0:2]/=s_L (런타임 편집).

마일스톤 토글:
  M1: task='T1', latent_mode='fixed'(중앙값), perturb=False.
  M2: latent_mode='sample', task∈{T1,T2}.
  M3: perturb=True, curriculum k.
"""
import os
import sys

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from references import warp
from references import fpca_ref
from custom_envs import muscle_groups as MG

XML_DEFAULT = os.path.join(
    os.path.dirname(mujoco.__file__), "..", "myosuite", "simhive",
    "myo_sim", "arm", "myoarm.xml")


def _find_xml():
    import myosuite
    base = os.path.dirname(myosuite.__file__)
    p = os.path.join(base, "simhive", "myo_sim", "arm", "myoarm.xml")
    if os.path.exists(p):
        return p
    import glob
    hits = glob.glob(os.path.join(base, "**", "myoarm.xml"), recursive=True)
    if not hits:
        raise FileNotFoundError("myoarm.xml 없음")
    return hits[0]


_MODEL_CACHE = {}


def build_locked_model(xml=None, cache=True, lock_joints=None):
    """지정 관절을 equality 상수잠금한 myoArm MjModel.

    lock_joints=None이면 기본(원위 22: 손목+손가락). 손목해제 모드는
    손가락 20만 잠그도록 MG.LOCK_JOINTS_WRIST 전달 → 손목은 자유.
    cache=True면 (xml,lock) 조합당 컴파일 1회 캐시.
    """
    xml = xml or _find_xml()
    lock_joints = MG.LOCK_JOINTS if lock_joints is None else lock_joints
    key = (xml, tuple(lock_joints))
    if cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    spec = mujoco.MjSpec.from_file(xml)
    for jn in lock_joints:
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = jn
        eq.name2 = ""
        eq.data[:5] = [0, 0, 0, 0, 0]   # 상수 0 잠금
        eq.active = True
    model = spec.compile()
    if cache:
        _MODEL_CACHE[key] = model
    return model


class ArmPerturbEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task="T1", latent_mode="fixed", perturb=False,
                 horizon_s=3.5, frame_skip=10, rsi=True, seed=None,
                 curriculum_k=(1, 2, 3), curriculum_k_weights=(0.5, 0.3, 0.2),
                 reward_cfg=None, xml=None, act_lowpass=0.0, effort_pow=2,
                 joint_damping_mult=1.0, armature_mult=1.0,
                 motor_noise=0.0, motor_noise_floor=0.01, ref_gen="warp", wrist=False):
        super().__init__()
        # 신호의존 운동노이즈(Harris&Wolpert 1998): SD ∝ 명령크기 → 현실적 변동·동시수축
        self.motor_noise = float(motor_noise)
        self.motor_noise_floor = float(motor_noise_floor)
        # 손목 해제 모드(T1 full-sequence): 손목 잠금해제·손목근 행동추가·full-seq fPCA 참조
        self.wrist = bool(wrist)
        if self.wrist:
            from references import fpca_full_ref
            ref_gen = "fpca_full"
            self._refmod = fpca_full_ref
        else:
            self._refmod = {"fpca": fpca_ref}.get(ref_gen, warp)   # 기본 warp, 채택 fpca (vae 기각·제거)
        self.ref_gen = ref_gen
        self.task_mode = task               # 'T1' | 'T2' | 'mix'
        self.tasks = ["T1", "T2"] if task == "mix" else [task]
        self.task = self.tasks[0]           # 현재 에피소드 task(reset서 갱신)
        self.latent_mode = latent_mode      # 'fixed' | 'sample'
        self.perturb_on = perturb
        self.frame_skip = frame_skip
        self.rsi = rsi
        # anti-tremor 에스컬레이션 노브(obs크기 불변=warm-start 호환):
        #   act_lowpass>0 → 누설적분(leaky-integral) 행동필터 ctrl_t=(1-α)ctrl_{t-1}+α·u
        #   effort_pow=3 → cubic effort(고활성 강벌점)
        self.act_lowpass = float(act_lowpass)
        self.effort_pow = int(effort_pow)
        self.curriculum_k = list(curriculum_k)
        self.curriculum_k_weights = list(curriculum_k_weights)
        self._eval = False                  # 평가 모드(RSI off, fixed latent)

        lock = MG.get_cfg(self.wrist)["lock"]    # 손목 모드면 손가락만 잠금(손목 자유)
        mod = (joint_damping_mult != 1.0 or armature_mult != 1.0)
        self.model = build_locked_model(xml, cache=not mod, lock_joints=lock)
        if mod:   # 어깨·팔꿈치·전완 DoF에만(원위 잠금 관절 제외)
            for n in (MG.TRACK_DOF + MG.FREE_DOF):
                dof = self.model.jnt_dofadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                self.model.dof_damping[dof] *= joint_damping_mult
                self.model.dof_armature[dof] *= armature_mult
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * frame_skip
        self.n_steps = int(round(horizon_s / self.dt))

        ids = MG.resolve(self.model, wrist=self.wrist)
        self.action_idx = np.array(ids["action_idx"])
        self.perturb_idx = ids["perturb_idx"]
        self.track_dof = ids["track_dof"]                 # 모드별 추적 DoF
        self.track_w = ids["track_w"]; self.track_band = ids["track_band"]
        self.track_kout = ids["track_kout"]
        self.track_qadr = ids["track_qadr"]
        self.track_dadr = ids["track_dadr"]
        self.rot_qadr = ids["rot_qadr"]; self.rot_dadr = ids["rot_dadr"]
        self.n_act = len(self.action_idx)                 # 26 또는 32(손목)
        self.n_perturb = len(self.perturb_idx)            # 10 또는 12(손목)

        # 명목 gainprm/biasprm 백업(섭동 복원용)
        self._gain0 = self.model.actuator_gainprm.copy()
        self._bias0 = self.model.actuator_biasprm.copy()

        # reward 계수
        #  c_jerk: 원시 qacc는 중력보상 가속 포함 → 작게(정상 거상 과벌점 방지).
        #  주 anti-tremor는 dctrl(행동평활)/CAPS. settle은 hold서 정지 유도.
        c = dict(w_effort=0.05, c_jerk=5.0e-6, c_dctrl=0.5, c_settle=1.0,
                 w_rot=0.5, w_vel=0.0, term_elv=np.radians(50),
                 term_elb=np.radians(60), term_qvel=40.0)
        if reward_cfg:
            c.update(reward_cfg)
        self.rc = c

        self.rng = np.random.default_rng(seed)
        self.lookahead = 5                   # 참조 선행(스텝)

        # 공간 정의 (obs는 한번 reset 후 크기 확정)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_act,), np.float32)
        self._fixed_latents = {t: self._median_latent(t) for t in self.tasks}
        obs = self.reset(seed=seed)[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)

    # ---- 라텐트/참조 ----
    def _median_latent(self, task):
        if self.ref_gen in ("vae", "fpca"):
            return self._refmod.median_latent(task)
        import json
        with open(os.path.join(os.path.dirname(warp.__file__), "out",
                               f"latent_{task}.json")) as fh:
            L = json.load(fh)
        return {"T": L["T"]["p50"], "peak": L["peak"]["p50"],
                "skew": L["skew"]["p50"], "plane": L["plane_nominal"],
                "elbow_peak": L["elbow_peak"]["p50"]}

    def set_eval(self, flag=True, sample_latent=False):
        self._eval = flag
        self._eval_sample_latent = sample_latent   # M2 게이트: 분포 전반 평가

    def _sample_latent(self):
        if self._eval:
            if getattr(self, "_eval_sample_latent", False):
                return self._refmod.sample_latent(self.task, self.rng)
            return dict(self._fixed_latents[self.task])
        if self.latent_mode == "fixed":
            return dict(self._fixed_latents[self.task])
        return self._refmod.sample_latent(self.task, self.rng)

    def set_perturbation(self, s_F=None, s_L=None):
        """제어된 고정 섭동 설정(진단·M4 데이터생성용). None이면 해제(랜덤복귀).

        s_F, s_L: shape (N_PERTURB_CH=10,). reset마다 이 값으로 적용(eval서도).
        """
        if s_F is None:
            self._fixed_perturb = None
        else:
            self._fixed_perturb = (np.asarray(s_F, float).copy(),
                                   np.asarray(s_L, float).copy())

    def _apply_perturbation(self):
        """C 섭동: 표적 채널에 s_F(force)·s_L(length range) 적용. 건강이면 전부 1."""
        self.model.actuator_gainprm[:] = self._gain0
        self.model.actuator_biasprm[:] = self._bias0
        self.s_F = np.ones(self.n_perturb)
        self.s_L = np.ones(self.n_perturb)
        # 고정 섭동(진단/데이터생성) 우선
        fp = getattr(self, "_fixed_perturb", None)
        if fp is not None:
            self.s_F[:] = fp[0]; self.s_L[:] = fp[1]
            self._write_perturb_to_model()
            return
        if not self.perturb_on or self._eval:
            return
        # 희소-k: 손상 채널 수 k 선택 후 해당 채널만 약화
        k = int(self.rng.choice(self.curriculum_k, p=self._k_weights_norm()))
        chans = self.rng.choice(self.n_perturb, size=k, replace=False)
        for ch in chans:
            self.s_F[ch] = float(self.rng.uniform(0.05, 1.0))
            self.s_L[ch] = float(self.rng.uniform(0.70, 1.20))
        self._write_perturb_to_model()

    def _write_perturb_to_model(self):
        """self.s_F/s_L(채널별)을 모델 gainprm/biasprm에 적용(명목 기준 절대)."""
        for ch in range(self.n_perturb):
            sF = self.s_F[ch]; sL = self.s_L[ch]
            if sF == 1.0 and sL == 1.0:
                continue
            for ai in self.perturb_idx[ch]:
                self.model.actuator_gainprm[ai, 2] = self._gain0[ai, 2] * sF
                self.model.actuator_biasprm[ai, 2] = self._bias0[ai, 2] * sF
                self.model.actuator_gainprm[ai, 0:2] = self._gain0[ai, 0:2] / sL
                self.model.actuator_biasprm[ai, 0:2] = self._bias0[ai, 0:2] / sL

    def _k_weights_norm(self):
        w = np.array(self.curriculum_k_weights[:len(self.curriculum_k)], float)
        return w / w.sum()

    def set_curriculum(self, ks, weights):
        self.curriculum_k = list(ks); self.curriculum_k_weights = list(weights)

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if len(self.tasks) > 1:                 # mix: 에피소드마다 task 선택
            self.task = self.tasks[int(self.rng.integers(len(self.tasks)))]
        mujoco.mj_resetData(self.model, self.data)
        self.latent = self._sample_latent()
        self._apply_perturbation()
        ref = self._refmod.make_reference(self.task, self.latent, self.n_steps, self.dt)
        self.ref = {k: np.radians(v).astype(np.float64) for k, v in ref.items()}
        # pro_sup 참조는 시작=0 상대 → 중립(0)에서 시작하는 상대각
        self.rise_steps = int(np.clip(self.latent["T"] / self.dt, 1, self.n_steps))

        # RSI: 학습 시 임의 위상에서 시작(평가는 위상0=rest)
        if self.rsi and not self._eval:
            self.t_idx = int(self.rng.integers(0, max(1, self.n_steps - 5)))
        else:
            self.t_idx = 0
        self._set_pose_to_ref(self.t_idx, noise=(self.rsi and not self._eval))
        self.prev_action = np.zeros(self.n_act)
        self.prev_qvel = self.data.qvel.copy()
        self._ctrl_prev = np.zeros(self.model.nu)
        return self._obs(), {}

    def _set_pose_to_ref(self, idx, noise=False):
        q = self.data.qpos
        for n in self.track_dof:
            v = self.ref[n][idx]
            if noise:
                v = v + self.rng.normal(0, np.radians(3.0))
            q[self.track_qadr[n]] = v
        # shoulder_rot: 작은 임의 초기(자유)
        q[self.rot_qadr] = self.rng.normal(0, np.radians(5.0)) if noise else 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        u = 0.5 * (action + 1.0)                        # [-1,1]→[0,1] 근육 활성
        if self.motor_noise > 0.0:                      # 신호의존 운동노이즈(SD∝u)
            sd = self.motor_noise * u + self.motor_noise_floor
            u = np.clip(u + self.rng.normal(0.0, sd), 0.0, 1.0)
        if self.act_lowpass > 0.0:                      # 누설적분(anti-tremor)
            a_prev = self._ctrl_prev[self.action_idx]
            u = (1 - self.act_lowpass) * a_prev + self.act_lowpass * u
        ctrl = np.zeros(self.model.nu)
        ctrl[self.action_idx] = u
        self._ctrl_prev = ctrl
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.t_idx += 1
        rew, info = self._reward(action)
        self.prev_action = action.copy()

        terminated = info["diverged"] or not np.all(np.isfinite(self.data.qpos))
        truncated = self.t_idx >= self.n_steps - 1
        return self._obs(), float(rew), bool(terminated), bool(truncated), info

    # ---- 관측 ----
    def _q(self, name):
        return float(self.data.qpos[self.track_qadr[name]])

    def _qd(self, name):
        return float(self.data.qvel[self.track_dadr[name]])

    def _obs(self):
        i = min(self.t_idx, self.n_steps - 1)
        ia = min(self.t_idx + self.lookahead, self.n_steps - 1)
        q = np.array([self.data.qpos[self.track_qadr[n]] for n in self.track_dof]
                     + [self.data.qpos[self.rot_qadr]])
        qd = np.array([self.data.qvel[self.track_dadr[n]] for n in self.track_dof]
                      + [self.data.qvel[self.rot_dadr]])
        act = self.data.act[self.action_idx]
        ref_now = np.array([self.ref[n][i] for n in self.track_dof])
        ref_err = np.array([self.ref[n][i] - self.data.qpos[self.track_qadr[n]]
                            for n in self.track_dof])
        ref_ahead = np.array([self.ref[n][ia] for n in self.track_dof])
        phase = i / max(1, self.n_steps - 1)
        in_hold = 1.0 if self.t_idx >= self.rise_steps else 0.0
        L = self.latent
        lat = np.array([L["T"] / 3.0, L["peak"] / 120.0, L["skew"] / 2.0,
                        L["plane"] / 45.0, L["elbow_peak"] / 130.0])
        perturb = np.concatenate([self.s_F, self.s_L])      # 20 (건강=ones)
        task_flag = 1.0 if self.task == "T2" else 0.0
        obs = np.concatenate([
            q / np.pi, qd / 5.0, act,
            ref_now / np.pi, ref_err / np.pi, ref_ahead / np.pi,
            [phase, in_hold], lat, perturb, [task_flag],
        ]).astype(np.float32)
        return obs

    # ---- 보상 ----
    @staticmethod
    def _band_gauss(err, band, k):
        e = max(0.0, abs(err) - band)
        return np.exp(-k * e * e)

    def _reward(self, action):
        i = min(self.t_idx, self.n_steps - 1)
        # TASK_track
        task = 0.0
        errs = {}
        for n in self.track_dof:
            err = self.data.qpos[self.track_qadr[n]] - self.ref[n][i]
            errs[n] = err
            task += self.track_w[n] * self._band_gauss(err, self.track_band[n],
                                                       self.track_kout[n])
        # QUALITY: jerk(가속) + dctrl(행동평활) + settle(정지)
        td = [self.track_dadr[n] for n in self.track_dof]
        qacc = self.data.qacc[td]
        jerk = float(np.mean(qacc ** 2))
        dctrl = float(np.mean((action - self.prev_action) ** 2))
        in_hold = self.t_idx >= self.rise_steps
        qvel_track = self.data.qvel[td]
        settle = float(np.mean(qvel_track ** 2)) if in_hold else 0.0
        # 속도/위상 추적(escalation #4): 참조속도와 어긋나면 벌점 → 과속(racing) 억제
        vel_pen = 0.0
        if self.rc["w_vel"] > 0.0:
            j = max(1, i)
            for n in self.track_dof:
                ref_v = (self.ref[n][j] - self.ref[n][j - 1]) / self.dt
                q_v = self.data.qvel[self.track_dadr[n]]
                vel_pen += self.track_w[n] * (q_v - ref_v) ** 2
            vel_pen *= self.rc["w_vel"]
        quality = np.exp(-(self.rc["c_jerk"] * jerk + self.rc["c_dctrl"] * dctrl
                           + self.rc["c_settle"] * settle + vel_pen))
        # EFFORT (곱 밖 가산, 음수 비용; effort_pow=3 → cubic)
        act_mag = float(np.mean(self.data.act[self.action_idx] ** self.effort_pow))
        effort = -self.rc["w_effort"] * act_mag
        # shoulder_rot 약정칙화
        rot = float(self.data.qpos[self.rot_qadr])
        rot_pen = -self.rc["w_rot"] * max(0.0, abs(rot) - MG.ROT_FREE_BAND) ** 2  # rot밴드 공용

        reward = task * quality + effort + rot_pen

        diverged = (abs(errs["shoulder_elv"]) > self.rc["term_elv"]
                    or abs(errs["elbow_flexion"]) > self.rc["term_elb"]
                    or np.abs(qvel_track).max() > self.rc["term_qvel"])
        if diverged:
            reward -= 1.0
        info = {"task": task, "quality": quality, "effort": effort,
                "err_elv_deg": np.degrees(errs["shoulder_elv"]),
                "err_elb_deg": np.degrees(errs["elbow_flexion"]),
                "diverged": bool(diverged), "in_hold": bool(in_hold)}
        return reward, info


def make_env(**kw):
    return ArmPerturbEnv(**kw)
