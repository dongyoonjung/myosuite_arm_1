"""myoArm 근육 그룹 정의 (LOCKED, DESIGN.md C 결정).

행동공간 26근 = 어깨15 + 팔꿈치9 + PT·PQ(pro_sup 회내).
제거 37 = 손목6(ECRL,ECRB,ECU,FCR,FCU,PL) + 손31.

섭동 표적(M3, 10채널×2param=20-dim):
  개별 8 = DELT1,DELT2,DELT3,SUPSP,PECM1,CORB,BIClong,TRIlong
  묶음 A = {INFSP,SUBSC,TMIN}  (하부 회전근개 force-couple)
  묶음 B = {TMAJ,LAT1,LAT2,LAT3}  (등·내전)
명목활성 9(정책 사용·param 정상) = PECM2,PECM3,TRIlat,TRImed,ANC,SUP,BICshort,BRA,BRD
(+ PT,PQ 도 param 정상; pro_sup 구동근)
"""

# 행동공간(정책이 구동하는 26근, 이름 순서 고정)
SHOULDER_15 = ["DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
               "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB"]
ELBOW_9 = ["TRIlong", "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD"]
PRONATION_2 = ["PT", "PQ"]
ACTION_MUSCLES = SHOULDER_15 + ELBOW_9 + PRONATION_2          # 26

# 잠금(제거) 근육 — 정책이 구동 안 함(ctrl=0)
WRIST_REMOVE_6 = ["ECRL", "ECRB", "ECU", "FCR", "FCU", "PL"]
# 손가락 31은 이름 대신 "ACTION에 없는 나머지 전부"로 처리(아래 helper)

# 섭동 표적 채널(M3): (채널명, [근육들]). 개별=단일, 묶음=공유 s_F/s_L.
PERTURB_CHANNELS = [
    ("DELT1", ["DELT1"]),
    ("DELT2", ["DELT2"]),
    ("DELT3", ["DELT3"]),
    ("SUPSP", ["SUPSP"]),
    ("PECM1", ["PECM1"]),
    ("CORB", ["CORB"]),
    ("BIClong", ["BIClong"]),
    ("TRIlong", ["TRIlong"]),
    ("A_lowcuff", ["INFSP", "SUBSC", "TMIN"]),
    ("B_latadd", ["TMAJ", "LAT1", "LAT2", "LAT3"]),
]
N_PERTURB_CH = len(PERTURB_CHANNELS)  # 10

# 잠글 원위 관절(손목+손가락) — equality 상수잠금 대상 22개
LOCK_JOINTS = [
    "deviation", "flexion",
    "cmc_abduction", "cmc_flexion", "mp_flexion", "ip_flexion",
    "mcp2_flexion", "mcp2_abduction", "pm2_flexion", "md2_flexion",
    "mcp3_flexion", "mcp3_abduction", "pm3_flexion", "md3_flexion",
    "mcp4_flexion", "mcp4_abduction", "pm4_flexion", "md4_flexion",
    "mcp5_flexion", "mcp5_abduction", "pm5_flexion", "md5_flexion",
]

# 추적 DoF (참조 있음) + 가중·허용밴드(rad) — DESIGN reward 표
#   shoulder_elv 0.15 / elv_angle 0.15 / elbow 0.25 / pro_sup 0.20(저가중)
TRACK_DOF = ["shoulder_elv", "elv_angle", "elbow_flexion", "pro_sup"]
TRACK_W = {"shoulder_elv": 0.15, "elv_angle": 0.15, "elbow_flexion": 0.25, "pro_sup": 0.20}
import numpy as _np
TRACK_BAND = {  # 양면 허용폭(밴드 안=무벌점), deg→rad
    "shoulder_elv": _np.radians(10.0),
    "elv_angle":    _np.radians(15.0),   # 평면 자유=드리프트 → 넓게
    "elbow_flexion":_np.radians(12.0),
    "pro_sup":      _np.radians(20.0),   # 저신뢰·저가중 → 넓게
}
TRACK_KOUT = {  # 밴드 밖 가파름 k
    "shoulder_elv": 30.0,
    "elv_angle":    18.0,
    "elbow_flexion":22.0,
    "pro_sup":      12.0,
}

# 자유 DoF (참조 없음): shoulder_rot — 약한 정칙화로 극단/치팅만 차단
FREE_DOF = ["shoulder_rot"]
ROT_FREE_BAND = _np.radians(50.0)   # 이 안은 자유, 밖은 약벌점


# ── 손목 해제 모드(T1 full-sequence) 설정 ──────────────────────────
WRIST_MUSCLES = ["ECRL", "ECRB", "ECU", "FCR", "FCU", "PL"]   # 손목 굴근·신근
ACTION_MUSCLES_WRIST = ACTION_MUSCLES + WRIST_MUSCLES          # 32
LOCK_JOINTS_WRIST = [j for j in LOCK_JOINTS if j not in ("deviation", "flexion")]  # 손가락 20만
TRACK_DOF_WRIST = TRACK_DOF + ["flexion", "deviation"]        # +손목 굴곡·편위
TRACK_W_WRIST = dict(TRACK_W, flexion=0.10, deviation=0.10)   # 저가중(스코프밖·noisy)
TRACK_BAND_WRIST = dict(TRACK_BAND, flexion=_np.radians(15.0), deviation=_np.radians(15.0))
TRACK_KOUT_WRIST = dict(TRACK_KOUT, flexion=12.0, deviation=12.0)
PERTURB_CHANNELS_WRIST = PERTURB_CHANNELS + [
    ("W_ext", ["ECRL", "ECRB", "ECU"]), ("W_flex", ["FCR", "FCU", "PL"])]   # 12채널


def get_cfg(wrist=False):
    """모드별 설정(이름 리스트). wrist=True면 손목 해제(32근·손목추적·12섭동채널)."""
    if wrist:
        return dict(action=ACTION_MUSCLES_WRIST, lock=LOCK_JOINTS_WRIST,
                    track=TRACK_DOF_WRIST, w=TRACK_W_WRIST, band=TRACK_BAND_WRIST,
                    kout=TRACK_KOUT_WRIST, perturb=PERTURB_CHANNELS_WRIST)
    return dict(action=ACTION_MUSCLES, lock=LOCK_JOINTS, track=TRACK_DOF,
                w=TRACK_W, band=TRACK_BAND, kout=TRACK_KOUT, perturb=PERTURB_CHANNELS)


def resolve(model, wrist=False):
    """모델에서 이름→actuator index, 추적/자유 DoF→qpos addr 매핑 반환(모드별)."""
    import mujoco
    cfg = get_cfg(wrist)
    def aid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise KeyError(f"actuator 없음: {name}")
        return i
    def qadr(name):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"joint 없음: {name}")
        return int(model.jnt_qposadr[j])
    def dadr(name):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_dofadr[j])

    return {
        "action_idx": [aid(n) for n in cfg["action"]],
        "action_names": list(cfg["action"]),
        "perturb_idx": [[aid(n) for n in muses] for _, muses in cfg["perturb"]],
        "perturb_names": [c for c, _ in cfg["perturb"]],
        "track_dof": list(cfg["track"]),
        "track_w": cfg["w"], "track_band": cfg["band"], "track_kout": cfg["kout"],
        "track_qadr": {n: qadr(n) for n in cfg["track"]},
        "track_dadr": {n: dadr(n) for n in cfg["track"]},
        "rot_qadr": qadr("shoulder_rot"),
        "rot_dadr": dadr("shoulder_rot"),
    }
