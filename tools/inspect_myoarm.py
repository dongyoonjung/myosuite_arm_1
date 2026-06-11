"""myoArm 모델 구조 조사 — 관절·actuator·qpos 주소, import 순서 검증.

M1 환경 설계의 사실 기반. 실행:
  OMP_NUM_THREADS=1 .venv/bin/python tools/inspect_myoarm.py
"""
import os
import numpy as np

# LOCKED import 순서
import myosuite  # noqa: F401
from myosuite.utils import gym  # noqa: F401
import mujoco

print("=== 버전 ===")
print("numpy", np.__version__, "| mujoco", mujoco.__version__)
import myosuite as _m
print("myosuite", getattr(_m, "__version__", "?"))

# myoarm.xml 찾기
import glob
cands = []
base = os.path.dirname(_m.__file__)
for pat in ("simhive/myo_sim/arm/myoarm.xml", "**/myoarm.xml"):
    cands += glob.glob(os.path.join(base, pat), recursive=True)
cands = sorted(set(cands))
print("\n=== myoarm.xml 후보 ===")
for c in cands:
    print(" ", c)
assert cands, "myoarm.xml 못 찾음"
XML = cands[0]
print("사용:", XML)

m = mujoco.MjModel.from_xml_path(XML)
print(f"\n=== 모델 차원 ===\nnq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} nbody={m.nbody}")

print("\n=== 관절 (name | type | qposadr | dofadr | range) ===")
JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    jt = JT.get(m.jnt_type[j], str(m.jnt_type[j]))
    rng = m.jnt_range[j]
    lim = "limited" if m.jnt_limited[j] else "free"
    print(f"  [{j:2d}] {name:28s} {jt:5s} qpos={m.jnt_qposadr[j]:2d} "
          f"dof={m.jnt_dofadr[j]:2d} {lim} [{rng[0]:+.2f},{rng[1]:+.2f}]")

print("\n=== actuator/근육 (id | name | trntype | gaintype) ===")
TRN = {0: "joint", 1: "jointinparent", 2: "slidercrank", 3: "tendon", 4: "site", 5: "body"}
for a in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    tt = TRN.get(m.actuator_trntype[a], str(m.actuator_trntype[a]))
    gt = int(m.actuator_gaintype[a])
    print(f"  [{a:2d}] {name:14s} trn={tt:6s} gain={gt}")

# 핵심 관절 탐색
print("\n=== 핵심 DoF 탐색 ===")
for key in ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flex", "pro_sup",
            "deviation", "flexion", "wrist"):
    hits = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
            if key in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
    print(f"  '{key}': {hits}")

# 커플링(equality) 제약
print(f"\n=== equality 제약 neq={m.neq} ===")
ET = {0: "connect", 1: "weld", 2: "joint", 3: "tendon", 4: "flex", 5: "distance"}
for e in range(m.neq):
    et = ET.get(int(m.eq_type[e]), str(m.eq_type[e]))
    obj1 = m.eq_obj1id[e]; obj2 = m.eq_obj2id[e]
    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, obj1) if et == "joint" else obj1
    print(f"  [{e}] {et} obj1={n1} obj2={obj2} active={m.eq_active0[e] if hasattr(m,'eq_active0') else '?'}")

# 키프레임
print(f"\n=== keyframe nkey={m.nkey} ===")
for k in range(m.nkey):
    kn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k)
    print(f"  [{k}] {kn}")

# 기본 자세에서 한 스텝
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print(f"\n=== 기본 qpos (처음 12) ===\n{np.round(d.qpos[:12], 3)}")
print("OK: 모델 로드·forward 성공")
