"""운동학 매핑 검증: elv_angle/shoulder_elv/elbow가 해부학적으로 올바른 팔 자세를 내는지.
+ references.warp 참조가 venv에서 작동하는지.

myoArm 좌표 규약 확인용: 어깨 기준 손/팔꿈치 위치로 거상·평면 방향 판정.
"""
import numpy as np
import myosuite  # noqa
from myosuite.utils import gym  # noqa
import mujoco

XML = "/home/aaron/projects/myoarm_1/.venv/lib/python3.11/site-packages/myosuite/simhive/myo_sim/arm/myoarm.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

QADR = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ("elv_angle","shoulder_elv","shoulder_rot","elbow_flexion","pro_sup")}

# 신체 분절 위치(어깨 기준 상대) — body 이름 탐색
print("=== body 목록(상지) ===")
for b in range(m.nbody):
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    print(f"  [{b}] {bn}")

def set_pose(**deg):
    mujoco.mj_resetData(m, d)
    for n, v in deg.items():
        d.qpos[QADR[n]] = np.radians(v)
    mujoco.mj_forward(m, d)

def xpos(name):
    return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

# 분절 추정: 위팔=humerus, 전완=radius/ulna, 손=hand/proximal_row? 탐색 후 픽
bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
def pick(*keys):
    for k in keys:
        for b in bodies:
            if b and k in b.lower():
                return b
    return None
B_hum = pick("humerus","upperarm")
B_uln = pick("ulna","radius","lunate_hand","forearm")
B_hand = pick("hand","proxph2","secondmc","capitate")
print("\n분절 픽:", "humerus=",B_hum, "forearm=",B_uln, "hand=",B_hand)

# 기준: 팔 내림(전부 0)에서 어깨/팔꿈치/손목 위치
set_pose(shoulder_elv=0, elv_angle=0)
sho = xpos(B_hum); elb = xpos(B_uln);
print(f"\n[내림] humerus={np.round(sho,3)} forearm={np.round(elb,3)}")
print("  (팔 내림이면 forearm.z < humerus.z 여야: forearm가 아래)", elb[2] < sho[2])

# MuJoCo world: 보통 z=up. 거상 90 측면(관상,elv_angle=0): 팔 옆으로 → x(lateral) 큼
for ea in (0, 45, 90):
    set_pose(shoulder_elv=90, elv_angle=ea)
    e = xpos(B_uln) - xpos(B_hum)  # 어깨→팔꿈치 방향(위팔)
    print(f"[elv=90 plane={ea:2d}] 위팔벡터(어깨→팔꿈치) dx={e[0]:+.3f} dy={e[1]:+.3f} dz={e[2]:+.3f}")

# elbow_flexion 효과
set_pose(shoulder_elv=90, elv_angle=0, elbow_flexion=0)
w0 = xpos(B_hand)
set_pose(shoulder_elv=90, elv_angle=0, elbow_flexion=90)
w90 = xpos(B_hand)
print(f"\nelbow 0→90: hand 이동 {np.round(w90-w0,3)} (굽히면 위치 변화 커야)")

# 거상각 검증: 위팔벡터와 '아래(-z)'의 각도 = 해부학적 elevation 이어야
print("\n=== shoulder_elv qpos ↔ 해부학적 거상각 일치? ===")
for se in (0, 30, 60, 90, 120):
    set_pose(shoulder_elv=se, elv_angle=0)
    ua = xpos(B_uln) - xpos(B_hum)
    ua = ua/ (np.linalg.norm(ua)+1e-9)
    down = np.array([0,0,-1.0])
    elev_meas = np.degrees(np.arccos(np.clip(ua@down,-1,1)))
    print(f"  qpos shoulder_elv={se:3d}° → 측정 거상각(위팔 vs 아래)={elev_meas:5.1f}°")

# references.warp 작동 검증
print("\n=== references.warp 작동 ===")
import sys; sys.path.insert(0, "/home/aaron/projects/myoarm_1")
from references import warp
rng = np.random.default_rng(0)
for task in ("T1","T2"):
    lat = warp.sample_latent(task, rng)
    ref = warp.make_reference(task, lat, n_steps=175, dt=0.02)
    print(f"  [{task}] elv {ref['shoulder_elv'][0]:.0f}→{ref['shoulder_elv'].max():.0f}° "
          f"plane {ref['elv_angle'][0]:+.0f}→{ref['elv_angle'][-1]:+.0f}° "
          f"elbow→{ref['elbow_flexion'].max():.0f}° T={lat['T']:.2f}s")
print("OK")
