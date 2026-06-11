"""elv_angle 부호 확정: KIMHu plane_az(+=전방) ↔ myoArm elv_angle 방향 일치?
T2 전방 스윕(0→+45°)이 해부학적으로 전방(가슴앞 수평내전)인지 확인.

전방축 판정: 흉곽/쇄골 기하로 anterior 방향 추정 후, elv_angle 증가 시
위팔 수평성분이 anterior로 가는지 확인.
"""
import numpy as np
import myosuite  # noqa
from myosuite.utils import gym  # noqa
import mujoco

XML = "/home/aaron/projects/myoarm_1/.venv/lib/python3.11/site-packages/myosuite/simhive/myo_sim/arm/myoarm.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
QADR = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ("elv_angle", "shoulder_elv")}

def xpos(name):
    return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

def set_pose(se, ea):
    mujoco.mj_resetData(m, d)
    d.qpos[QADR["shoulder_elv"]] = np.radians(se)
    d.qpos[QADR["elv_angle"]] = np.radians(ea)
    mujoco.mj_forward(m, d)

# 해부학적 축 추정 (rest pose)
set_pose(0, 0)
thorax = xpos("thorax"); clav = xpos("clavicle"); scap = xpos("scapula")
hum = xpos("humerus"); world = np.zeros(3)
print("rest body xpos:")
for nm in ("thorax","clavicle","scapula","humerus"):
    print(f"  {nm:9s} {np.round(xpos(nm),3)}")

# lateral(우측): 쇄골/견갑이 흉곽에서 뻗는 수평방향
lateral = clav - thorax
lateral[2] = 0; lateral = lateral/ (np.linalg.norm(lateral)+1e-9)
up = np.array([0,0,1.0])
# anterior = up × lateral (오른손계: 우측 팔이면 전방)
anterior = np.cross(up, lateral); anterior = anterior/ (np.linalg.norm(anterior)+1e-9)
print(f"\n추정 lateral(우)≈{np.round(lateral,2)}  anterior(전방)≈{np.round(anterior,2)}")
print("  (주의: anterior 부호는 외적순서 가정 → 아래 거상으로 교차검증)")

# 거상 90, 평면 스윕: 위팔 수평성분의 anterior 성분 부호
print("\nelv_angle 스윕 시 위팔 수평방향의 anterior 투영:")
for ea in (-45, -20, 0, 20, 45, 90):
    set_pose(90, ea)
    ua = xpos("ulna") - xpos("humerus")
    uah = ua.copy(); uah[2] = 0; uah = uah/(np.linalg.norm(uah)+1e-9)
    ant_comp = uah @ anterior
    lat_comp = uah @ lateral
    print(f"  elv_angle={ea:+4d}° → 위팔수평 anterior={ant_comp:+.2f} lateral={lat_comp:+.2f}")

print("\n판정: elv_angle 증가 → anterior 성분 증가면 KIMHu(+=전방)와 동부호(직접매핑 OK).")
print("      감소(후방)면 부호 반전 필요(qpos = -ref_plane).")

# 가슴앞 수평내전 확인: 거상90 + elv_angle +45가 손을 몸 중앙선 쪽(전방+내측)으로?
set_pose(90, 45)
hand = xpos("proxph2"); sho = xpos("humerus")
print(f"\nelv90 plane+45: hand-shoulder 수평변위 = {np.round((hand-sho)[:2],3)} (전방내전이면 anterior+ 방향)")
