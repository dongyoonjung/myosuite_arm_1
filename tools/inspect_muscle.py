"""근육 actuator 파라미터 + MjSpec 편집 가능성 + 중력하 안정성 조사."""
import numpy as np
import myosuite  # noqa
from myosuite.utils import gym  # noqa
import mujoco

XML = "/home/aaron/projects/myoarm_1/.venv/lib/python3.11/site-packages/myosuite/simhive/myo_sim/arm/myoarm.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

print("=== 옵션 ===")
print("timestep", m.opt.timestep, "| gravity", m.opt.gravity, "| integrator", m.opt.integrator)

print("\n=== actuator 동역학 (처음 5 + PT/PQ + 손목 1) ===")
DYN = {0:"none",1:"integrator",2:"filter",3:"filterexact",4:"muscle",5:"user"}
for a in [0,1,2,3,20,30,31,24,32]:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    print(f"  [{a:2d}] {name:10s} dyntype={DYN.get(int(m.actuator_dyntype[a]),'?'):10s} "
          f"actnum={m.actuator_actnum[a]} actadr={m.actuator_actadr[a]} "
          f"ctrlrange={m.actuator_ctrlrange[a]} gainprm={np.round(m.actuator_gainprm[a][:3],2)} "
          f"biasprm={np.round(m.actuator_biasprm[a][:3],2)}")
print("na (총 activation 상태)", m.na)

print("\n=== Fmax 추출 (muscle: gainprm? / 실제는 actuator_acc0·gainprm) ===")
# myo 근육: scale(Fmax)은 gainprm[2] 근처. 섭동 C: gainprm/biasprm[:,2]*=s_F.
for a in [0,3,20,15]:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    print(f"  {name:10s} gainprm={np.round(m.actuator_gainprm[a],3)}")
    print(f"  {name:10s} biasprm={np.round(m.actuator_biasprm[a],3)}")
    print(f"  {name:10s} lengthrange={np.round(m.actuator_lengthrange[a],4)}")

# 중력하 안정성: 모든 근육 0, 50스텝 후 주요각
print("\n=== 중력하 안정성 (ctrl=0, 100스텝) ===")
mujoco.mj_resetData(m, d)
JIDX = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ("elv_angle","shoulder_elv","shoulder_rot","elbow_flexion","pro_sup","flexion","deviation")}
d.ctrl[:] = 0
for step in (0, 1, 10, 50, 100):
    while True:
        if step == 0: break
        mujoco.mj_step(m, d)
        if d.time >= step * m.opt.timestep: break
    deg = {k: np.degrees(d.qpos[v]) for k,v in JIDX.items()}
    print(f"  step{step:3d}: " + " ".join(f"{k}={deg[k]:+6.1f}" for k in JIDX))
# 발산 체크
print("  qpos finite:", np.all(np.isfinite(d.qpos)), "| max|qvel|", np.round(np.abs(d.qvel).max(),2))

# MjSpec 편집 가능성
print("\n=== MjSpec 편집 테스트 ===")
try:
    spec = mujoco.MjSpec.from_file(XML)
    acts = list(spec.actuators)
    print("  spec actuators:", len(acts), "first:", acts[0].name, "last:", acts[-1].name)
    jts = list(spec.joints)
    print("  spec joints:", len(jts))
    eqs = list(spec.equalities)
    print("  spec equalities:", len(eqs))
    # 삭제 테스트: 손가락 근육 1개 삭제 후 재컴파일
    target = [a for a in acts if a.name == "FDS5"][0]
    spec.delete(target)
    m2 = spec.compile()
    print("  삭제 후 재컴파일 nu:", m2.nu, "(원래 63 → 62 기대)")
    print("  MjSpec 사용 가능 ✓")
except Exception as e:
    import traceback; traceback.print_exc()
    print("  MjSpec 실패:", e)
