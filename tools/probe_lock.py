"""잠금 메커니즘 확정: MjSpec equality 추가 vs 런타임 필드 vs 키네마틱 리셋.
+ 런타임 gainprm 편집이 근육력에 반영되는지 확인(섭동 C 사전검증)."""
import numpy as np
import myosuite  # noqa
from myosuite.utils import gym  # noqa
import mujoco

XML = "/home/aaron/projects/myoarm_1/.venv/lib/python3.11/site-packages/myosuite/simhive/myo_sim/arm/myoarm.xml"

# --- MjSpec API 탐색 ---
spec = mujoco.MjSpec.from_file(XML)
print("=== MjSpec 메서드 (add/del/equality 관련) ===")
print([x for x in dir(spec) if not x.startswith("__") and
       any(k in x.lower() for k in ("add","del","equal","actuator","joint","find"))])
eq0 = list(spec.equalities)[0]
print("equality 요소 속성:", [x for x in dir(eq0) if not x.startswith("__")])
print("  예시 eq0.type", eq0.type, "name1", getattr(eq0,"name1",None), "objtype", getattr(eq0,"objtype",None))

# --- equality 추가 시도 ---
print("\n=== equality 추가(관절 잠금) 시도 ===")
LOCK = ["deviation","flexion","cmc_abduction","cmc_flexion","mp_flexion","ip_flexion",
        "mcp2_flexion","mcp2_abduction","pm2_flexion","md2_flexion",
        "mcp3_flexion","mcp3_abduction","pm3_flexion","md3_flexion",
        "mcp4_flexion","mcp4_abduction","pm4_flexion","md4_flexion",
        "mcp5_flexion","mcp5_abduction","pm5_flexion","md5_flexion"]
try:
    for jn in LOCK:
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = jn
        eq.name2 = ""          # 단일 관절 → 상수 잠금
        eq.data[:5] = [0,0,0,0,0]   # polycoef: 상수 0 (qpos0=0)
        eq.active = True
    m2 = spec.compile()
    print(f"  추가 후 재컴파일 OK: neq {m2.neq} (원래 11 → {11+len(LOCK)} 기대)")
    # 잠금 검증: 중력하 100스텝, 손목/손가락 고정 확인
    d2 = mujoco.MjData(m2)
    d2.ctrl[:] = 0
    for _ in range(200):
        mujoco.mj_step(m2, d2)
    qa = {n: np.degrees(d2.qpos[m2.jnt_qposadr[mujoco.mj_name2id(m2,mujoco.mjtObj.mjOBJ_JOINT,n)]])
          for n in ("shoulder_elv","elbow_flexion","flexion","deviation","mcp2_flexion")}
    print("  200스텝 후(deg):", {k: round(v,1) for k,v in qa.items()})
    print("  손목/손가락 잠김?", abs(qa["flexion"])<0.5 and abs(qa["deviation"])<0.5 and abs(qa["mcp2_flexion"])<0.5)
    SPEC_OK = True
except Exception as e:
    import traceback; traceback.print_exc()
    SPEC_OK = False

# --- 런타임 gainprm 편집 → 근육력 반영 확인 (섭동 C 검증) ---
print("\n=== 런타임 Fmax 편집 효과 (DELT1 s_F=0.3) ===")
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
a = 0  # DELT1
# 활성 최대로 주고 등척력 측정
def iso_force(model, data, act_id, ctrl=1.0, settle=20):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0; data.ctrl[act_id] = ctrl
    for _ in range(settle):
        mujoco.mj_step(model, data)
    return float(data.actuator_force[act_id])
f_nom = iso_force(m, d, a)
m.actuator_gainprm[a,2] *= 0.3
m.actuator_biasprm[a,2] *= 0.3
f_pert = iso_force(m, d, a)
print(f"  DELT1 force: nominal={f_nom:.1f}N  s_F=0.3→{f_pert:.1f}N  비율={f_pert/(f_nom+1e-9):.2f}")
print("  (비율 ~0.3이면 런타임 편집 유효)")
