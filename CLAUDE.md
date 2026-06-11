# CLAUDE.md — myoArm 근골격 파라미터 추정 프로젝트

전체 설계는 `DESIGN.md`. 이 파일은 작업 시 반드시 지킬 **핵심 원칙·결정·함정**만.

## 프로젝트 정체 (흐리지 말 것)
- 이건 **근골격 파라미터 추정(회귀) 개발**이다. 과제(T1/T2)는 **고정 입력**이지 식별성 끌어올리는 튜닝 손잡이가 아니다.
- 목표 = sim 내부 식별성(sim-to-real 아님). 파라미터 ground truth는 sim 전용.
- **데이터에 없는 조건을 발명하지 말 것**(90° 평면·부하 등 금지). 배터리 = KIMHu 실재 2과제(T1 관상~0°, T2 횡단~45°)뿐. 부하 없음(프로토콜 확인). 천장이 낮으면(DELT1/DELT3/cuff 약식별) 그게 측정 결과지 결함 아님.

## ★ 설계철학 개정 (최신 — DESIGN.md 초안보다 우선)
사용자 방향으로 핵심 설계가 진화함. **아래가 현재 유효 결정**(DESIGN.md 초안의 ACT·warp는 폐기):
1. **입력 = 운동학(KIN)만**. EMG/ACT 드롭 — 실제 환자에서 모션캡처로 *관측 가능한 것만* 쓴다(만성환자=CNS 적응이라 privileged 정책은 정당하나, **식별 입력**은 관측가능에 한정).
2. **추적오차(err) 제외 + 참조 latent 제외**. `err=q_ref−q`와 동작 latent(T·peak·skew·plane)는 *의도된 참조 궤적*을 인코딩 → 실환자 미지 → 참조 누수(privileged). 둘 다 회귀 입력에서 제거. 공변량은 **task_flag(어느 과제 지시=기지)만** 허용. (검증: latent 제거해도 R² 불변 → 모델이 누수 비의존, 운동학으로 식별.)
3. **참조 생성 = fPCA**(VAE보다 분포 충실). warp→VAE→fPCA로 진화.
4. **★같은 피험자 = 고정 θ로 T1·T2 둘 다 수행**. 한 궤적이 아니라 **두 시행(T1+T2)을 함께 입력해 공유 파라미터 θ를 추정**해야 함(`gen_pair.py`·`regress_pair.py`: 공유 TCN 인코더 + 어텐션 풀링, 순열불변). T1(관상~0°)·T2(횡단~45°)는 근부하가 달라 *상보적 식별*. θ만 피험자 고정, 참조 style(latent)은 시행별 독립 fPCA 샘플.
5. **운동노이즈**(Harris–Wolpert, env `motor_noise`) 적용. 손목 해제(env `wrist`)는 탐색결과 트레이드오프(손목근 식별↑/어깨·팔꿈치↓)라 기본은 손목 잠금.

## 환경
- conda `rl_myosuite`. import 순서: `import myosuite` → `from myosuite.utils import gym` → SB3.
- 버전 고정: MyoSuite 2.11.6 / Gym 0.29.1 / MuJoCo 3.3.0 / SB3 2.7.1 / torch CPU. `OMP_NUM_THREADS=1`.
- myoArm: `$SIMHIVE/myo_sim/arm/myoarm.xml`. nq=38(독립27+커플11), nu=63.

## LOCKED 결정 (요약 — 상세는 DESIGN.md)
- **섭동(C)**: Fmax+Lopt 둘 다, `s_F~U[0.05,1.0]`·`s_L~U[0.70,1.20]`. 10채널×2=20-dim. 개별표적 8(DELT1/2/3,SUPSP,PECM1,CORB,BIClong,TRIlong) + 묶음2(A{INFSP,SUBSC,TMIN}, B{TMAJ,LAT1/2/3}). 희소-k(1~3, 1~2에 무게), 나머지 ~정상. 커리큘럼 k=1→k≤3.
- **행동공간 26근**: 어깨15+팔꿈치9+PT·PQ(회내, pro_sup용). 손목·손가락 잠금(원위 37근 제거).
- **DoF 추적 가중**: shoulder_elv 0.15 / elv_angle 0.15 / elbow_flexion 0.25 / pro_sup 저가중(데이터 참조) / **shoulder_rot 저가중 자유·참조 없음**(물리가 외회전 공급) / 손목·손가락 잠금.
- **reward**: `TASK·QUALITY + w_e·EFFORT + penalty`, `TASK=Σ wᵢ exp(−k(q−q_ref)²)`. 추적 허용폭 = 양면(밴드 안 평평 / 밖 가파름).
- **참조(A1/D) = fPCA/ProMP** (`references/fpca_ref.py`·full-seq `fpca_full_ref.py`). KIMHu 궤적에 functional PCA(95%분산 K≈13–24) + 점수공간 KDE(h=0.35)로 생성 → 평균·공분산을 구조적으로 보존(분포 충실, under-dispersion 없음). VAE(`vae_ref.py`)는 비교군(분포 과소→기각). warp(평균+1PC)은 구버전. ★3D Joints.Position에서만 유도(사전계산각 금지), 단조상승만 분절(full-seq는 복귀 포함). thoracohumeral→shoulder_elv 1:1.
- **회귀 특징(G) = 운동학(KIN)만** (`gen_seq_data.py --kin-only`, `regress_seq.py` 시계열 TCN). EMG/ACT 미사용(↓아래 철학개정). **★추적오차(err=q_ref−q) 채널 제외**(`--no-err`): err는 *의도 궤적* 누수라 관측 불가 → 빼야 정당. 시계열(TCN)이 요약통계(스냅샷)보다 압도적(0.36→).

## ★검증 게이트 (M1, 임계경로)
약화 데이터 생성 전, 명목파라미터(s_F=s_L=1) 정책이 인간 템플릿을 재현하는지 통과해야 함:
정점 인간 91±11°내 · 단조상승(반전0) · 인간대역 속도 · 4–12Hz 떨림 없음 · 추종 RMS 허용내.
실패 시 escalation: ①넓은밴드 강제종료(myodm식) ②anti-tremor 스택(CAPS·cubic effort·leaky-integral+gravity obs·settle) ③RSI ④속도/위상 추적항 ⑤(최후)추적가중↑. **통과 전 다음 단계 금지.**

## 개발 워크플로
- 로컬(WSL): 코드·M0 실행·경량 디버그 → `git push`.
- GCP spot VM: GitHub pull → Docker로 M1–M4 학습. 체크포인트→GCS. Claude Code(`claude -p`, tmux)는 **모니터/디버그 조수**지 학습 생존 주체 아님(spot 회수=세션 죽음 → 생존은 체크포인트+시작스크립트가 책임).

## 파일 저장 컨벤션
- 참조: `references/out/template_{T1,T2}.npz`, `latent_{T1,T2}.json`
- 모델: `models/{algo}_arm_*.zip` + `models/vec_normalize_*.pkl`(함께)
- 체크포인트: `models/checkpoints/`, 로그 `logs/`(TensorBoard)

## 팔꿈치 레포 계승
`../myosuite_elbow_basic/`: elbow_perturb_v0.py(reward·섭동), train.py, diag_movement.py/diag_fft.py(검증게이트 계승), regress_nn.py(G 계승), caps_ppo.py(CAPS 평활).
