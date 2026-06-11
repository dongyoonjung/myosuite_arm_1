# myoArm 어깨-거상 — 근골격 파라미터 추정(회귀) 개발

myoArm(MuJoCo/MyoSuite)에서 근육 파라미터(Fmax·Lopt scale)를 흔든 뒤, 시뮬 롤아웃만 보고
**어느 근육이 얼마나 약해졌는지 복원하는 회귀기**를 개발한다. sim 내부 식별성(sim-to-real 아님).
실데이터(KIMHu)의 표준 거상동작(T1 관상·T2 횡단)을 파라미터화해 다양한 정상 궤적을 만들고,
정책이 그 참조를 *낮은 강도로 soft 추적* — 약화가 **이탈(KIN)** 또는 **활성(ACT)**으로 드러난다.

설계 초안은 [`DESIGN.md`](DESIGN.md). 작업원칙은 [`CLAUDE.md`](CLAUDE.md). **현재 유효 상태·결정은 [`SESSION_LOG.md`](SESSION_LOG.md) 0절**(설계 이후 참조생성·입력 방식이 개정됨).

## 마일스톤 (전 단계 ✅) — 상세 진행은 `SESSION_LOG.md`
- **M0 ✅** 참조 파이프라인 — `references/`. KIMHu→myoArm 각도 참조.
- **M1 ✅** 건강·고정 정책 + 검증게이트(`train.py`,`diagnostics/gate.py`). 정점 97°·RMS 2.8°.
- **M2 ✅** latent 분포 + T2(`--latent-mode sample --task mix`). 분포게이트 통과.
- **M3 ✅** 근육 섭동 커리큘럼(k=1→k≤3, `--perturb`). 신호진단 `diagnostics/signal.py`.
- **M4 ✅** 스윕 롤아웃 → 특징(`gen_data.py`/`gen_seq_data.py` + `tools/concat_*.py`).
- **G ✅** 회귀기 — 스냅샷 `regress_nn.py`, **시계열 TCN `regress_seq.py`(우월)**.

## 방법론 개정 (현재 유효 — 사용자 방향)
- **참조(track 대상) 생성 = fPCA/ProMP** (`references/fpca_ref.py`, full-seq `fpca_full_ref.py`). warp→VAE(`vae_ref.py`)→**fPCA**로 발전; fPCA가 실제 KIMHu 분포 충실(under-dispersion 없음).
- **식별 입력 = 운동학(KIN)만** (EMG/ACT 미사용). `gen_seq_data.py --kin-only`.
- **운동노이즈**(Harris–Wolpert) env `motor_noise`. **손목 해제** env `wrist`(body 모델 손목 equality 해제, 행동 26→32근, full-seq).

## 결과·보고서
- **엄밀(수식)**: [`REPORT_technical_2026-06-11.md`](REPORT_technical_2026-06-11.md) — RL·fPCA·VAE·회귀기 입출력·구조·손실·HP.
- **서사**: `PROGRESS_REPORT_2026-06-11.md`(§9 방법론 개정). **쉬운설명/상세 Word**: `보고서_쉬운설명_*.docx`, `보고서_myoArm_*.docx`.
- 핵심: **KIN-only s_F 식별 R² 0.81**(rise·손목잠금). 손목해제+full-seq는 손목근 식별(~0.8) 추가/어깨·팔꿈치 희석. 그림 `results/report/fig1~11`, 비디오 `M1_T1.mp4`.

## 실행 환경
- venv `.venv`(또는 conda). MyoSuite 2.11.6 / Gym 0.29.1 / MuJoCo 3.3.0 / SB3 2.7.1 / torch CPU. `OMP_NUM_THREADS=1`.
- 학습·데이터생성·회귀 전부 이 머신(32코어) 직접 수행.

## 빠른 실행
```bash
# 참조 생성기(fPCA) 학습 — KIMHu 원시 필요(tools/download_kimhu.py)
.venv/bin/python -m references.extract_traj && .venv/bin/python -m references.fpca_ref
# 정책 학습(예: M3, fPCA 참조, 운동노이즈)
OMP_NUM_THREADS=20 .venv/bin/python train.py --task mix --latent-mode sample --perturb \
  --ref-gen fpca --motor-noise 0.1 --out ppo_arm_M3 --timesteps 4000000 --n-envs 20
# KIN-only 데이터 → 시계열 회귀
.venv/bin/python gen_seq_data.py --model models/ppo_arm_M3.zip --vecnorm ..._vec.pkl \
  --ref-gen fpca --kin-only 1 --motor-noise 0.1 --out data/seq/shard_0.npz   # (병렬 샤드 후 concat_seq)
.venv/bin/python regress_seq.py --data data/seq/train.npz --channels all
```
