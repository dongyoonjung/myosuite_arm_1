# myoArm 어깨-거상 — 근골격 파라미터 추정(회귀) 개발

myoArm(MuJoCo/MyoSuite)에서 근육 파라미터(Fmax·Lopt scale)를 흔든 뒤, 시뮬 롤아웃만 보고
**어느 근육이 얼마나 약해졌는지 복원하는 회귀기**를 개발한다. sim 내부 식별성(sim-to-real 아님).
실데이터(KIMHu)의 표준 거상동작(T1 관상·T2 횡단)을 파라미터화해 다양한 정상 궤적을 만들고,
정책이 그 참조를 *낮은 강도로 soft 추적* — 약화가 **이탈(KIN)** 또는 **활성(ACT)**으로 드러난다.

설계 초안은 [`DESIGN.md`](DESIGN.md). 작업원칙은 [`CLAUDE.md`](CLAUDE.md). **현재 유효 상태·결정은 [`SESSION_LOG.md`](SESSION_LOG.md) 0절**(설계 이후 참조생성·입력 방식이 개정됨). 레포 구조 지도는 [`STRUCTURE.md`](STRUCTURE.md).

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

## 결과 (정본 = `SESSION_LOG.md` 0절, 원칙 = `CLAUDE.md`)
- **현재 최선: 같은 피험자 T1+T2 페어 → 공유 θ, 관측가능 운동학만(err·참조latent 제외), s_F 평균 R² = 0.715.**
- 진행: err 포함 단일 0.81(참조 누수·폐기) → err 제외 단일 0.63 → **T1+T2 페어 0.715**(약식별 채널 개선). latent 제거해도 불변 → 누수 비의존(robust).
- 손목해제+full-seq는 손목근 식별(~0.8) 추가하나 어깨/팔꿈치 희석 → 기본은 손목 잠금.
- 별도 보고서 파일은 폐기(정보는 MD 문서에 통합). 분석 그림 `tools/fig_*.py`, 비디오 `results/report/M1_T1.mp4`.
- **데이터 보존**: `data/pair/train.npz`(삭제 금지; 재생성 `bash tools/run_pair.sh`).

## 실행 환경
- venv `.venv`(또는 conda). MyoSuite 2.11.6 / Gym 0.29.1 / MuJoCo 3.3.0 / SB3 2.7.1 / torch CPU. `OMP_NUM_THREADS=1`.
- 학습·데이터생성·회귀 전부 이 머신(32코어) 직접 수행.

## 빠른 실행
```bash
# 1) 참조 생성기(fPCA) — KIMHu 원시 필요(tools/download_kimhu.py)
.venv/bin/python -m references.extract_traj && .venv/bin/python -m references.fpca_ref
# 2) 정책 학습(rise·fPCA·운동노이즈)
OMP_NUM_THREADS=20 .venv/bin/python train.py --task mix --latent-mode sample --perturb \
  --ref-gen fpca --motor-noise 0.1 --out ppo_arm_M3 --timesteps 4000000 --n-envs 20
# 3) ★페어 데이터(같은 피험자 T1+T2, 공유 θ) → 공유 회귀(err 자동 제외)
.venv/bin/python gen_pair.py --model models/ppo_arm_M3.zip --vecnorm models/ppo_arm_M3_vec.pkl \
  --n 8000 --ref-gen fpca --motor-noise 0.1 --out data/pair/train.npz   # (병렬 샤드는 tools/run_pair.sh)
.venv/bin/python regress_pair.py --data data/pair/train.npz --epochs 120
# (단일과제 비교: gen_seq_data.py --kin-only 1 → regress_seq.py --channels all --no-err)
```
