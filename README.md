# myoArm 어깨-거상 — 근골격 파라미터 추정(회귀) 개발

myoArm(MuJoCo/MyoSuite)에서 근육 파라미터(Fmax·Lopt scale)를 흔든 뒤, 시뮬 롤아웃만 보고
**어느 근육이 얼마나 약해졌는지 복원하는 회귀기**를 개발한다. sim 내부 식별성(sim-to-real 아님).
실데이터(KIMHu)의 표준 거상동작(T1 관상·T2 횡단)을 파라미터화해 다양한 정상 궤적을 만들고,
정책이 그 참조를 *낮은 강도로 soft 추적* — 약화가 **이탈(KIN)** 또는 **활성(ACT)**으로 드러난다.

설계 전모는 [`DESIGN.md`](DESIGN.md). 핵심 원칙·결정은 [`CLAUDE.md`](CLAUDE.md).

## 마일스톤 (F 로드맵) — 진행상태는 `SESSION_LOG.md`
- **M0 ✅** 참조 파이프라인 — `references/`. KIMHu→myoArm 각도 참조 + latent(T·peak·skew·plane) 분포.
- **M1 ✅** 건강·단일·고정 정책 + **검증 게이트** (`train.py` + `diagnostics/gate.py`). 통과(정점 97°·RMS 2.8°). `models/ppo_arm_T1.zip`.
- **M2 ✅** latent 다양성 + T2 (`--latent-mode sample`, `--task mix`). 분포게이트 통과(T1 peakErr 3°·T2 5°). `models/ppo_arm_M2b.zip`.
- **M3 🔄** 근육 섭동 커리큘럼(k=1→k≤3, `--perturb`). 신호 진단 `diagnostics/signal.py`. 식별 메커니즘 검증됨(경증=ACT보상·중증=KIN이탈).
- **M4** 스윕 롤아웃 → 특징 추출 (`gen_data.py` + `tools/concat_shards.py`).
- **G** 회귀기 (`regress_nn.py`) — 채널별 식별성(R²/MAE) 보고.

## 실행 환경
- conda `rl_myosuite` (로컬) 또는 Docker 이미지(VM). MyoSuite 2.11.6 / Gym 0.29.1 / MuJoCo 3.3.0 / SB3 2.7.1 / torch CPU.
- `OMP_NUM_THREADS=1` (오버서브스크립션 회피).
- 학습/데이터생성은 GCP CPU spot VM(`cloud/`), 데이터·체크포인트는 GCS.

## M0 빠른 실행 (로컬)
```bash
conda activate rl_myosuite
python -m references.build_templates   # 참조 템플릿 + latent 분포 생성 → references/out/
```
