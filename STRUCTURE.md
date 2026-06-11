# 레포 구조 안내 (STRUCTURE)

myoArm 근육 파라미터(Fmax·Lopt scale) 추정 프로젝트. **무엇이 어디 있는지**의 지도.
설계원칙=`CLAUDE.md`, 현재상태·결과=`SESSION_LOG.md`(정본), 초기설계=`DESIGN.md`(개정됨).

## 파이프라인 한눈에
```
KIMHu 실데이터 ──[references/]──► fPCA 참조 생성기 (모방 대상 궤적)
                                        │
                  [train.py]── RL 정책 학습(PPO, soft-tracking) ──► models/ppo_arm_*.zip
                                        │
        [gen_pair.py] 같은 피험자 T1+T2 롤아웃(θ 고정) ──► data/pair/train.npz
                                        │
        [regress_pair.py] 공유 인코더+어텐션 풀링 ──► 공유 θ 추정 (현 최선 R²=0.715)
```

## 디렉터리·파일

### 📄 문서 (루트)
| 파일 | 내용 |
|---|---|
| `CLAUDE.md` | 핵심 원칙·LOCKED 결정·설계철학 개정 (작업 시 필독) |
| `SESSION_LOG.md` | **현재 상태·결과·결정의 단일 정본** (0절) |
| `DESIGN.md` | 초기 설계 초안 (상단에 개정 주의 — 충돌부 폐기) |
| `README.md` | 진입점·마일스톤·빠른 실행 |
| `STRUCTURE.md` | (이 파일) 레포 구조 지도 |
| `GRILL_HANDOFF.md` | grill 설계인터뷰 스냅샷 (구버전 참고용) |
| `Dockerfile`·`requirements.txt` | 실행 환경 |

### 🦾 `custom_envs/` — 시뮬 환경
| 파일 | 내용 |
|---|---|
| `arm_perturb_v0.py` | MuJoCo myoArm 환경: 관측(76/96)·행동(26/32)·보상(soft밴드)·섭동(약화 주입)·운동노이즈·손목해제. RL의 핵심. |
| `muscle_groups.py` | 행동근 26/32, 섭동채널 10/12, 추적 DoF·가중·밴드, 손목 모드 설정, `resolve()`. |

### 🎯 `references/` — 모방 대상(참조 궤적) 생성
| 파일 | 내용 |
|---|---|
| `kimhu_io.py` | KIMHu 3D 관절위치 → 각도(거상·팔꿈치·평면·회내·손목) 유도. |
| `extract_traj.py`·`extract_traj_full.py` | 반복별 rise / full-cycle 궤적 추출 → `out/traj*.npz`. |
| **`fpca_ref.py`·`fpca_full_ref.py`** | **채택**: functional PCA + 점수공간 KDE 생성기(분포 충실). |
| `warp.py` | 구 참조(평균+1PC 워핑) — env 기본 fallback·경로해석에 잔존. |
| `segment.py`·`similarity.py`·`build_templates.py`·`viz_pipeline.py` | 분절·유사도·M0 템플릿·시각화 보조. |
| `out/` | 생성 산출물: `fpca*_{T1,T2}.npz`·`traj*_{T1,T2}.npz`(현용), `template/latent`(warp용). |

### 🧠 학습·데이터·회귀 (루트)
| 파일 | 내용 |
|---|---|
| `train.py` | PPO 정책 학습(커리큘럼·warm-start·게이트 콜백). `--ref-gen/--wrist/--motor-noise`. |
| `caps_ppo.py` | CAPS 공간평활 PPO(anti-tremor escalation 옵션). |
| **`gen_pair.py`** | **현용**: 피험자 단위 T1+T2 페어(θ 고정) 롤아웃 → `data/pair/`. |
| `gen_seq_data.py` | 단일과제 시계열 롤아웃(KIN-only, `--no-err` 호환). |
| `gen_data.py` | 구 스냅샷 요약통계 데이터(레거시). |
| **`regress_pair.py`** | **현 최선**: 공유 TCN 인코더+어텐션 풀링 → 공유 θ. |
| `regress_seq.py` | 단일과제 시계열 TCN 회귀(`--no-err`로 참조누수 제거). |
| `regress_nn.py` | 구 스냅샷 MLP 회귀(레거시). |

### 🔬 `diagnostics/` — 검증
| 파일 | 내용 |
|---|---|
| `gate.py` | M1 검증게이트(정점·단조·속도·떨림·RMS). |
| `signal.py` | 섭동 신호 진단(채널별 KIN/ACT 민감도). |

### 🛠 `tools/` — 보조 스크립트
| 파일 | 내용 |
|---|---|
| `download_kimhu.py` | KIMHu 스켈레톤 CSV 다운로드. |
| `concat_pair.py`·`concat_seq.py`·`concat_shards.py` | 병렬 샤드 병합. |
| `run_pair.sh` | **현용** 페어 파이프라인(생성→회귀). |
| `run_noerr.sh` | 단일과제 err-제외 회귀. |
| `run_wrist_mix.sh`·`run_wrist_post.sh` | 손목해제 파이프라인. |
| `render_episode.py` | 에피소드 mp4 렌더. |
| `inspect_myoarm.py`·`inspect_muscle.py`·`probe_lock.py` | 모델·근육·잠금 점검. |
| `smoke_env.py`·`debug_feasibility.py`·`verify_kinematics.py`·`verify_plane_sign.py` | 환경·운동학 검증(개발용). |

### 📦 `models/` — 산출물 (git 포함, gitignore 예외)
- 정책: `ppo_arm_M3v`(현 기준)·`ppo_arm_wrist`/`wrist_mix`(손목해제).
- 회귀: **`regressor_pair.pt`(현 최선)**·`regressor_kin_noerr.pt`·`regressor_wrist`/`wrist_mix.pt`.
- 각 정책엔 `_vec.pkl`(VecNormalize) 동반.

### 🗂 `data/`·`results/` — gitignore (재생성 가능)
- `data/pair/train.npz` — **현 학습 데이터(삭제 금지·재생성 `tools/run_pair.sh`)**.
- `results/report/` — 비디오·그림(로컬, git 제외).

## 빠른 진입
1. 현재 상태·결과 → `SESSION_LOG.md` 0절.
2. 원칙·결정 → `CLAUDE.md`.
3. 실행 → `README.md` 빠른 실행 + `tools/run_pair.sh`.
