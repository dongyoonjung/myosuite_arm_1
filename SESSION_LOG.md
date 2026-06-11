# SESSION_LOG — 작업 상태·인계 (GCP/SSH Claude용)

> **용도:** GCP spot VM에서 `git pull`로 레포를 받는 Claude(학습 모니터/디버그 조수)가 **지금 어디까지 됐고 다음에 뭘 하는지** 빠르게 파악하기 위한 살아있는 상태 문서.
> **정본 우선순위:** 설계 근거·결정 = `DESIGN.md`(현재 유효), 작업 원칙·함정 = `CLAUDE.md`, 본 문서 = 진행 상태/변경 로그.
> ⚠️ `GRILL_HANDOFF.md`는 **grill 설계단계(2026-06-10) 스냅샷**이라 일부 결정이 구버전(예: shoulder_rot/pro_sup "압핀", "orientation 없음")이다. **유효 결정은 항상 DESIGN.md를 신뢰**할 것.

---

## 0. 한 눈에 — 현재 상태 (2026-06-11)
- **단계: M0~M4 + G 전 단계 ✅ 완료. 파이프라인 닫힘.**
- **핵심 결과(G 회귀)**: 20k 롤아웃 → 근육별 Fmax scale(s_F) 복원 **ACT만 평균 R² 0.90·KIN만 0.36·전체 0.92**, 공변량만 ≈0(누수없음). 전 채널 강식별(R² 0.80–0.97). DESIGN "ACT 주채널·KIN 보너스" 실증.
- M1(정점98°)·M2(T1 peakErr3°·T2 5°)·M3(약화→ACT보상/KIN이탈 graded). 모델 `ppo_arm_{T1,M2a,M2b,M3a,M3b}.zip`·`regressor_G.pt`. 데이터 `data/sim/train.npz`.
- crawl-walk warm-start: M1(T1고정)→M2a(T1분포)→M2b(+T2)→M3a(섭동k=1)→M3b(k≤3). obs 76 불변. 학습 이 머신(32코어) 직접.
- 보고서 `PROGRESS_REPORT_2026-06-11.md` + 그림 `results/report/fig1~5`·비디오 `M1_T1.mp4`. 코드: `custom_envs/`·`train.py`·`diagnostics/{gate,signal}.py`·`gen_data.py`·`regress_nn.py`·`tools/`.
- 핵심 결정 전부 LOCKED(DESIGN.md). 정체성: **근골격 파라미터(per-muscle Fmax/Lopt scale) 추정(회귀) 개발**. T1/T2는 *고정 입력*이지 식별성 끌어올리는 튜닝 손잡이가 아님. 목표 = sim 내부 식별성(sim-to-real 아님), 파라미터 ground truth는 sim 전용.

## 1. 이번 세션 한 일 (2026-06-11)
어깨 3축 파이프라인 플롯(raw→가공→생성)의 의문점에서 출발 → 진단·수정 + 시퀀스 유사도 측정 도입 + KIMHu 프로토콜 문서화. **모두 시각화/검증/문서 작업이며, 데이터 생성 로직(템플릿/warp)은 정상으로 확인되어 변경 없음.**

### (a) 시각화 버그 2건 수정 — `references/viz_pipeline.py`
- **가공(분절) 곡선 끝의 가짜 급강하** = 평활 `np.convolve(mode="same")`의 0-패딩 경계 아티팩트(끝 샘플이 0과 평균돼 끌려 내려감). → `_sm`을 **edge-padding** 평활로 교체.
- **생성 곡선이 절삭 후 평탄(hold)** = 생성을 정점-이후 하강까지 포함한 창에 그려 raise-and-hold 꼬리가 노출됨(가공은 정점까지만). → 생성을 **상승 구간 T**에만 그려 가공과 같은 창으로 정렬. (raise-and-hold 자체는 의도된 환경 설계 — 정점 유지구간이 중력보상 부하로 식별성에 유리.)

### (b) T2 평면(elv_angle) 급변 = 측정 노이즈 확진 → 표시 정리
- 원인: ①저거상서 팔이 수직 → 위팔 수평성분(`plane_mag`) 작아 방위각 `arctan2`가 정의불량(`plane_mag<0.13`서 중앙편차 ~38°), ②MAYR02 1명의 순간 Kinect 추적 흔들림(−38° 스파이크).
- 수정: `kimhu_io.reliable_plane`(plane_mag<0.13 프레임만 마스킹·선형보간) + 그림용 **대표(최단조) 반복** 선택.
- **템플릿/npz 불변** — 코호트 평균은 이미 단조(편차 0°). 적대 검증: 마스킹이 신뢰프레임(plane_mag≥0.16) **0개 변경**, 거상>60° 전방스윕 **100% 보존**.

### (c) shoulder_rot 참조 = "없는 게 맞다" (재확인, LOCKED 유지)
- *제대로* 분해한 swing-twist 쿼터니언조차 **elbow_flexion과 ~0.84 교란** → Kinect로 *고립된* 위팔 축회전 측정 불가. 참조를 만들면 팔꿈치 신호가 rot 목표에 섞임 = "데이터에 없는 것 발명"(기각). 물리(거상 동반 의무적 외회전)가 시뮬에서 공급. 플롯 주석으로 *측정 한계*임을 명시.

### (d) processed↔generated 시퀀스 유사도 — `references/similarity.py` (신규)
- 측정자: **DTW(주, 순수 numpy DP) + NRMSE + Pearson r + 절대 MAE(deg)**. (env에 fastdtw/dtaidistance 없음 → 직접 구현; 동일입력→0·대칭·타이밍흡수 검증.)
- 성격 = **self-fit 표현충실도**: 각 반복서 latent(T·peak·skew·plane·elbow) 재추출 후 생성·비교 → "4-latent + 단조 템플릿"이 실 반복 형태를 얼마나 담는가. (런타임은 *sampled* latent로 생성; 미지 latent 일반화 검증은 M1/M2 게이트 몫.)
- **정직 헤드라인 = 전체 반복 분포**(대표 1개로 과장 안 함):

  | 채널 | DTW med (p90) | MAE med (p90) | r med |
  |---|---|---|---|
  | T1 거상 | 0.018 (0.026) | **8.3° (13.0°)** | +0.99 |
  | T2 거상 | 0.017 (0.040) | **7.2° (10.5°)** | +0.98 |
  | T1 평면 | 0.053 (0.108) | 2.9° (4.6°) | +0.74 |
  | T2 평면 | 0.048 (0.123) | 7.9° (12.1°) | +0.87 |

  → 4-latent+단조템플릿이 실 거상을 **거상각 ~7–8° 잔차**로 재현.

### (e) KIMHu 프로토콜 → `DESIGN.md` 기재
- 데이터셋 자체 프로토콜 문서는 레포에 없음 → **"관측(데이터서 직접 계산) vs 추론(선행지식)" 구분 표**로 작성. 핵심: **부하 없음(4가지로 확정)**, 휴식 6s→**실측 ~3.5–5s 정정**, 20명×10반복, Kinect V2 ~29.9fps, EMG Noraxon 4ch(ECU·FCU·이두·중삼각근).

### 적대적 검증 (워크플로 2건, 둘 다 "sound")
- 대표반복 선택이 유사도를 **과장하지 않음**(대표반복이 평균적 반복보다 *덜* 맞음 = 보수적). `reliable_plane`은 신뢰프레임 불변·진짜 전방스윕 보존(과잉편집 아님).

## 2. references/ 파일 지도
| 파일 | 역할 | 실행 |
|---|---|---|
| `kimhu_io.py` | KIMHu CSV 로드 + 3D 위치서 각 유도(col2-7 금지), `reliable_plane`, `weighted_circmean` | (모듈) |
| `segment.py` | 단조 상승구간(저점→정점) 분절 | (모듈) |
| `build_templates.py` | T1/T2 템플릿 + latent 분포 생성 → `out/template_*.npz`, `latent_*.json` | `python -m references.build_templates` |
| `warp.py` | 런타임 참조 생성(`make_reference`: 상승 후 정점유지). **M1 환경이 reset마다 호출** | `python -m references.warp` (단조 selftest) |
| `similarity.py` | DTW/NRMSE/MAE/r + 전체분포 표 + 정렬 그림 | `python -m references.similarity` |
| `viz_pipeline.py` | raw→가공→생성 어깨 3축 비교 플롯 | `python -m references.viz_pipeline` |
| `out/` | 템플릿 npz/json + 검증 PNG(`shoulder_axes_pipeline.png`, `similarity_proc_vs_gen.png`, `m0_templates.png`) — **커밋됨**(데이터 없이도 산출물 확인 가능) | — |

## 3. 재현/검증 커맨드 (conda `rl_myosuite`, `OMP_NUM_THREADS=1`)
```bash
python -m references.build_templates   # 템플릿 재생성 (data/kimhu/V2/ 필요)
python -m references.viz_pipeline      # 파이프라인 3축 플롯
python -m references.similarity        # 시퀀스 유사도 표 + 그림
python -m references.warp              # 참조 단조성 selftest
```
※ `data/`(KIMHu ~1.5GB)는 **gitignore** — GCS/로컬서 별도 확보. `references/out/*`는 커밋돼 있어 데이터 없이 산출물 검토 가능.

## 4. M1 구현·통과 기록 (2026-06-11, ★임계경로 통과)

### 구현 (신규 파일)
| 파일 | 역할 |
|---|---|
| `custom_envs/muscle_groups.py` | 26 행동근(어깨15+팔꿈치9+PT/PQ) idx, 섭동 10채널(개별8+묶음A/B), 잠금 22관절, 추적 DoF 가중·밴드 |
| `custom_envs/arm_perturb_v0.py` | **ArmPerturbEnv** — myoArm 로드, 원위 22관절 **MjSpec equality 상수잠금**, 행동공간 26근(나머지 ctrl=0), 곱셈형 soft-tracking reward(양면밴드), `warp` 참조 통합, RSI, 섭동 런타임 적용(gainprm/biasprm). anti-tremor 노브(act_lowpass/effort_pow/w_vel/댐핑배수) 내장(M1엔 전부 off). obs=76(q5·qd5·act26·ref4·err4·ahead4·phase2·latent5·perturb20·task1). |
| `train.py` | PPO+VecNormalize+체크포인트+**학습중 게이트 콜백**. CAPS·anti-tremor·frame_skip 인자. |
| `diagnostics/gate.py` | M1 5기준 게이트(정점/단조/속도/떨림/RMS). `--plot`. |
| `caps_ppo.py` | CAPS 공간평활 PPO(백스톱; M1엔 불필요). |
| `tools/inspect_*.py`, `verify_*.py` | 모델구조·근육·운동학·평면부호 검증 스크립트. |

### 검증된 사실 (모델 실측)
- **운동학 매핑 1:1 직접**: shoulder_elv qpos=거상각(rest offset 2.7°), elbow_flexion 0=신전, **elv_angle 부호=KIMHu plane_az 동일**(0=관상→+전방, M2 T2 스윕 정방향 확인). 부호반전 불필요.
- **실현가능성**: DELT1+2+3+SUPSP로 팔 180°까지 거상 가능(95° 여유). 단 전체 26근 동시최대=48°(길항 동시수축) → 정책이 **선택활성** 학습 필요.
- **섭동 런타임**: `gainprm/biasprm[ai,2]*=s_F`, `[ai,0:2]/=s_L` → 등척력 비율 정확히 s_F(검증). 재컴파일 불필요.
- 학습 처리량: 단일 env 944 control-steps/s(50Hz), 20 envs ~6500 fps → **3M步 ~8분**.

### ★게이트 버그 (중요 교훈 — 향후 반드시 주의)
- **증상**: 게이트가 "4–12Hz 떨림 0.31, 진폭 6.4°" FAIL을 계속 보고. anti-tremor(leaky-integral·cubic effort·CAPS·댐핑×10)가 **전혀 효과 없음**(진폭 6.4→6.0). 실제 hold 궤적은 **완벽 매끈**(98.1→91.6 단조감쇠, 부호반전 3/69).
- **진짜 원인**: `DummyVecEnv`는 `done` 시 **자동 리셋** → 롤아웃 루프가 step *후* 각도를 기록하면 마지막 done 스텝에 **리셋된 rest(≈0°) 샘플**이 섞임. 92°→0° 급강하 1샘플이 FFT에서 광대역 "떨림"으로 오독. (RMS는 elv−ref라 둘 다 0이라 무영향이었음 → 떨림만 오염.)
- **수정**: 롤아웃을 **step 전(action 결정 시점) 기록**으로 변경(`gate.py`·`train.py` 콜백 둘 다). 수정 후 동일 1.5M 체크포인트가 떨림 0.05·진폭 0.43° = **PASS ✅**.
- **교훈**: 자동리셋 VecEnv에서 에피소드 시계열을 모을 땐 post-done 샘플을 절대 포함하지 말 것. 떨림·jerk 등 *차분/스펙트럼* 지표는 경계 아티팩트에 극도로 민감.

### 게이트 정교화 2 (반전 측정 — 미세 jitter 무시)
- 2.5M 모델이 RMS 2.4°로 너무 정밀히 추종하자, 상승 중 **0.1° 수준 제어 jitter**가 "반전 2"로 오집계돼 단조성 FAIL. 실측: 최대 단일하강 0.11°, 총하강 0.73°(전체 95° 상승 중) = 거시적 완전 단조.
- 수정: `_reversals`를 **데드밴드 기반**(running-max 대비 1.5°↑ 하강만 반전)으로 교체. 인간 템플릿(구성상 단조)과 의도 일치. 미세 jitter 면역.

### M1 결론 (완료 ✅)
- **plain PPO 베이스라인 보상**(곱셈형 soft-tracking, c_jerk 5e-6, c_dctrl 0.5, c_settle 1.0, w_effort 0.05, effort_pow 2)이 게이트 통과. anti-tremor 스택 불필요(유령이었음 — 측정버그). 노브(act_lowpass/CAPS/댐핑)는 M2/M3서 *진짜* 떨림 발생 시만.
- 500k–1M에 정점 120–133° **오버슈트 transient** 후 1.5M 자가수정(원인: shoulder_elv 가중 0.15 낮아 초기엔 나머지 DoF 농락→racing). M2/M3는 warm-start로 이 transient 회피.
- **공식 모델 `models/ppo_arm_T1.zip`(+`_vec.pkl`) 저장·플롯검증 완료**(3M步). 최종 게이트: 정점 97°·반전 0·속도 1.62s·떨림 진폭 0.12°·RMS 2.84° **PASS**. 플롯 `models/ppo_arm_T1_gate.png`: 매끈한 거상+hold, shoulder_rot −60° 외회전(물리 공급, 의도대로).
- 게이트 상대모드 추가: `--sample-latent`(M2 분포) 시 G1=참조정점 매칭(|달성−참조|<12°), 속도밴드 확대. M1은 절대밴드 유지.

## 5. M2 기록 (latent 분포 + T2, 완료 ✅)
- **M2a**(T1 분포, M1 warm-start, 2.5M步): 분포 상대게이트 PASS(peakErr 10.2°·RMS 7.2°). ref_peak↔달성 상관 **0.91** = 정책이 latent 활용 확인. `models/ppo_arm_M2a.zip`.
- **M2b**(T1+T2 mix, M2a warm-start, 4M步): T1·T2 분포게이트 둘 다 PASS(T1 peakErr 3.0°·RMS 5.2° / T2 peakErr 5.2°·RMS 6.8°). T2 전방평면 스윕 빠르게 습득(1M부터 통과). `models/ppo_arm_M2b.zip` = **M3 시작점**.
- env 확장: `task='mix'`(에피소드마다 T1/T2 랜덤), `set_eval(sample_latent=True)`(분포평가), 게이트 상대모드(참조정점 매칭).
- 교훈: soft밴드 ±10°라 healthy 추적도 ±3–7° 슬롭(설계 의도). 식별은 ACT 주채널이라 OK. 중앙값-latent 절대게이트(콜백)는 살짝 오버슈트(104°) 보이나 분포 상대게이트가 진짜 지표.

## 6. M3 (섭동 커리큘럼, 완료 ✅)
- **M3a**(k=1, M2b 워밍스타트, 4M) → **M3b**(k≤3, M3a 워밍스타트, 4M). 둘 다 healthy 게이트 유지. `models/ppo_arm_M3{a,b}.zip`.
- **방법**: `train.py --task mix --latent-mode sample --perturb --curriculum-k 1[,2,3] --curriculum-w ... --resume <prev>`.
- 섭동 20-dim(s_F·s_L 10채널)이 obs(55:74)에 포함 → 정책이 privileged로 약화 알고 보상. 런타임 gainprm/biasprm 편집(`_write_perturb_to_model`).
- **신호 진단**(`diagnostics/signal.py`): 약화→KIN(정점미달·회전이탈)/ACT(보상) 검증. 건강 M2a 프리뷰 실측: DELT2 약화→19° 미달, **A_lowcuff(회전근개)→회전 66° 이탈**(DESIGN "cuff=회전신호" 확증), SUPSP→ACT 대폭보상. 근육별 뚜렷한 신호.
- **주의(VecNorm)**: M2b vecnorm은 섭동 obs(55:74)가 항상 1.0이라 분산~0 → M3서 섭동 시작 시 정규화 일시 클립(−10). VecNorm이 적응하며 회복(클립이 폭주 방지). 콜백 게이트는 healthy(섭동off) 추적 모니터.
- M3 성공기준(DESIGN): 약화→신호 발생 + 정책이 가능한 보상. 게이트 아닌 **신호 스캔으로 판정**.

## 7. M4 + G (완료 ✅)
- **M4**(`gen_data.py` + `tools/concat_shards.py`): M3b 정책으로 20 병렬샤드×1000=**20k 롤아웃**(k∈{0,1,2,3}×latent×T1/T2) → 특징 143(ACT 63근 mean/max + 운동학 11 + latent/task 6) + 라벨(s_F·s_L 20). `data/sim/train.npz`.
- **G**(`regress_nn.py`): FUSED-MLP(LayerNorm·약화 재가중) 15% 홀드아웃. **결과**: s_F 복원 ACT만 R²0.90·KIN만 0.36·전체 0.92, s_L R²0.75–0.86, 공변량만 ≈0. ablation `--features act|kin|actkin`. `models/regressor_G.pt`.
- **그림**(`tools/make_report_figs.py`·`fig_identifiability.py`): fig1 추적갤러리·fig2 latent활용·fig3 섭동신호·fig4 학습곡선·fig5 식별성R². **비디오**(`tools/render_episode.py`, MUJOCO_GL=osmesa): `M1_T1.mp4`.

## 8. 후속 가능(선택)
- 학습분포 밖 일반화·미지 정책 식별성, 2채널 EMG 제약판 천장, 회전근개 개별분해 degeneracy 정량, s_L 개선, latent별 층화.
