# SESSION_LOG — 작업 상태·인계 (GCP/SSH Claude용)

> **용도:** GCP spot VM에서 `git pull`로 레포를 받는 Claude(학습 모니터/디버그 조수)가 **지금 어디까지 됐고 다음에 뭘 하는지** 빠르게 파악하기 위한 살아있는 상태 문서.
> **정본 우선순위:** 설계 근거·결정 = `DESIGN.md`(현재 유효), 작업 원칙·함정 = `CLAUDE.md`, 본 문서 = 진행 상태/변경 로그.
> ⚠️ `GRILL_HANDOFF.md`는 **grill 설계단계(2026-06-10) 스냅샷**이라 일부 결정이 구버전(예: shoulder_rot/pro_sup "압핀", "orientation 없음")이다. **유효 결정은 항상 DESIGN.md를 신뢰**할 것.

---

## 0. 한 눈에 — 현재 상태 (2026-06-11)
- **단계: M0(참조 파이프라인) 구현 + 검증 시각화·시퀀스 유사도까지 완료.** 다음 = **M1**(myoArm 환경 + 건강 베이스라인 검증 게이트).
- **학습은 아직 시작 안 함**(M1부터 GCP). 현재 레포엔 RL 코드 없음 — `references/`(데이터→참조 파이프라인) + 검증 산출물뿐.
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

## 4. 다음 = M1 (★임계경로)
- **만들 것:** `custom_envs/arm_perturb_v0.py` — myoArm 로드, 원위 37근 actuator 제거·손목/손가락 잠금(**행동공간 26근**), soft-tracking reward(곱셈형·양면 허용폭), `references.warp` 참조 통합, thoracohumeral→shoulder_elv **1:1** 좌표 보정.
- **★건강 베이스라인 검증 게이트(M2 약화 전 선결):** 명목파라미터(s_F=s_L=1) 정책이 인간 템플릿 재현 — 정점 인간 91±11°내 · 단조상승(반전0) · 인간대역 속도 · 4–12Hz 떨림 없음 · 추종 RMS 허용내. **통과 못 하면 "이탈=신호" 전제 무효 → 진행 금지.** 실패시 escalation 사다리는 DESIGN.md "F 학습 로드맵".
- 환경 코드는 **로컬 작성·로드/reset/step/reward 테스트 가능**, 학습만 GCP(Docker, 체크포인트→GCS; spot 회수=세션 죽음이므로 생존은 체크포인트+시작스크립트 책임, Claude는 모니터/디버그).
