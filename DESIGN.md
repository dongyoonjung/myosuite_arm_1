# myoArm 어깨-거상 데이터생성·식별 프로젝트 (myosuite_arm_1)

팔꿈치 프로젝트(`myosuite_elbow_basic`)의 후속. 어깨 거상(myoArm)으로 확장 + 실데이터(KIMHu) 모방 + soft motion-tracking 도입. grill-me 설계 인터뷰로 결정 누적 중(2026-06-10 시작).

> ## ⚠️ 이 문서는 초기 설계 초안 — 일부 결정 개정됨 (현재 유효 = `SESSION_LOG.md` 0절 + `CLAUDE.md` "설계철학 개정")
> 구현·실험 중 사용자 방향으로 다음이 바뀌었다. 본문 중 아래와 충돌하는 부분은 폐기:
> 1. **참조 생성: warp(평균+1PC) → fPCA/ProMP**. fPCA가 평균·공분산 보존으로 분포 충실(VAE는 under-dispersion 기각).
> 2. **식별 입력: ACT 주채널+KIN → 운동학(KIN)만**. EMG/ACT 드롭(실환자 모션캡처로 관측 가능한 것만).
> 3. **추적오차(err=q_ref−q) 입력 제외**: 의도 궤적 누수(실환자 미지) → 빼야 정당.
> 4. **단일 궤적 → 같은 피험자 T1+T2 페어로 공유 θ 추정**: 공유 TCN 인코더 + 어텐션 풀링(순열불변). T1·T2 상보부하로 식별성↑(단일 0.63 → 페어 0.71).
> 5. 스냅샷 요약통계 회귀 → 시계열 TCN. 운동노이즈(Harris–Wolpert) 적용. 손목 해제는 트레이드오프로 기본 잠금.

> **★프로젝트 정체:** 이건 **근골격 파라미터 추정(회귀) 개발**이다. 과제(T1/T2)는 **고정 입력**이지 식별성을 끌어올리는 튜닝 손잡이가 아니다. 배터리를 키워 식별성↑ 추구 = 프레이밍 오류. 천장이 낮으면 그게 측정 대상=정직한 결과.

## 변경 동기 (3가지)
1. 실 kinematic 데이터 모방 필수 → latent 요약 후 정책 obs (근육param + 궤적 동시 조정).
2. 학습은 GCP에서(로컬 아님).
3. tracking 추가 → myoArm 확장 + 그에 맞는 human data 확보.

## 확정 결정 (LOCKED)

### 목표·메커니즘
- **최종 목표 = sim 내부 식별성**(sim-to-real 아님). 질문: *"현실적 궤적변동(nuisance)을 가로질러 근육파라미터를 sim에서 복원할 수 있는가."* 실 환자 테스트셋 없음. **파라미터 ground truth는 sim 전용**(per-muscle Fmax/Lopt 라벨이 붙은 실데이터는 존재 안 함).
- **메커니즘 = latent-conditioned SOFT motion tracking.** 실데이터 거상궤적 → latent → 디코드 참조 → 정책이 *낮은 강도로* 추종.
  - **핵심 원리: "추적 실패/이탈 = 식별 신호."** 강한(완벽) 추적은 시나리오 B(궤적이 섭동 흡수)를 극대화해 운동학 신호를 지운다. soft 추적은 경증=완전보상(ACT 신호)·중증=이탈(KIN 신호)로 자연히 가른다. 임상의 "이 동작 해보세요 → 어떻게 실패/보상하나 관찰"과 동형.
  - 이로써 "일관된 기준 vs 식별성" 긴장 해소: 참조 = 표준화된 *시도*, soft 추적 + 노력비용 = 결손이 *이탈*로 발현.
- **nuisance 처리 = 조건부.** 식별모델(회귀, RL 아님)은 궤적인자/latent을 **알려진 공변량으로 입력**(sim에서 우리가 샘플하므로 known) → well-posed.
- **시나리오 = 적응 마친 만성 개인**(CNS 재적응, 근육param 고정), 다양한 self-paced 속도로 수행.

### 모델·과제·스코프
- **모델 = myoArm**(MoBL/Holzbaur). 어깨 독립 3-DoF: `elv_angle`(평면), `shoulder_elv`(거상각 0–180°), `shoulder_rot`(축회전) + `elbow_flexion` + `pro_sup`. nq=38(독립 27 + 커플링 11), nu=63근육.
- **견갑 = `shoulder_elv`에 선형커플링된 rhythm**(자동, 예: acromioclavicular_r3 = 0.396·shoulder_elv). **견갑 구동근육 부재**(SERR/TRAP/RHOMB 없음) → 견갑 병리(dyskinesis)·전거근/승모근 약화 **스코프 밖**.
- **손목·손가락 잠금** + 원위 ~40근육 actuator 제거(행동공간 축소). 식별 스코프 = **GH 근육 + 팔꿈치**.
- **과제 = 어깨 거상**(팔 들어올리기; shrug 아님). 팔꿈치는 **항상 free**(거상에 공동동원; KIMHu T1 실측 elbow 5→124°). KIMHu의 표준화 지시동작이라 일관기준 확보(자유제스처 아님).
- **식별 표적 근육**: 현실적 = DELT1/2/3 + SUPSP + 이관절 BIClong/TRIlong (+부분 PECM1/CORB). 혼동(병합) = INFSP/SUBSC/TMIN/TMAJ(하부 force-couple), TMAJ↔LAT1-3. 회전근개 회전근은 rotation 압핀으로 식별 포기.

### Reward (2D 팔꿈치 곱셈형 계승)
```
reward = TASK_track · QUALITY + w_e·EFFORT + penalties
  TASK_track = Σᵢ wᵢ · exp(−k(qᵢ − q_refᵢ)²)
  QUALITY    = exp(−(jerk + dctrl + settle))
  EFFORT     = −act_mag/na          (노력비용: 보상 쥐어짜기 억제, 곱 밖 가산)
```
**DoF별 추적 가중 wᵢ (LOCKED):**
| DoF | wᵢ | 역할 |
|---|---|---|
| shoulder_elv | **0.15** | 핵심 신호(거상근 약화→못미침) |
| elv_angle | **0.15** | 평면도 자유=신호(drift). 조건은 참조 초기평면으로만 구분 |
| elbow_flexion | **0.25** | 이관절 신호(이두/삼두 장두) |
| pro_sup | **저가중(~0.15–0.25) 신호** | KIMHu서 관측됨(아래) → 데이터 참조 추적. biceps-회외 결합이 부가 신호 |
| shoulder_rot | **저가중 자유, 참조 없음** | 데이터에 믿을 참조 없음(아래) + 압핀시 거상 막힘 → 물리가 외회전 공급. cuff 약화=회전 이탈 신호 |
| 손목·손가락 | 잠금 | — |

- **DoF 처리 근거(2026-06-10 데이터 실측으로 수정)**: KIMHu BodyInfoJson에 **JointOrientations(쿼터니언) 존재** — 과거 "orientation 없음" 전제는 *틀림*. 실측:
  - **pro_sup**: 관측됨·거상 구간 ~68° 변동, *위치기하(엄지/손끝)·쿼터니언 두 독립방법 일치*(상관 +0.56~+0.77) → 참조 신뢰 가능 → **저가중 추적 신호 DoF로 전환(압핀 철회)**. 큰 회전 대부분은 제외구간(손목조작) 산물이라 거상 구간만 retarget. 구동근 **PT·PQ 유지(행동공간 26근)**.
  - **shoulder_rot**: 두 독립방법 **불일치**(쿼터니언 excursion 4°/16° vs 위치-굽은전완 56°/43°, 상관 +0.2~+0.4) → **믿을 참조 없음**(pro_sup과 대조). 게다가 워크플로 측정: **중립 압핀시 거상이 ~90°에 갇힘**(107° 도달엔 의무적 외회전 필요 — 거상↔회전 물리 커플). → **압핀 철회, 저가중 자유(참조 없음, 약한 정칙화로 극단/치팅만 차단)**. 물리가 외회전 공급(거상 보상 받으려면 정책이 외회전 학습). 보너스: 회전근개(혼동군=최난 표적) 약화가 *회전 이탈*로 드러나 cuff 식별 ↑(압핀시엔 ACT로만 숨었음). 비용: 심한 cuff 약화시 회전제어 저하→거상 불안정 가능(RL 수렴은 sim 검증).
- `k`(추적 날카로움)·`w_e`·pro_sup/shoulder_rot 가중·추적강도 = **스윕 후보**.

### D 참조 구성 + "정상 모션 보장" (워크플로 검증, 2026-06-10)
**핵심 결론: D는 정상 모션을 자동 보장하지 않는다. 참조는 충실히 만들 수 있으나, RL 정책이 그걸 재현하는지는 검증 게이트를 통과해야만 조건부 보장.** (8-에이전트 적대검증, myoArm 모델·팔꿈치 코드·MyoSuite 선례 직접 측정.)
- **참조 구성 가드레일(전부 측정 확인)**: ①참조는 **3D Joints.Position에서만 유도** — 사전계산 각도열(col2–7) 금지(col4=흉곽상완과 ~+90° 오프셋, 상관 0.97이나 정의 다름=조용한 함정). ②**단조 상승 구간만 분절**(trough→첫 peak; 고원=손목조작은 elbow/pro_sup 채널 오염; 상승은 20명 전원 반전 0). ③시간워핑+진폭정규화 **후** 풀링(로버스트 평균, 이상치 EXT19 0.66s·I03 66.6° 처리). ④shoulder_elv 20명 전원 100% tracked(깨끗), thoracohumeral→shoulder_elv **1:1 직접**(견갑 rhythm 이중계산 아님=측정확인), 6s 오프셋은 운동학참조 무관.
- **latent 인자 = 4개(T·peak·skew=fPCA PC1·plane)로 충분**(형태잔차 PC1=68–75% 지배=skew, PC2 이하 불필요).
- **★건강 베이스라인 검증 게이트(필수 선결)**: 약화 데이터 생성 전, 명목파라미터(s_F=s_L=1) 정책이 인간 템플릿을 재현하는지 확인 — 정점 인간 91±11° 안, 단조상승(반전 0), 인간대역 속도, 4–12Hz 떨림 없음. **통과 못 하면 "이탈=신호" 전제 무효, 진행 금지.**
- **soft 추적 충실도 = 최대 위험**: 팔꿈치 선배가 *더 강한* 추적(유효 ~1.0)·*더 쉬운* 문제(정적·2DoF·5근)에서도 정상모션 실패(8Hz 떨림, 첨두 13.99 rad/s, 방향전환 16.6, 5/8 도달실패). MyoSuite 작동 모션모방(myodm)은 **sharp scale+이탈시 강제종료** 사용(우리 soft 저가중과 정반대). **완화**: 넓은-밴드 강제종료(약화 이탈은 살아남게)+RSI+CAPS 평활+팔꿈치 anti-tremor 스택 계승+속도/위상 추적항.
- **★신호 재프레임(중요)**: KIN "이탈=신호"는 약함(단일근 대개 보상가능, 심한약화도 KIN은 60–90° 통과구간에만). **그러나 회귀기는 privileged 전체 ACT를 봄 → ACT가 주채널, KIN은 보너스.** 성공기준="두 채널 중 하나로 복원" → scenario B(보상흡수) 더는 치명적 아님. **추적 허용폭 = 양면(밴드 안 평평 k_inner / 밖 가파른 k_outer)** — 단일 Gaussian이면 넓은 self-paced 밴드가 중등도 약화를 통째로 숨김.

### 로드맵
- 배터리는 **T1+T2 고정**(B 확정; 평면·부하 발명 없음). 식별성은 *주어진 표준동작에서 무엇이 복원되나*의 측정이지 끌어올릴 손잡이 아님(프로젝트 정체).
- **M0 실측 발견 + 결정(b)**: 외전 *상승만* 쓰면 T1·T2 평면이 ~7°밖에 안 갈림(둘 다 관상)=2평면 richness 소실(T2의 45° 전방은 정점-이후 가슴앞 국면에 있었음). → **(b) T2를 전방 어깨 수평내전(가슴앞) 국면까지 확장**(손목 아닌 어깨동작=발명 아님, 실데이터). 결과: 참조의 평면을 *스칼라→궤적*화. T2 plane 궤적 +44° 스윕(end ~+32°) vs T1 관상 유지(end ~+14°) → 2평면 대비 회복. (코호트 평균이 절대 전방각 희석; 추후 분절을 가슴-hold까지 좁히면 선명.)

## 데이터

### KIMHu (1차, `data/kimhu/`, CC0/CC-BY)
- ScienceDB DOI 10.57760/sciencedb.01902. 20명(오른팔), 2테스트×10반복, 반복간 휴식 **~3.5–5s 실측**('6s'는 근사·선행지식), **속도 미지시(self-paced)**. Kinect V2 30fps 25관절 **Position + JointOrientations(쿼터니언) 둘 다 존재**(과거 "orientation 없음" 전제 *오류 정정*). 단 축회전(roll) 신뢰도는 관절마다 다름: pro_sup은 위치기하와 교차검증돼 신뢰(상관 +0.56~+0.77), shoulder_rot(어깨 roll)은 교차검증 불가→불신. + EMG 1500Hz 4채널(ECU,FCU,Biceps,**Deltoideus Medius**). EMG에 20마커(반복 start/end 쌍).
- **T1=관상**: 외전→팔꿈치 머리로→신전→손목(굴/중/신)→rest. **T2=횡단**: 외전→팔꿈치굽혀 손목 가슴앞→손목→rest. 둘 다 복합시퀀스 → **외전구간 [shoulder_elv + elbow]만 사용**.
- 실측(MAYR02): T1 shoulder_elv 6→107°·평면az~3°; T2 6→103°·az~45°(앞). elbow T1 스파이크~115°/T2 지속~120°. T 중앙 2.4s(1.6–4.6), skew~0.35. → **2 실재평면(0°,45°), 순수시상면(90°)은 sim 설계 필요**.
- 손목 잠그면 EMG ECU/FCU(손목근) 무관 → **유효 EMG = 중삼각근(거상)·이두(팔꿈치) 2채널뿐**.
- skeleton(150s)↔EMG(160s) **~6s 클럭오프셋** → 정밀동기 `*_summary.csv` 필요.
- 포맷: skeleton CSV 세미콜론구분, col10=BodyInfoJson(Joints.<name>.Position.XYZ 미터), col2-7=사전계산각(law-of-cosines). 다운로드: 비이미지 1.5GB만 (이미지 287GB 제외). 매니페스트 `scidb.cn/api/sdb-filetree-service/getAllUrl?dataSetId=2f68a8f8377b41aba79ce53104dc3ca6&version=V2`, 파일별 `download.scidb.cn/download?fileId=<id>`.

#### KIMHu 프로토콜 명세 (T1/T2) — 관측(데이터) vs 추론(선행지식)
M0 조사 워크플로(실데이터 직접 적재)로 확정. **데이터셋 자체 프로토콜 문서는 레포에 없음**(README/PDF/논문 부재) → 아래는 *데이터 파일에서 직접 계산·확인한 값*(관측)과 *외부근거 필요한 선행지식*(추론)을 구분 기재. 부하·동작서술을 발명하지 않기 위함.

| 항목 | 값 | 근거 |
|---|---|---|
| 피험자 | 20명, 전원 오른팔 | **관측**: anthropometric_information_participants.json(20), 과제폴더 40, Skeleton_Tracking_Count.csv(40행) |
| 반복 | 과제당 10회/인 | **관측**: EMG 마커 20=10쌍/파일, 상승분절 중앙 10 |
| 휴식 | 반복간 ~3.5–5s | **관측**(MAYR02 EMG 마커 간격). '6s'=선행지식·근사 |
| 속도 | 미지시(self-paced) | 선행지식. 관측 T분포와 정합(T1 중앙 2.08s / T2 2.53s) |
| 센서 | Kinect V2, 25관절, Position+쿼터니언, 유효 ~29.9fps | **관측**: BodyInfoJson, TimeStamp |
| EMG | Noraxon 1500Hz 4ch: ECU·FCU·Biceps·**중삼각근** | **관측**: *_emg_data.mat channelNames |
| **부하** | **없음(확정)** | **관측**: data/ 전체 load/dumbbell/object 키워드 0(본인 체중 컬럼만); EMG Activities={Begin,Pause}; 손 object 추적 없음 |
| **T1 = 관상면** | 외전(측면거상)→팔꿈치 머리로→신전→손목(굴/중/신)→rest. 평면 az **~7°**(거의 관상), 거상정점 중앙 **~95°**, elbow정점 ~122° | 각도=**관측**(latent_T1.json, plane_nominal 7.09°). 동작 문구=선행/운동학 추론(데이터셋 명문 아님) |
| **T2 = 횡단/전방** | 외전/거상→팔꿈치 굽혀 손 가슴앞(수평내전)→손목→rest. 평면 az 코호트 **~13.5°**(개별 전방국면 +45°까지, p95=45° max=63°), 거상정점 중앙 **~87°**, elbow정점 ~111° | 각도=**관측**(latent_T2.json). 동작 문구=선행/추론 |

- **사용 구간**: 둘 다 복합시퀀스 → **상승(외전) 구간 [shoulder_elv + elbow]만** 사용(정점 뒤 손목조작=채널 오염). T2는 결정(b)로 전방 가슴앞 국면까지 분절 확장 → 평면을 스칼라→*궤적*화.
- **라벨 주의**: T1/T2='관상'/'횡단'은 관측 평면각과 정합하나 데이터셋이 문자 그대로 명명한 건 아님(폴더명은 `<ID> T1`/`<ID> T2`). DOI도 파일로 존재하지 않는 선행지식.

### U-Limb (참고, `data/ulimb_mhh/`)
- Harvard Dataverse, MHH 센터(건강 20, Vicon 마커+EMG). 속도분포 참고용. ADL 제스처라 표준 아님(거상 41–88° 제멋대로). `MHH.rar` 2GB + 추출.

### 산출
- `results/kimhu_protocol_MAYR02_T1.png` (프로토콜↔데이터↔EMG 대응), `results/kimhu_T1_vs_T2_MAYR02.png` (평면 비교).

## 확정 (C 근육 섭동 — LOCKED 2026-06-10)
- **type = Fmax + L_opt 둘 다(C2).** `s_F~균등[0.05,1.0]`(0.25 아닌 0.05까지=중증 KIN신호), `s_L~균등[0.70,1.20]`. 평가는 **경증/중등도/중증 구간별 분리**.
- **10채널 × 2파라미터 = 20-dim.** 개별 표적 8(DELT1/2/3, SUPSP, PECM1, CORB, BIClong, TRIlong) + 묶음 2(A{INFSP,SUBSC,TMIN}=하부cuff, B{TMAJ,LAT1/2/3}=등·내전). 16=주 보고, 4=거친/강건성.
- **명목활성 9**(PECM2/3, TRIlat, TRImed, ANC, SUP, BICshort, BRA, BRD) — 정책은 쓰되 param 정상. **제거(잠금) 37**(손목6+손31). **행동공간 26**(PT·PQ 유지=pro_sup 추적용).
- **joint 구조 = 희소-국소 + 커리큘럼.** 샘플마다 손상근 k∈{1,2,3}(1~2에 무게), 나머지 ~정상(0.9~1.0 잔잔). k는 묶음팀 포함. 커리큘럼: 1단계 k=1 → 2단계 k≤3. (조밀독립=포화로 식별불가 → 기각.)
- 식별 회귀기 특징은 **sim-privileged 전체 근육 활성 + 운동학**(팔꿈치 계승; 2채널 EMG 제약은 참조/현실성 앵커에만).

## 확정 (F 학습 로드맵 — LOCKED 2026-06-10)
- **전이**: 팔꿈치→팔 가중치 전이 불가(차원 다름) → **fresh**. 레시피(곱셈형 reward·anti-tremor 스택·커리큘럼)만 계승. 의미있는 전이 = 팔 *내부* 건강→섭동 warm-start(커리큘럼). **단일 조건부 정책**(전략3a; (근육param20+latent4+과제) 조건).
- **crawl-walk-run 마일스톤**: **M0** 참조 파이프라인(RL 아님: 3D만·상승만·정규화후풀링, T1/T2 템플릿+4인자 분포, pro_sup 참조, thoracohumeral→shoulder_elv 1:1 calib). **M1(★임계)** 건강·단일·고정궤적(T1, 명목param, latent OFF) → **검증 게이트**(정점 91±11°내·단조상승 반전0·인간대역속도·4–12Hz떨림無·추종RMS). **M2** latent ON+T2(건강, 게이트를 분포 전반에서). **M3** 섭동 커리큘럼 k=1→k≤3(20-dim 조건; 약화→신호 발생 확인). **M4** param×latent 스윕 롤아웃→특징(전체ACT+운동학) 추출→G.
- **M1 게이트 실패시 escalation 사다리(신호 보존 우선)**: ①넓은-밴드 강제종료(myodm식, 약화 이탈은 생존) ②anti-tremor 스택(CAPS·cubic effort·leaky-integral+gravity obs·settle) ③RSI ④속도/위상 추적항 ⑤(최후) 추적가중↑(신호소멸 감수, ACT로 보완).
- 규모(26-dim 행동·시변참조·20-dim 조건 → 팔꿈치보다 무거움)는 **E의 입력**.

## 미해결 (OPEN)
- **A. latent 인코더/참조** — A1 LOCKED. **M0 구현 완료**(`references/`): kimhu_io(3D 유도·col4 가드)·segment(상승/전방 분절)·build_templates(정규화·로버스트풀·latent fit·plane 궤적)·warp(런타임 참조). 산출 `references/out/`.
  - **M0 검증(viz·유사도, 2026-06-11 조사 워크플로)**: ①plane 급변=**저거상 방위각 노이즈**(팔 수직→수평성분 작아 arctan2 정의불량; plane_mag<0.13서 중앙편차 ~38°) + 일부 반복 **Kinect 흔들림**(MAYR02 등 1/20) → `kimhu_io.reliable_plane`(마스킹·보간) + 표시 시 **대표(최단조) 반복** 선택. 템플릿은 코호트 평균이라 이미 단조(편차 0°)·무영향(npz 재생성 불필요). ②**shoulder_rot 참조 없음 재확인**: naive 쿼터니언=짐벌 불안정, *제대로* 푼 swing-twist조차 elbow_flexion과 **~0.84 교란**(전완 운반을 측정, 고립 축회전 아님) → Kinect로 고립 축회전 측정 불가 → 참조 생성=발명(기각). 물리가 외회전 공급(LOCKED 유지). ③**processed↔generated 시퀀스 유사도** = **DTW(주, 순수 numpy DP)+NRMSE·Pearson r(보조)**(`references/similarity.py`). 실측: elev DTW~0.01–0.04(r≥+0.9 대부분), plane DTW~0.02–0.07. 산출 `similarity_proc_vs_gen.png`·`shoulder_axes_pipeline.png`.
- **E. GCP 인프라** ← 다음. MuJoCo CPU·PPO 소형망, 다코어 spot VM, GCS 버킷. 규모=배터리(T1+T2)·행동공간(26근)이 정함.
- **G. 식별모델(회귀)** — **시뮬레이션 데이터 나온 뒤 진행**(M4 후). 구조(MLP-FUSED 계승), 특징=ACT 주채널+KIN 보조+known latent 공변량, 약화정도별 재가중, 시계열 vs 스냅샷.

## 환경
- conda `rl_myosuite`. myoarm 경로 `$SIMHIVE/myo_sim/arm/myoarm.xml`. unrar/aria2/p7zip conda 설치됨.
