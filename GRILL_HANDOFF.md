# GRILL 인계 문서 — myoArm 어깨거상 프로젝트

> **이 문서의 용도:** `myosuite_elbow_basic`에서 시작된 grill-me 설계 인터뷰를 이 폴더(`myosuite_arm_1`)에서 *이어서* 진행하기 위한 완전 인계 브리프. 새 세션은 이 문서 + `DESIGN.md` + 자동로드 메모리(`arm-datagen-design.md`)를 읽고 **아래 "열린 결정 큐"의 C부터** grill을 재개하면 된다.
> **재개법:** `~/projects/myosuite_arm_1/`에서 Claude 실행 → 팔 메모리 자동로드 → `/grill-me`로 C 질문부터. 진행방식: 한 번에 한 질문, 매 질문에 추천답 제시, 코드/데이터로 답할 수 있으면 탐색.

작성 2026-06-10. 부모 프로젝트: `myosuite_elbow_basic`(팔꿈치 6근육 파이프라인 완성·검증; 메모리 elbow-datagen-design).

---

## 0. 세 가지 변경 동기 (사용자 원안)
1. **실 kinematic 데이터 모방 필수** → 여러 궤적의 변동을 차원축소(latent)로 요약 → 정책 obs. 근육param 조정 + 궤적 조정 동시. (같은 근육상태도 다양한 속도로 동작 가능하므로 타당.)
2. **GCP 사용**(로컬 아님).
3. **tracking 추가 → myoArm 확장** + 그에 맞는 human data 확보.

---

## 1. 확정 결정 (LOCKED) — 결정 순서대로, 근거 포함

### (1) 최종 목표 = sim 내부 식별성 (NOT sim-to-real)
- 질문: *"현실적 궤적변동(nuisance)을 가로질러 근육파라미터를 sim에서 복원 가능한가."*
- 실 환자 테스트셋 없음. **파라미터 ground truth는 sim 전용** — per-muscle Fmax/Lopt 라벨 붙은 실데이터는 세상에 없음. (이 사실이 데이터셋의 역할을 "현실성 앵커"로 한정.)

### (2) 메커니즘 = latent-conditioned SOFT motion tracking ★핵심★
- 실데이터 거상궤적 → latent → 디코드 참조 → 정책이 **낮은 강도로** 추종.
- **원리: "추적 실패/이탈 = 식별 신호."** 강한(완벽) 추적은 시나리오 B(적응정책이 섭동을 궤적으로 흡수)를 극대화 → 모두가 참조와 동일 → 운동학 신호 소멸. soft 추적 + 노력비용 → **경증=완전보상(ACT 신호)·중증=이탈(KIN 신호)**로 자연히 갈림. 임상 "이 동작 해보세요→어떻게 실패/보상하나 관찰"과 동형.
- 이로써 앞서의 **"일관된 기준 vs 식별성" 긴장 해소**: 참조 = 표준화된 *시도*, 이탈 = 신호.

### (3) nuisance 처리 = 조건부(conditional)
- 식별모델(회귀, RL 아님)은 궤적인자/latent을 **알려진 공변량으로 입력**(sim에서 우리가 샘플 → known). well-posed.
- 시나리오 = **적응 마친 만성 개인**(CNS 재적응, 근육param 고정), self-paced 속도 변동.

### (4) 데이터셋 = KIMHu (검증·다운로드 완료)
- 표준화 *지시* 동작이라 일관기준 확보(자유제스처 아님). 판정 CAUTION(좁게 쓰면 GO): **표준동작 템플릿 + EMG 현실성 앵커**로만, ground truth 아님.

### (5) 모델 = myoArm (MoBL/Holzbaur), 스코프 = GH 근육 + 팔꿈치
- 어깨 독립 3-DoF: `elv_angle`(평면), `shoulder_elv`(거상각 0–180°), `shoulder_rot`(축회전) + `elbow_flexion` + `pro_sup`. nq=38(독립27+커플11), nu=63.
- **견갑 = shoulder_elv에 선형커플링 rhythm(자동), 견갑근육 부재(SERR/TRAP/RHOMB 없음)** → 견갑 병리·전거근/승모근 약화 **스코프 밖**.
- **손목·손가락 잠금** + 원위 ~40근육 actuator 제거(행동공간 축소).
- 과제 = 어깨 거상(팔 들어올리기, shrug 아님). **팔꿈치 항상 free**(거상에 공동동원; KIMHu T1 실측 elbow 5→124°).

### (6) Reward 구조 (2D 팔꿈치 곱셈형 계승)
```
reward = TASK_track · QUALITY + w_e·EFFORT + penalties
  TASK_track = Σᵢ wᵢ · exp(−k(qᵢ − q_refᵢ)²)   ← soft 추적
  QUALITY    = exp(−(jerk + dctrl + settle))
  EFFORT     = −act_mag/na   (노력비용: 보상 쥐어짜기 억제 → 곱 밖 가산)
```

### (7) DoF별 추적 가중 wᵢ (LOCKED) — "신호=저w(자유) / 압핀=고w"
| DoF | wᵢ | 역할 / 근거 |
|---|---|---|
| shoulder_elv | **0.15** | 핵심 신호(거상근 약화→못미침) |
| elv_angle | **0.15** | 평면도 자유=신호(drift). 조건은 참조 초기평면으로만 구분 |
| elbow_flexion | **0.25** | 이관절 신호(BIClong/TRIlong) |
| shoulder_rot | **0.9 압핀** | KIMHu서 관측불가(축회전은 위치점에 안 실림) → 명목고정 + 치팅차단 |
| pro_sup | **0.9 압핀** | 관측불가 → 명목고정 |
| 손목·손가락 | 잠금 | — |
- `k`(추적 날카로움)·`w_e`·압핀 명목값·추적강도 = **스윕 후보**(강도 vs 식별정확도 곡선).

### (8) latent/참조 구성 (LOCKED, A1) = 해석가능 인자
- 학습 인코더 불필요. KIMHu에서 **(T 이동시간, peak 최대거상각, skew 속도프로파일 비대칭, plane 평면azimuth)** 분포 fit → 샘플 → **KIMHu 평균 거상템플릿을 [T 시간워핑·peak 진폭스케일·plane azimuth설정]으로 워핑** = 참조.
- 샘플값 = 정책 obs + 회귀기 known 공변량. (변동 저차원이라 VAE 불필요; 형태잔차 크면 함수형PCA 보완.)

### (9) 로드맵 = 풍부한 다조건 배터리부터
- 식별성 = 과제 풍부도에서(팔꿈치 "DoF↑→식별↑" 교훈의 어깨판). 구체 구성은 OPEN(B).

### 식별 표적 근육 (식별성 워크플로 결론)
- **현실적 식별가능**: DELT1/2/3 + SUPSP + 이관절 BIClong/TRIlong (+부분 PECM1/CORB).
- **혼동(병합/명목)**: INFSP/SUBSC/TMIN/TMAJ(하부 force-couple), TMAJ↔LAT1-3. 회전근개 회전근은 rotation 압핀으로 식별 포기.

---

## 2. 데이터 (검증·확보)

### KIMHu (`data/kimhu/`, CC0/CC-BY) — 1차
- ScienceDB DOI 10.57760/sciencedb.01902. 20명(오른팔), 2테스트×10반복, 6s휴식, **속도 미지시(self-paced)**.
- Kinect V2 **30fps 25관절(Position만, orientation 없음)** + EMG 1500Hz 4채널(ECU,FCU,Biceps,**Deltoideus Medius**). EMG에 20마커(반복 start/end 쌍).
- **T1=관상**: 외전→팔꿈치 머리로→신전→손목(굴/중/신)→rest. **T2=횡단**: 외전→팔꿈치굽혀 손목 가슴앞→손목→rest. 둘 다 복합 → **외전구간 [shoulder_elv+elbow]만 사용**.
- 실측(MAYR02): T1 shoulder_elv 6→107°·평면az~3°; T2 6→103°·az~45°. elbow T1 스파이크~115°/T2 지속~120°. **T 중앙 2.4s(1.6–4.6), skew~0.35**. → **2 실재평면(0°,45°), 순수시상면(90°)은 sim 설계 필요**.
- 손목 잠그면 EMG ECU/FCU(손목근) 무관 → **유효 EMG = 중삼각근(거상)·이두(팔꿈치) 2채널뿐**.
- skeleton(150s)↔EMG(160s) **~6s 클럭오프셋** → 정밀동기 `*_summary.csv` 필요.
- 포맷: skeleton CSV 세미콜론구분, **col10=BodyInfoJson**(Joints.<name>.Position.XYZ 미터), col2-7=사전계산각(law-of-cosines). 40 trial(V2/<SUBJECT TRIAL>/), 1.6GB.
- 다운로드(무인증): 매니페스트 `scidb.cn/api/sdb-filetree-service/getAllUrl?dataSetId=2f68a8f8377b41aba79ce53104dc3ca6&version=V2`, 파일별 `download.scidb.cn/download?fileId=<id>`. 비이미지 1.5GB만(이미지 287GB 제외).

### U-Limb (`data/ulimb_mhh/`) — 참고용
- Harvard Dataverse MHH(건강20, Vicon+EMG). 속도분포 참고. ADL제스처라 표준 아님. MHH.rar 2GB + 추출.

### 산출 플롯 (`results/`)
- `kimhu_protocol_MAYR02_T1.png` (프로토콜↔kinematics↔EMG 대응: 중삼각근이 외전에 점화 확인).
- `kimhu_T1_vs_T2_MAYR02.png` (평면 비교: T1 az~3° vs T2 az~45°, 같은 거상높이).

---

## 3. 확립된 핵심 통찰/원리 (재논의 불필요)
1. **sim-only ground truth**: 근육param 정답은 sim에서만 → 식별성 검증은 본질적으로 sim 내부.
2. **이탈=신호 / 강한추적=신호소멸**: soft 추적이 핵심. 추적강도가 신호-현실성 손잡이.
3. **어깨 degeneracy > 팔꿈치**: 거상근 ~10-14개. 단일평면으론 개별식별 불가 → 과제 풍부도(평면·회전·부하)가 핵심 레버. 회귀(분류 아님)로 프레이밍.
4. **관측 한계**: KIMHu는 관절 위치점만(orientation 없음) → shoulder_rot·pro_sup 원리적 관측불가 → 명목고정·압핀. shoulder_elv·elv_angle·elbow는 깨끗이 복원.
5. **견갑은 rhythm으로 처방**(근육 없음) → GH 근육만 스코프.
6. **EMG 현실성은 2채널(중삼각근·이두)로 한정**(ECU/FCU는 손목근=스코프밖).
7. **FUSED(ACT+KIN) 필수**(팔꿈치 교훈): 활성=근육별 노력, 운동학=잔차. 단독 모달리티로 불충분.

---

## 4. 열린 결정 큐 (OPEN) — 여기서 grill 재개. 순서·추천 포함.

### ▶ C. 근육 섭동 type·범위 ← **다음 질문(사용자 보류 중)**
팔꿈치 스킴 myoArm에 그대로 적용(`gainprm·biasprm[:,2]*=s_F`, `[:,0:2]/=s_L`). 어깨 특이점: ①임상약화 심함(파열·마비≈힘0) → s_F 하한 0근처 확장 + "중증→추적실패=신호" 설계가 그 영역 필요. ②근육多·degenerate → 표적 ~8근육만 섭동, 혼동군 명목/병합.
- **C1 (추천)**: Fmax-only, `s_F∈[~0.05,1.0]`, 표적 ~8근(DELT1/2/3,SUPSP,BIClong,TRIlong,+PECM1/CORB). L_opt 보류. 8차원.
- C2: Fmax+L_opt 둘 다(팔꿈치 계승), 표적×2=16차원. 완전하나 난이도↑.
- C3: Fmax, 전 GH근(~14) 혼동군 포함. 커버리지↑ 식별 흐림.

### ▶ B. 배터리 구성 (평면·회전·부하)
사용자 "풍부한 배터리"(option2) 택함, 구체 미확정. 평면이 신호DoF(자유)가 됐으므로 "평면 조건"=참조 초기 azimuth. 실재=0°(T1)/45°(T2), 90°(시상)는 sim설계. 회전(IR/ER)은 압핀이라 sim 명목 setpoint, 부하=sim 가중. 결정: {평면 setpoint × 회전 setpoint × 부하} 어디까지. (추천 방향: 평면 0/45/90 + 부하 2수준부터, 회전은 추가축으로 후순위 — 단 사용자 재논의 원함.)

### ▶ D. 참조 retargeting 디테일
KIMHu각→myoArm좌표 매핑(thoracohumeral elev → shoulder_elv), mean±SD 템플릿 구축(SD=추적 허용밴드 ~8-12°), ~6s skeleton-EMG sync 보정(summary.csv).

### ▶ E. GCP 인프라 (변경#2, 미착수)
MuJoCo=CPU, SB3 PPO 소형MLP(GPU 무익). 다코어 CPU VM(spot) 권장 방향. 배터리 규모·근육수(행동공간↑)가 비용 결정 → B 이후. GCS 버킷(KIMHu+체크포인트). 관리형 vs raw VM, 예산.

### ▶ F. 학습 로드맵
팔꿈치 정책 전이 vs fresh. PPO 대형 행동공간(어깨~13-22근육 vs 팔꿈치6) 학습규모·수렴. 베이스라인(단일평면)→배터리 확장 점진 vs 일괄.

### ▶ G. 식별모델(회귀, RL 아님)
구조(MLP-FUSED 계승), 특징(운동학 이탈 + 활성 + known 공변량 latent), **약화정도별 데이터 재가중**(중증=큰 이탈, 이질적 난이도). 시계열 vs 스냅샷(팔꿈치: 스냅샷FUSED가 실용천장).

---

## 5. 환경·도구
- conda `rl_myosuite` (MyoSuite 2.11.6, Gymnasium 0.29.1, MuJoCo 3.3.0, SB3 2.7.1, torch CPU).
- myoarm: `$SIMHIVE/myo_sim/arm/myoarm.xml`. 근육 actuator 63, 어깨근 DELT1-3/SUPSP/INFSP/SUBSC/TMIN/TMAJ/PECM1-3/LAT1-3/CORB.
- unrar/aria2/p7zip conda 설치됨. torch CPU 소형망: OMP_NUM_THREADS=1 권장(오버서브스크립션 ~4배 느림).
- 팔꿈치 코드 참조: `myosuite_elbow_basic/`(elbow_perturb_v0.py, train.py, regress_nn.py 등) — reward·섭동·회귀 패턴 재사용.
