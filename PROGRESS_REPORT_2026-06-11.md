# myoArm 근골격 파라미터 추정 — 진행 보고서

> 작성: 2026-06-11 (자율 작업 세션). 정본 설계=`DESIGN.md`, 작업원칙=`CLAUDE.md`, 진행상태=`SESSION_LOG.md`.
> 이 문서는 **무엇을·왜·어떻게** 했는지 수식·결정·해설 포함 종합 보고.

---

## 0. 한 문장 요약

> **목표 = "시뮬레이션 안에서, 팔의 근육을 일부러 약하게 만든 뒤, 그 움직임(운동학)과 근육 활성만 보고 *어느 근육이 얼마나 약해졌는지* 되맞히는 회귀기(regressor, 입력→연속값 예측 모델)를 만드는 것."** 실제 환자 데이터로 검증하는 게 아니라(*sim-to-real* 아님), **시뮬레이션 내부 식별성**(sim-internal identifiability = 가상 실험에서 정답 라벨이 있으니 복원 가능 여부를 따짐)을 측정한다.

진행: **M0~M4 + G 전 단계 ✅ 완료** — 파이프라인 닫힘. 핵심 결과: **근육별 Fmax scale을 ACT(privileged) 주채널로 평균 R² 0.90 복원**, KIN은 비보상 약화(주외전근·이관절·회전근개)에서 보조. (M1 비디오 `results/report/M1_T1.mp4`.)

---

## 1. 용어 해설 (어려운 말 먼저)

| 용어 | 영어 | 쉬운 설명 |
|---|---|---|
| 근골격 모델 | musculoskeletal model | 뼈·관절·근육을 물리적으로 흉내 낸 가상 팔. 여기선 MuJoCo의 **myoArm**(63개 근육). |
| 거상 | elevation | 팔을 들어올리는 동작(shoulder_elv 관절각, 0°=차렷, 90°=수평). |
| 식별성 | identifiability | "관측만으로 숨은 파라미터를 유일하게 알아낼 수 있는가". 약한 근육이 신호를 남기면 식별 가능. |
| 강화학습 | RL, reinforcement learning | 보상(reward)을 최대화하도록 시행착오로 **정책(policy)** = "상태→행동" 함수를 학습. |
| 정책 | policy | 현재 상태(obs)를 받아 26개 근육에 줄 활성(activation)을 출력하는 신경망. |
| 섭동 | perturbation | 근육 파라미터(Fmax 등)를 인위적으로 바꿔 "약화"를 만드는 것. |
| soft 추적 | soft motion tracking | 참조 궤적을 **느슨하게** 따라가게 함. 빡빡하게 따르면 약화가 궤적에 안 드러나서 일부러 느슨하게. |
| 운동학/활성 | KIN / ACT | KIN=관절 각도 등 *움직임*. ACT=근육 *활성도*(0~1). 식별의 두 신호 채널. |
| 워밍스타트 | warm-start | 앞 단계에서 학습한 정책 가중치를 **이어받아** 다음 단계 학습 시작(처음부터 안 함). |
| 게이트 | validation gate | 다음 단계로 넘어가기 전 통과해야 하는 **검증 관문**(5개 기준). |

---

## 2. 전체 파이프라인 (crawl-walk-run = 기어→걷기→뛰기 점진법)

```
M0  참조 만들기   : 실제 사람 데이터(KIMHu) → 표준 거상 궤적 템플릿 + 변동 요약(latent)
M1  건강 베이스라인: 정상 근육 정책이 "사람다운" 거상을 재현하는가? (★임계경로)
M2  다양성 추가   : 여러 속도·정점(latent 분포) + 두 번째 과제 T2(전방 거상)
M3  약화 학습     : 근육을 약화시키고, 정책이 보상(compensate)하도록 학습 (k=1→k≤3)
M4  데이터 생성   : (약화 정도 × 변동) 조합을 대량 롤아웃 → 특징·라벨 데이터셋
G   회귀기        : 특징 → 약화 정도(s_F,s_L) 복원, 채널별 식별성 R²/MAE 보고
```

**핵심 원리(DESIGN)**: *"추적 실패/이탈 = 식별 신호."* 약화된 근육을
- **경증**이면 정책이 다른 근육으로 보상 → 궤적은 정상이지만 **근육 활성 패턴(ACT)이 변함** = ACT 신호.
- **중증**이면 보상 불가 → 팔이 목표에 **못 미침(KIN 이탈)** = KIN 신호.
- 임상의가 "이 동작 해보세요 → 어떻게 실패/보상하나 관찰"하는 것과 동형(同型).

---

## 3. 환경 설계 (custom_envs/arm_perturb_v0.py)

### 3.1 모델·행동공간 (LOCKED 결정)
- **myoArm**: 독립 관절 27 + 커플링 11 = nq 38, 근육 63개. 어깨 3축(`elv_angle`=평면, `shoulder_elv`=거상, `shoulder_rot`=축회전) + `elbow_flexion` + `pro_sup`(전완 회내외).
- **행동공간 = 26개 근육** (정책이 직접 구동):
  - 어깨 15: DELT1/2/3, SUPSP, INFSP, SUBSC, TMIN, TMAJ, PECM1/2/3, LAT1/2/3, CORB
  - 팔꿈치 9: TRIlong/lat/med, ANC, SUP, BIClong, BICshort, BRA, BRD
  - 회내 2: PT, PQ (pro_sup용)
- **제거 37개**(손목6 + 손가락31): 정책이 안 건드림(ctrl=0). **손목·손가락 22관절은 잠금**.
- **잠금 방법(기술 결정)**: MuJoCo **MjSpec**(모델을 코드로 편집 후 재컴파일) API로 22개 관절에 **equality 제약**(`mjEQ_JOINT`, 상수 0)을 추가. 검증: 중력 200스텝 후 손목/손가락 0.0°±0.1° 고정, 어깨/팔꿈치는 자유.
- **견갑 리듬**(scapulohumeral rhythm): 11개 커플 관절이 shoulder_elv에 선형 연동(모델 내장 equality). 견갑 구동 근육은 모델에 없음 → 견갑 병리는 스코프 밖.
- **shoulder_rot = 자유(참조 없음)**: Kinect 데이터로 *고립된* 축회전을 신뢰성 있게 못 잼(전완 운반과 ~0.84 교란). 그래서 참조를 안 만들고 정책이 자유롭게 씀. 거상엔 의무적 외회전(external rotation)이 필요해 **물리가 외회전을 공급**. 부수효과: 회전근개(cuff) 약화가 *회전 이탈*로 드러나 식별에 유리. (실측 확인: M1 정책이 거상 시 shoulder_rot을 −60°까지 부드럽게 외회전.)

### 3.2 보상 함수 (reward) — 정밀 수식

매 스텝 보상:
$$
r_t \;=\; \underbrace{\text{TASK}_t \cdot \text{QUALITY}_t}_{\text{추적×품질(곱셈형)}} \;+\; \underbrace{w_e\,\text{EFFORT}_t}_{\text{노력비용}} \;+\; \underbrace{\text{pen}^{\text{rot}}_t}_{\text{회전 정칙화}}
$$

**(1) TASK — soft 추적, 양면 허용밴드(two-sided tolerance band):**
$$
\text{TASK}_t \;=\; \sum_{i\in\{\text{elv, plane, elbow, prosup}\}} w_i \cdot \exp\!\Big(-\,k_i^{\text{out}}\,\big[\max(0,\;|q_i - q_i^{\text{ref}}| - b_i)\big]^2\Big)
$$
- $q_i$=현재 관절각, $q_i^{\text{ref}}$=참조각, $b_i$=허용밴드 반폭(half-width).
- **밴드 안**($|q_i-q_i^{\text{ref}}|\le b_i$) → 지수의 안이 0 → 기여 = $w_i\cdot 1$ (**무벌점**). 이게 "self-paced 변동(사람마다 속도·자세 다름)을 허용"하는 핵심.
- **밴드 밖** → $k_i^{\text{out}}$로 가파르게 감쇠.
- 가중·밴드(LOCKED): $w_{\text{elv}}=0.15,\ w_{\text{plane}}=0.15,\ w_{\text{elbow}}=0.25,\ w_{\text{prosup}}=0.20$ / $b$ = 10°, 15°, 12°, 20°. ($k^{\text{out}}$ = 30, 18, 22, 12)
- *왜 거상 가중이 0.15로 낮나?* → 빡빡하게 추적하면 약화가 궤적에 안 드러남. 느슨해야 중증 약화가 *이탈*로 보임. 정밀 복원은 ACT(주채널)가 담당.

**(2) QUALITY — 움직임 품질(곱셈 인자, 0~1):**
$$
\text{QUALITY}_t \;=\; \exp\!\Big(-\big(c_J\,\text{jerk} + c_D\,\text{dctrl} + c_S\,\text{settle}\big)\Big)
$$
- $\text{jerk}=\tfrac1{|DoF|}\sum \ddot q^2$ (관절 가속² 평균; 부드러움). $c_J=5\times10^{-6}$.
- $\text{dctrl}=\tfrac1{26}\sum (a_t-a_{t-1})^2$ (행동 변화량; **떨림 억제 핵심**). $c_D=0.5$.
- $\text{settle}=\tfrac1{|DoF|}\sum \dot q^2$ (정점 유지 구간에서만; 멈춤 유도). $c_S=1.0$.

**(3) EFFORT — 노력비용(보상 쥐어짜기·동시수축 억제, 곱 밖 가산):**
$$
\text{EFFORT}_t \;=\; -\,\frac1{26}\sum_{j} \text{act}_j^{\,p},\qquad w_e=0.05,\ p=2
$$

**(4) 회전 정칙화**(자유 DoF의 극단/치팅 차단): $\text{pen}^{\text{rot}}=-w_r\,[\max(0,|q_{\text{rot}}|-50°)]^2$.

**종료(termination)**: $|q_{\text{elv}}-q^{\text{ref}}_{\text{elv}}|>50°$ 또는 $|q_{\text{elbow}}-\cdot|>60°$ 또는 속도 폭주 → 종료 + −1. **밴드를 넓게(50°)** 둔 이유: 약화로 인한 이탈은 *살아남아야*(신호 보존) 하고, 정상 발산만 차단(myodm식 wide-band early termination).

### 3.3 관측(observation) — 76차원
$$
\text{obs} = [\,\underbrace{q_5}_{\text{관절각}},\ \underbrace{\dot q_5}_{\text{속도}},\ \underbrace{\text{act}_{26}}_{\text{근육활성}},\ \underbrace{q^{\text{ref}}_4,\ \text{err}_4,\ q^{\text{ref+ahead}}_4}_{\text{참조·오차·선행}},\ \underbrace{[\text{phase},\text{hold}]}_{2},\ \underbrace{\text{latent}_5}_{\text{변동인자}},\ \underbrace{[s_F,s_L]_{20}}_{\text{섭동조건}},\ \underbrace{\text{task}}_{1}\,]
$$
- **섭동 20차원을 항상 포함**(건강이면 전부 1.0): 그래야 obs 크기가 M1~M3 내내 76으로 **불변** → 워밍스타트 호환(가중치 그대로 이어받음).
- latent 5 = (T 이동시간, peak 정점각, skew 속도비대칭, plane 평면방위, elbow_peak). 정책 입력 겸 회귀기 **known 공변량**(known covariate = 우리가 sim에서 직접 정한 값이라 회귀 입력으로 줄 수 있음).

### 3.4 제어 주파수·에피소드
- 물리 500Hz(timestep 0.002s), **제어 50Hz**(frame_skip 10, dt=0.02s), 에피소드 3.5s = 175스텝. raise-and-hold(상승 후 정점 유지; 중력보상 부하가 식별에 유리).

### 3.5 섭동(M3) — 정밀 수식
근육 $i$에 force scale $s_F$, length scale $s_L$ 적용 (런타임 모델 필드 편집, 재컴파일 불필요):
$$
\text{gainprm}[i,2]\leftarrow s_F\cdot\text{gainprm}_0[i,2],\qquad \text{gainprm}[i,0\!:\!2]\leftarrow \text{gainprm}_0[i,0\!:\!2]\,/\,s_L
$$
(biasprm도 동일). 여기서 **gainprm[2]=Fmax**(최대 등척력, peak isometric force), **gainprm[0:2]=작동 길이범위**(Lopt 관련). 검증: $s_F{=}0.3$ → 등척력 정확히 0.30배.
- **표본 분포**: $s_F\sim U[0.05,\,1.0]$ (0.05까지=중증, 강한 KIN 신호), $s_L\sim U[0.70,\,1.20]$.
- **10채널 × 2파라미터 = 20차원**: 개별 8(DELT1/2/3, SUPSP, PECM1, CORB, BIClong, TRIlong) + 묶음 2(A={INFSP,SUBSC,TMIN}=하부 회전근개, B={TMAJ,LAT1/2/3}=등·내전). 묶음은 한 채널이 여러 근육을 공유 약화.
- **희소-k 커리큘럼**: 한 에피소드에 손상 채널 수 $k$만 약화, 나머지 정상. M3a: $k{=}1$ → M3b: $k\in\{1,2,3\}$ 가중 (0.5,0.3,0.2). (조밀 약화는 포화→식별 불가라 기각.)

---

## 4. 검증 게이트 (diagnostics/gate.py) — 5기준

| 기준 | 정의(수식/임계) | 의미 |
|---|---|---|
| **G1 정점** | M1: 정점∈[80,102]° (인간 91±11°) / M2: $\lvert\text{달성}-\text{참조}\rvert<12°$ | 사람만큼 든다 |
| **G2 단조** | 상승 중 *데드밴드(deadband) 1.5°* 이상 후퇴 횟수 ≤ 1 | 부드럽게 올라간다(되돌아가지 않음) |
| **G3 속도** | 상승시간(10%→90%) ∈ 인간대역 | 사람 속도 |
| **G4 떨림** | hold 구간 4–12Hz 성분 진폭 < 0.5° **또는** 대역파워비 < 0.15 | 손떨림 없음 |
| **G5 추종** | $\text{RMS}(q_{\text{elv}}-q^{\text{ref}}_{\text{elv}}) < 12°$ | 궤적을 따라간다 |

- **M1(고정 latent)**: 절대 정점밴드. **M2(latent 분포)**: 참조 정점이 77~107°로 변하니 절대밴드 부적합 → **상대모드**(참조와의 매칭).

---

## 5. 마일스톤별 결과

### M0 ✅ (이전 세션, 참조 파이프라인)
- KIMHu 실데이터(20명×2과제) → 3D 위치에서 거상각 유도(사전계산각 col2-7 금지) → 상승구간만 분절 → 정규화 후 로버스트 평균 템플릿 + latent(T·peak·skew·plane) 분포. `references/out/`에 커밋됨.

### M1 ✅ 건강 베이스라인 (★임계경로)
- **학습**: PPO(MlpPolicy [256,256]) + VecNormalize(관측·보상 정규화), 20 병렬 env, T1 고정 latent·명목 파라미터, 3M스텝.
- **결과(최종 게이트)**: 정점 **97°**·반전 **0**·속도 **1.62s**·떨림 진폭 **0.12°**·추종 RMS **2.84°** → **5기준 통과**. 모델 `models/ppo_arm_T1.zip`.
- **플롯 검증**: 매끈한 거상+hold, shoulder_rot −60° 외회전(물리 공급, 의도대로).

### M2 ✅ latent 분포 + T2
- **M2a**(T1 분포, M1 워밍스타트, 2.5M): 분포게이트 통과. **참조정점 ↔ 달성정점 상관 0.91** = 정책이 latent을 입력으로 *실제로 활용*해 목표 높이를 조절. `models/ppo_arm_M2a.zip`.
- **M2b**(T1+T2 혼합, M2a 워밍스타트, 4M): **T1·T2 둘 다 통과**(T1 정점오차 3.0°·RMS 5.2° / T2 정점오차 5.2°·RMS 6.8°). T2(전방 평면 스윕 0→+45°)를 1M스텝부터 습득. `models/ppo_arm_M2b.zip` = M3 시작점.
- 평면 부호 검증: myoArm `elv_angle`이 KIMHu `plane_az`(0=관상, +=전방)와 동일 규약 → T2 전방 스윕이 해부학적으로 정방향.

### M3 🔄 섭동 커리큘럼 (진행 중, 메커니즘 검증 완료)
- **M3a**(k=1, M2b 워밍스타트, 4M 완료): healthy 추적 유지(정점오차 3°). **신호 진단 결과(중증 s_F=0.1 vs 경증 s_F=0.5)**:

  | 채널(약화 근육) | 중증 KIN 미달 | 경증 KIN 미달 | 신호 종류 |
  |---|---|---|---|
  | DELT2 (중간 삼각근=주 외전근) | 22° | 10° | **KIN**(보상불가, 정도별 graded) |
  | TRIlong (삼두 장두=이관절) | 16° | 9° | KIN |
  | SUPSP (극상근) | ~0°(보상) | ~0°(보상) | **ACT**(보상 활성↑) |
  | A_lowcuff (하부 회전근개) | 회전 66°→14°로 제어 | — | ACT+회전 |
  | 대부분 | 작음 | 작음 | ACT 보상 |

  → **DESIGN 의도 정확히 실현**: 경증=ACT 보상(궤적 회복), 핵심근 손실=KIN 이탈, **약화 정도에 비례한 graded 반응**. 채널마다 보상 패턴(어느 근육이 대신 켜지나)이 달라 회귀 식별 가능.
- **M3b**(k≤3, M3a 워밍스타트, 4M): **현재 학습 중** — 다중근 동시 약화 추가.

---

## 5.5 그림으로 보는 학습 증거 (results/report/)

> 정책이 *제대로* 학습됐음을 궤적으로 확인. 모든 그림은 학습된 정책을 **deterministic**(탐색 노이즈 끈 채)으로 롤아웃해 얻음. 점선=참조(reference), 실선=정책(policy).

### 그림 1 — 단일 정책이 다양한 latent·두 과제를 추적 (`fig1_tracking_gallery.png`)
![tracking gallery](results/report/fig1_tracking_gallery.png)
- **상단 T1(관상면)·하단 T2(전방)**. 각 색=다른 latent 표본(속도·정점 다름).
- 거상(좌)이 각자의 참조(점선)를 따라 **서로 다른 정점·속도로 상승 후 유지**. 팔꿈치(중)도 추종. 평면(우): **T1은 ~0°(관상) 유지, T2는 0→+45° 전방 스윕** — 한 정책이 두 과제·변동을 모두 처리.

### 그림 2 — 정책이 latent을 실제로 활용 (`fig2_latent_usage.png`)
![latent usage](results/report/fig2_latent_usage.png)
- 가로=참조 정점, 세로=달성 정점. **y=x 대각선을 따라 분포**(T1 상관 **0.92**, T2 **0.81**) = 정책이 latent(목표 정점)을 입력으로 받아 *그에 맞춰* 높이를 조절. (latent을 무시했다면 수평선이 됐을 것.)

### 그림 3 — M3 식별 신호: 약화→KIN이탈/ACT보상 (`fig3_perturbation_signal.png`)
![perturbation signal](results/report/fig3_perturbation_signal.png)
- **상단(거상 KIN)**: 초록=건강(s_F=1), 주황=경증(0.5), 빨강=중증(0.1).
  - **DELT2·TRIlong**: 중증(빨강)이 명확히 **목표에 못 미침**(undershoot) = KIN 신호, 경증<중증의 graded.
  - **SUPSP**: 세 곡선이 거의 겹침 = **다른 근육으로 보상해 궤적 회복**(KIN엔 안 보임).
- **하단(활성 ACT)**: 약화근(초록→빨강으로 감소) + **보상근(파랑)이 중증에서 증가**. 궤적이 회복돼도 **활성 패턴이 변해 식별 가능**(ACT 주채널).

### 그림 4 — 학습곡선: 게이트 지표 수렴 (`fig4_learning_curves.png`)
![learning curves](results/report/fig4_learning_curves.png)
- 가로=학습 스텝(백만). 빨간 점선=통과 임계. **정점오차·추종 RMS가 임계(12°) 아래로 내려가 유지** = 수렴.
- M1(파랑)의 초반 봉우리(0.5–1M, 정점오차 26→37°)=**오버슈트 transient** 후 1.5M에 **자가수정**(본문 6절). M2~M3는 워밍스타트로 처음부터 낮음.
- (떨림 비율은 noisy하나 *절대진폭*은 전부 <0.5°=생리적 수준; 게이트는 healthy 추적 평가, M3 약화신호는 그림3·신호스캔으로 별도 판정.)

### 그림 5 — ★최종 결과: 채널별 식별성 (`fig5_identifiability.png`)
![identifiability](results/report/fig5_identifiability.png)
- 세로=s_F(Fmax scale) 복원 R²(1=완벽). 채널별 3막대: **ACT만(파랑)·KIN만(주황)·ACT+KIN(초록)**.
- **ACT만으로 전채널 R² 0.80–0.95**(평균 0.90) = privileged 근육활성이 주채널. **KIN만**은 이질적(평균 0.36): A_lowcuff(회전근개) 0.84·DELT2 0.52·TRIlong 0.44는 *이탈*로 드러나나, 보상가능근(DELT3/SUPSP/PECM1/CORB/BIClong)은 0.07–0.26으로 KIN엔 거의 안 보임.
- **결론 = DESIGN 명제 실증**: "ACT 주채널·KIN 보너스, 두 채널 중 하나로 복원." 보상가능근은 ACT가, 비보상근(주외전근·이관절·회전근개)은 KIN도 함께 복원.

---

## 5.6 최종 결과: 식별성 정량 (M4+G 완료 ✅)

20,000 롤아웃(M3b 정책, k∈{0,1,2,3} 섭동×latent 분포) → 특징 143차원 → FUSED-MLP 회귀 → **20개 파라미터(s_F·s_L ×10채널) 복원**. 15% 홀드아웃 검증.

| 신호 | s_F 평균 R² | 의미 |
|---|---|---|
| **ACT만** (privileged 전체 63근 활성) | **0.90** | 주채널. 적응한 운동지령이 약화를 부호화 |
| **KIN만** (운동학 11특징) | 0.36 | 보너스. 비보상 약화만 드러냄 |
| **ACT+KIN (전체)** | **0.92** | 최종 |
| 공변량만 (latent+task) | **≈0** | 누수 없음(정직성 검증) ✓ |

- **전 채널 s_F 강식별**(R² 0.80–0.97), s_L(Lopt) R² 0.75–0.86.
- **묶음(A_lowcuff, B_latadd)이 강식별인 이유**: 개별 회전근개 근육은 서로 degenerate라 *묶어서* 한 채널(공유 s_F)로 추정 → "묶음 전체 약화"는 잘 식별(DESIGN의 병합 전략대로). 개별 근육 분해는 시도 안 함(천장 정직 인정).
- 모델 `models/regressor_G.pt`.

---

## 6. 발견·교훈 (중요 — 디버깅 기록)

### (a) "떨림 실패"는 측정 버그였다 (phantom tremor)
- 증상: 게이트가 4–12Hz 떨림 진폭 6.4°로 계속 FAIL. anti-tremor 대책(leaky-integral 행동필터, cubic effort, CAPS 정칙화, 관절댐핑×10)이 **전부 무효**.
- 진짜 원인: `DummyVecEnv`는 에피소드 종료 시 **자동 리셋** → 롤아웃 루프가 step *후* 각도를 기록하면 마지막에 **리셋된 rest(≈0°) 샘플**이 섞임. 92°→0° 급강하 1샘플이 FFT(주파수분석)에서 광대역 가짜 "떨림"으로 읽힘.
- 수정: **step 전(행동 결정 시점) 기록**으로 변경. 수정 후 동일 모델이 떨림 0.05·진폭 0.12° = PASS. **실제 정책은 처음부터 매끈했다.**
- 교훈: 자동리셋 VecEnv에서 시계열을 모을 때 post-done 샘플 배제 필수. *차분/스펙트럼* 지표는 경계 아티팩트에 극도로 민감.

### (b) 미세 jitter 반전 오집계 → 데드밴드 도입
- 추적이 너무 정밀(RMS 2.4°)해지자 상승 중 0.1° 수준 jitter가 "반전 2"로 집계돼 단조성 FAIL. 실측: 최대 단일하강 0.11°(전체 95° 상승 중) = 거시적 완전 단조.
- 수정: 반전을 **running-max 대비 1.5° 이상 하강**만 세도록(deadband). 미세 jitter 면역.

### (c) 오버슈트 transient는 자가수정
- 학습 초반(500k–1M) 정점 120–133° 오버슈트 후 1.5M에 자가수정. 원인: 거상 가중(0.15)이 낮아 초반엔 다른 DoF만 맞추며 racing. 워밍스타트(M2~)로 이 transient 회피.

### (d) anti-tremor 스택은 결국 불필요
- plain PPO 베이스라인 보상으로 충분(떨림은 유령이었음). 노브(act_lowpass/effort_pow/CAPS/댐핑/w_vel)는 코드에 남겨두되 *진짜* 떨림 발생 시만 사용.

---

## 7. 핵심 결정 요약 (LOCKED, 누락 없이)

1. **목표** = sim 내부 식별성(회귀), sim-to-real 아님. 파라미터 정답은 sim 전용.
2. **메커니즘** = latent-conditioned soft motion tracking. "이탈=신호", 강한추적=신호소멸.
3. **모델** = myoArm, 스코프 = GH(어깨관절) 근육 + 팔꿈치. 견갑·손목·손가락 제외.
4. **행동 26근**, 원위 22관절 equality 잠금, 견갑 리듬 자동.
5. **shoulder_rot 자유**(참조없음, 물리가 외회전 공급), **pro_sup 저가중 추적**(데이터 교차검증됨).
6. **보상** = TASK(양면밴드 soft)·QUALITY + EFFORT + 회전정칙. wide-band 종료.
7. **섭동** = Fmax+Lopt, $s_F\in[0.05,1]$·$s_L\in[0.70,1.20]$, 10채널(개별8+묶음2), 희소-k 커리큘럼.
8. **단일 조건부 정책** + crawl-walk 워밍스타트(obs 76 불변). 배터리 = T1+T2 고정(평면·부하 발명 금지).
9. **게이트** = 5기준, M1 절대/M2 상대모드.
10. **학습 = 이 머신(32코어)에서 직접**(Docker/GCS 미사용 — 충분). 체크포인트는 `models/checkpoints/`.

---

## 8. 산출물·재현 커맨드

**모델**: `ppo_arm_T1`(M1) · `ppo_arm_M2a`·`ppo_arm_M2b`(M2) · `ppo_arm_M3a`·`ppo_arm_M3b`(M3) · `regressor_G.pt`(G). 데이터 `data/sim/train.npz`(20k). 그림 `results/report/fig1~5`, 비디오 `M1_T1.mp4`.

```bash
# (1) 참조 템플릿 [M0]              python -m references.build_templates
# (2) M1 학습+게이트                python train.py --task T1 --timesteps 3000000 --out ppo_arm_T1
# (3) M2a/M2b (워밍스타트)          --latent-mode sample [--task mix] --resume <prev>.zip --vecnorm <prev>_vec.pkl
# (4) M3a/M3b (섭동)               --perturb --curriculum-k 1[,2,3] --resume ppo_arm_M2b.zip ...
# (5) 게이트 검증                   python -m diagnostics.gate --model M --vecnorm V [--sample-latent]
# (6) 신호 진단                     python -m diagnostics.signal --model M --vecnorm V --sf 0.1
# (7) M4 데이터(병렬샤드)            python gen_data.py --model ppo_arm_M3b... --n 1000 --seed i --out shard_i.npz
#     병합                          python tools/concat_shards.py 'data/sim/shard_*.npz' data/sim/train.npz
# (8) G 회귀 + ablation             python regress_nn.py --data data/sim/train.npz [--features act|kin|actkin]
# (9) 그림/비디오                   python tools/make_report_figs.py ; tools/fig_identifiability.py ;
#                                   MUJOCO_GL=osmesa python tools/render_episode.py --model ppo_arm_T1...
```

### 후속 가능 작업(선택)
- **강건성/일반화**: 학습 분포 밖 latent·미지 정책에서의 식별성(현재는 학습분포 내 홀드아웃).
- **2채널 EMG 제약판**: 현실 앵커(중삼각근·이두)만으로의 식별성(privileged 전체 대비 천장 비교).
- **회전근개 개별 분해 시도**: 묶음 대신 개별 s_F — degeneracy 한계 정량(예상: 낮은 R²).
- 더 큰 데이터/스윕, s_L 식별성 개선, T·plane 등 latent별 식별성 층화 분석.
