# myoArm 근육 파라미터 추정 — 상세 기술 보고서 (발표용)

> 모든 학습 구성요소를 **단계별·차원별·항별**로 상세 기술. RL(관측·행동·보상·알고리즘), 참조생성(fPCA·VAE), 회귀기(TCN). 수식 LaTeX(GitHub/VSCode/Typora 렌더). 발표 자료로 그대로 사용 가능하도록 "무엇을·왜·어떻게"를 명시.

## 목차
- A. 문제 정의와 전체 파이프라인
- B. 시뮬레이션 환경 (myoArm)
- C. 강화학습 정책 — 관측 / 행동 / 보상 / 알고리즘 / 학습절차
- D. 참조(모방대상) 생성 — KIMHu 추출 → fPCA / VAE
- E. 식별 회귀기 (시계열 TCN)
- F. 결과 종합

---

# A. 문제 정의와 전체 파이프라인

## A.1 무엇을 푸는가
상지 근골격 시뮬레이션(myoArm)에서 특정 근육의 **최대 힘**($F_{\max}$, 기호 $s_F$로 배율화)과 **최적 길이**($L_{opt}$, $s_L$)를 인위적으로 낮춰 "약화"를 만든다. 그 약화된 팔이 표준 거상 동작을 수행할 때의 **관절 운동학**(joint kinematics)만 관측하여, 어느 근육이 얼마나 약한지($s_F, s_L$)를 복원하는 **회귀기**를 학습한다.

- **목표 = sim 내부 식별성**(identifiability). 실제 환자로 외삽(sim-to-real)이 아니라, 정답이 알려진 시뮬레이션에서 "관측으로 파라미터를 복원할 수 있는가"를 측정.
- **핵심 메커니즘 = soft motion tracking.** 가상 팔(정책)이 사람 동작(참조)을 *느슨하게* 따라 한다. 느슨하게 두는 이유: 빡빡하게 따라 하면 근육이 약해도 다른 근육으로 보상해 약화가 가려진다. 느슨하면 **경증 약화 → 활성 보상 / 중증 약화 → 운동학 이탈**로 자연히 갈라져 식별 가능.

## A.2 파이프라인 (4단계 + 학습 2종)
```
[1] 참조 생성        KIMHu 실데이터 → (fPCA/VAE) → 다양한 목표 궤적 q_ref(t)
[2] 정책 학습(RL)    PPO로 "약화 조건을 알고 q_ref를 soft 추적"하는 정책 π 학습
[3] 데이터 생성(M4)  π를 (약화×동작변동) 조합으로 대량 롤아웃 → (운동학 시계열, 약화라벨)
[4] 회귀 학습(G)     시계열 TCN: 운동학 → (s_F, s_L) 복원, 채널별 식별성 R² 보고
```
학습되는 신경망은 **두 개**: (2)의 RL 정책 π, (4)의 회귀기. 참조 생성은 fPCA(통계) 또는 VAE(딥러닝).

---

# B. 시뮬레이션 환경 (myoArm)

## B.1 물리 모델
- MuJoCo 3.3.0 `myoarm.xml`. 관절 38개(독립 27 + 견갑 rhythm 커플 11, equality 제약으로 자동 연동), **근육 액추에이터 63개**.
- 어깨 독립 3축: `elv_angle`(거상 평면), `shoulder_elv`(거상각 0–180°), `shoulder_rot`(축 회전) + `elbow_flexion`(팔꿈치) + `pro_sup`(전완 회내외).
- **시간**: 물리 적분 500 Hz(timestep $0.002$s). 정책은 **50 Hz**로 제어(frame skip $F=10$, 제어 간격 $\Delta t=0.02$s) — 한 제어 스텝마다 동일 명령으로 물리 10번 적분.
- **원위 잠금**: 손목·손가락 22개 관절을 MuJoCo **MjSpec equality 제약**(상수 0)으로 고정 → 분석 범위를 어깨·팔꿈치로 한정. (손목 해제 모드는 후술 §B.6.)

## B.2 행동공간 — 어떤 근육을 제어하나
정책이 직접 구동하는 근육 **26개**($\mathcal A$):
- 어깨 15: `DELT1,2,3`(삼각근 전/중/후), `SUPSP`(극상근), `INFSP,SUBSC,TMIN`(회전근개), `TMAJ`(대원근), `PECM1,2,3`(대흉근), `LAT1,2,3`(광배근), `CORB`(오훼완근)
- 팔꿈치 9: `TRIlong,lat,med`(삼두), `ANC`, `SUP`, `BIClong,short`(이두), `BRA`, `BRD`
- 회내 2: `PT`, `PQ` (pro_sup 구동)

나머지 37개(손목 6 + 손가락 31)는 정책이 안 건드림(흥분 0).

## B.3 근육 구동과 운동노이즈 (행동 → 힘)
정책 출력 $\mathbf a_t\in[-1,1]^{26}$ → 근육 흥분(excitation)으로 사상:
$$
u_j=\tfrac12(a_{t,j}+1)\in[0,1].
$$
**신호의존 운동노이즈** (Harris & Wolpert 1998; 표준편차가 명령 크기에 비례 → 생리적 변동·동시수축 유발):
$$
u_j\leftarrow \mathrm{clip}\!\big(u_j+\eta_j,\,0,1\big),\qquad \eta_j\sim\mathcal N\!\big(0,(\sigma_{\mathrm{sd}}\,u_j+\sigma_0)^2\big),\quad \sigma_{\mathrm{sd}}=0.1,\ \sigma_0=0.01.
$$
이 $u$를 `data.ctrl`에 넣고 MuJoCo가 **근육 활성동역학**(activation dynamics, 시간상수 $\tau_{act}{=}10$ms·$\tau_{deact}{=}40$ms)과 **force–length–velocity** 곡선으로 실제 장력으로 변환. 즉 흥분 $u$ → 활성 $a^{\mathrm{act}}$ → 힘. (활성 $a^{\mathrm{act}}$는 관측에 포함, §C.1.)

## B.4 약화(섭동)를 넣는 법
근육 $i$에 force 배율 $s_F$, length 배율 $s_L$ 적용 — MuJoCo muscle gain 파라미터를 직접 수정(런타임, 재컴파일 불요):
$$
\text{gainprm}_{i,2}\leftarrow s_F\cdot\text{gainprm}^0_{i,2}\quad(\text{gainprm}_2=F_{\max}),\qquad
\text{gainprm}_{i,0:2}\leftarrow \text{gainprm}^0_{i,0:2}/s_L\quad(\text{작동 길이범위}).
$$
$s_F{=}0.3$이면 등척력이 정확히 $0.3$배로 검증됨. **섭동 채널 10개**($\mathcal C$): 개별 8(DELT1/2/3, SUPSP, PECM1, CORB, BIClong, TRIlong) + 묶음 2(A={INFSP,SUBSC,TMIN} 하부 회전근개, B={TMAJ,LAT1-3}). 묶음은 한 배율을 공유(개별 회전근개는 서로 degenerate라 묶어 추정).

**희소-k 커리큘럼**: 매 에피소드 손상 채널 수 $k\sim\mathrm{Cat}(\{1,2,3\};\,0.5,0.3,0.2)$만 약화시키고, 그 채널만 $s_F\sim U[0.05,1.0]$($0.05$=중증), $s_L\sim U[0.70,1.20]$. 나머지 채널은 $s_F{=}s_L{=}1$(정상).

## B.5 에피소드 (raise-and-hold)
- 길이 $H$: 거상-rise 모드 $H{=}175$($3.5$s), 손목·full-seq 모드 $H{=}200$($4.0$s).
- reset: 참조의 시작 자세로 관절 세팅(평가는 위상 0=시작, 학습은 RSI로 임의 위상에서도 시작).
- 종료: 추적 발산(아래 §C.3) 또는 $H$ 도달(truncation).

## B.6 손목 해제 모드 (선택, T1 full-sequence)
사용자 요청 변형: 손목 equality 제약 해제(neq 33→31, 견갑11+손가락20만 잠금) → 손목 자유. 행동근에 손목근 6개(ECRL/ECRB/ECU/FCR/FCU/PL) 추가(26→**32**), 추적 DoF에 손목 굴곡·편위 추가, 섭동 채널 12개(+W_ext, W_flex), obs 96차원. 참조는 전체 사이클(저점→정점→복귀).

---

# C. 강화학습 정책 (PPO)

## C.1 관측 (Observation) — 76차원, 차원별 상세
매 제어 스텝 정책에 들어가는 벡터 $\mathbf o_t$. 표준화 전 원소(정규화 상수는 안정화용; 이후 VecNormalize가 running mean/var로 한 번 더 표준화):

| # | 블록 | 차원 | 내용 / 정규화 | 의미 |
|---|---|---|---|---|
| 1 | 관절각 $\mathbf q$ | 5 | [shoulder_elv, elv_angle, elbow_flexion, pro_sup, shoulder_rot] (rad) ÷ $\pi$ | 현재 자세 |
| 2 | 관절속도 $\dot{\mathbf q}$ | 5 | 위 5축 속도(rad/s) ÷ 5 | 운동 상태 |
| 3 | 근육 활성 $\mathbf a^{\mathrm{act}}$ | 26 | 26개 행동근의 현재 활성도 [0,1] | 자기 근육 상태(협응·anti-tremor에 필요) |
| 4 | 참조 현재 $\mathbf q^{\mathrm{ref}}_t$ | 4 | [elv, plane, elbow, prosup] 참조각 ÷ $\pi$ | 지금 따라갈 목표 |
| 5 | 추적 오차 $\mathbf e_t$ | 4 | $\mathbf q^{\mathrm{ref}}-\mathbf q$ ÷ $\pi$ | 얼마나 벗어났나 |
| 6 | 참조 선행 $\mathbf q^{\mathrm{ref}}_{t+5}$ | 4 | 5스텝(0.1s) 뒤 참조 ÷ $\pi$ | 다가올 목표(예측 제어) |
| 7 | 위상 | 2 | [$\varphi=t/H$, $\mathbb 1_{\text{hold}}$] | 동작 진행도·유지국면 여부 |
| 8 | latent $\boldsymbol\ell$ | 5 | [T/3, peak/120, skew/2, plane/45, elbow_peak/130] | 이 에피소드 동작의 변동 인자 |
| 9 | 섭동조건 $[\mathbf s_F,\mathbf s_L]$ | 20 | 10채널×(force,length) 배율 (건강=1) | **약화 정보**(privileged: 적응한 만성환자 가정) |
| 10 | 과제 | 1 | T2면 1, T1이면 0 | 어느 표준동작인지 |

합계 $5{+}5{+}26{+}4{+}4{+}4{+}2{+}5{+}20{+}1=\mathbf{76}$. (손목 모드: 1·2·4·5·6에 손목 2DoF 추가, 9가 24 → **96**.)

핵심 설계점:
- **(9) 섭동을 obs에 준다**: 만성 환자는 CNS가 자신의 근육 상태에 *이미 적응*했으므로, 정책이 약화를 알고 보상하는 것이 현실적. 약화가 obs에 있어 정책은 보상 전략을 학습하고, 그 결과 운동학·활성에 약화의 "서명"이 남는다 → 회귀기가 그걸 역산.
- **(8) latent**: 사람마다 다른 속도(T)·정점(peak)·모양(skew)·평면(plane). 회귀에선 known 공변량(우리가 샘플하므로 known).

## C.2 행동 (Action)
$\mathbf a_t\in[-1,1]^{26}$ (가우시안 정책의 표본) → §B.3대로 흥분 $u$ → 근육 힘. 26개 = 어깨15+팔꿈치9+회내2.

## C.3 보상 (Reward) — 항별 상세
정책은 누적 보상을 최대화. 매 스텝:
$$
\boxed{\,r_t=\underbrace{\mathrm{TASK}_t}_{\text{추적}}\cdot\underbrace{\mathrm{QUALITY}_t}_{\text{품질}}\;+\;w_e\,\underbrace{\mathrm{EFFORT}_t}_{\text{노력}}\;+\;\underbrace{\mathrm{pen}^{\mathrm{rot}}_t}_{\text{회전}}\,}
$$

### (1) TASK — 양면 soft 밴드 추적 (핵심)
$$
\mathrm{TASK}_t=\sum_{i\in\{\text{elv,plane,elbow,prosup}\}} w_i\;\exp\!\Big(-\,k_i^{\mathrm{out}}\,\big[\max(0,\;|q_i-q^{\mathrm{ref}}_i|-b_i)\big]^2\Big).
$$
- **밴드 $b_i$ 안**($|q_i-q^{\mathrm{ref}}_i|\le b_i$): 지수 인자가 0 → 기여 $=w_i$ (**벌점 없음**). 사람마다 다른 자세·속도를 허용.
- **밴드 밖**: 초과분 $(|e|-b_i)$의 제곱에 $k_i^{\mathrm{out}}$ 비례로 급감.
- 예시(elv): $b{=}10°$, $k^{\mathrm{out}}{=}30$, $w{=}0.15$. 오차 $5°$ → 기여 $0.15$(밴드 내). 오차 $20°$ → 초과 $10°{=}0.175$rad → $0.15\,e^{-30\cdot0.175^2}=0.15\cdot e^{-0.92}=0.060$ (급감).
- 계수(LOCKED): $w=(0.15,0.15,0.25,0.20)$, $b=(10°,15°,12°,20°)$, $k^{\mathrm{out}}=(30,18,22,12)$.
- **왜 거상 가중이 0.15로 낮나**: 빡빡하면 약화가 궤적에 안 드러남. 느슨해야 중증 약화가 *이탈*로 보임(식별 신호 보존).

### (2) QUALITY — 매끄러움 (0~1, 곱 인자)
$$
\mathrm{QUALITY}_t=\exp\!\Big(-\big(c_J\,\overline{\ddot q^2}\;+\;c_D\,\overline{(\mathbf a_t-\mathbf a_{t-1})^2}\;+\;c_S\,\mathbb 1_{\text{hold}}\,\overline{\dot q^2}\big)\Big).
$$
- $\overline{\ddot q^2}$: 추적 DoF 가속 제곱평균(부드러움). $c_J=5\times10^{-6}$.
- $\overline{(\mathbf a_t-\mathbf a_{t-1})^2}$: 행동 변화량(떨림 억제). $c_D=0.5$.
- $\mathbb 1_{\text{hold}}\overline{\dot q^2}$: 정점유지 국면에서만 속도 벌점(정지 유도). $c_S=1.0$.

### (3) EFFORT — 힘 아끼기 (곱 밖 가산, 음수)
$$
\mathrm{EFFORT}_t=-\frac1{|\mathcal A|}\sum_{j\in\mathcal A}\big(a^{\mathrm{act}}_j\big)^{p},\qquad w_e=0.05,\ p=2.
$$
과활성·동시수축 억제 → 자연스러운 최소노력. ($p{=}3$ cubic은 escalation 옵션.)

### (4) 회전 정칙화 + 종료
$$
\mathrm{pen}^{\mathrm{rot}}_t=-w_r\big[\max(0,|q_{\text{rot}}|-50°)\big]^2,\quad w_r=0.5.
$$
shoulder_rot은 참조 없는 자유 DoF(데이터로 신뢰측정 불가) — 극단·치팅만 약하게 차단. 물리가 거상에 필요한 외회전을 공급.
**종료(divergence)**: $|e_{\text{elv}}|>50°$ 또는 $|e_{\text{elbow}}|>60°$ 또는 $\max|\dot q|>40$rad/s → $r_t\mathrel{-}=1$, 에피소드 종료. 밴드를 **넓게(50°)** 둔 이유: 약화로 인한 이탈은 살아남아 신호가 되게, 정상 발산만 차단.

### 설계 의도 요약
"느슨하게 따라가되, 부드럽고, 힘 아끼게." → 경증=ACT 보상으로 궤적 회복, 중증=KIN 이탈. 약화 정도가 연속적으로 운동학·활성에 매핑되어 회귀로 복원 가능.

## C.4 정책·가치망과 PPO
- **정책** $\pi_\psi(\mathbf a|\mathbf o)=\mathcal N\!\big(\boldsymbol\mu_\psi(\mathbf o),\,\mathrm{diag}\,e^{2\boldsymbol\sigma_\psi}\big)$. $\boldsymbol\mu_\psi$: MLP [256,256]+tanh, 출력 26(또는 32). $\boldsymbol\sigma_\psi$: 상태독립 log-std 파라미터.
- **가치** $V_\psi(\mathbf o)$: 별도 MLP [256,256]+tanh.
- **VecNormalize**: 관측·보상을 running mean/var로 표준화(clip $\pm10$), 20개 병렬 환경 통계 공유.
- **GAE**: $\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$, $\delta_t=r_t+\gamma V(\mathbf o_{t+1})-V(\mathbf o_t)$, $\gamma{=}0.99$, $\lambda{=}0.95$.
- **PPO clipped surrogate** ($\rho_t=\pi_\psi/\pi_{\psi_{old}}$):
$$
\mathcal L(\psi)=-\,\mathbb E\Big[\min\!\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1{-}\epsilon,1{+}\epsilon)\hat A_t\big)\Big]+c_v\,\mathbb E\big[(V_\psi-\hat R_t)^2\big]-c_e\,\mathbb E[\mathcal H(\pi_\psi)].
$$

**하이퍼파라미터**: $\epsilon{=}0.2$, lr $3\times10^{-4}$, rollout $512\times20{=}10240$, minibatch $4096$, epochs $10$, $c_v{=}0.5$, $c_e{=}0$, grad-clip $0.5$. CPU(torch), `OMP_NUM_THREADS` 분산.

## C.5 학습 절차 — 커리큘럼 + 워밍스타트
관측 차원 76을 단계 내내 고정(워밍스타트 호환):
1. **M1** 건강·고정(T1, latent OFF) → **검증 게이트**(정점 91±11°·단조상승·인간대역속도·4–12Hz 떨림無·추종 RMS<12°) 통과.
2. **M2** latent 분포 ON + T2(mix). 분포게이트(참조정점 매칭).
3. **M3** 섭동 커리큘럼 $k{=}1\to k{\le}3$. 각 단계 직전 가중치 warm-start.
참조생성기·운동노이즈·손목해제는 플래그(`--ref-gen`, `--motor-noise`, `--wrist`)로 주입.

---

# D. 참조(모방대상) 생성

가상 팔이 따라 할 **목표 궤적** $q^{\mathrm{ref}}(t)$를 사람 데이터에서 만든다. 한 동작만 쓰면 단조로우니 *다양하고 현실적인* 변형을 생성해야 한다. 방법을 발전시킴: warp(평균+1모드) → VAE → **fPCA(채택)**.

## D.1 KIMHu 원시 → 궤적 데이터셋 (공통 전처리)
- **데이터**: KIMHu(20명, 오른팔, 표준 거상, Kinect V2 30fps 3D 관절위치). 스켈레톤 CSV만 다운로드(`tools/download_kimhu.py`).
- **각도 유도(3D 위치에서만; 사전계산 각도열 금지)**:
  - 거상 $=\angle(\hat{\mathbf{u}}_{arm},\ \hat{\mathbf{d}}_{trunk})$, $\mathbf u_{arm}=$어깨→팔꿈치, $\mathbf d=$체간 아래.
  - 팔꿈치 $=180°-\angle(\text{어깨→팔꿈치},\ \text{손목→팔꿈치})$.
  - 평면 방위 $=\mathrm{atan2}$(위팔 수평성분의 전방·측방 투영). (저거상서 수평성분 작아 노이즈 → `reliable_plane`로 마스킹·보간.)
  - 회내 $=$손바닥 법선의 전완축 둘레 회전. (손목 모드는 손목 굴곡·편위 추가, §B.6.)
- **분절**: rise 모드 = 저점→첫 정점(상승만; 정점 뒤 손목조작은 채널 오염이라 제외). full 모드 = 저점→정점→복귀 전체 사이클.
- **리샘플**: 각 반복을 위상 $N$점으로($N{=}50$ rise, $80$ full). 결과 데이터셋 $\{\mathbf X^{(m)}\in\mathbb R^{N\times d}\}_{m=1}^M$, $d{=}4$(rise) 또는 6(full), $M\approx200$.

## D.2 fPCA / ProMP — 단계별 (채택)
**목적**: 궤적 분포의 평균과 *공분산*을 데이터에서 그대로 잡아 다양성을 재현(VAE의 분포 과소 문제 해결).

**1) 정규화·벡터화**: 채널별 평균·표준편차 $(\mu_c,\sigma_c)$로 표준화 후 평탄화
$$
\mathbf x^{(m)}=\mathrm{vec}\big((\mathbf X^{(m)}-\boldsymbol\mu)/\boldsymbol\sigma\big)\in\mathbb R^{Nd}.
$$

**2) 주성분 분해(SVD)**: 평균 $\bar{\mathbf x}$, 중심화 $\mathbf X_c$,
$$
\mathbf X_c=\mathbf U\mathbf S\mathbf V^\top,\quad \lambda_k=\frac{S_{kk}^2}{M-1},\quad \boldsymbol\phi_k=\mathbf V_{:,k},\quad
K=\min\Big\{K:\tfrac{\sum_{k\le K}\lambda_k}{\sum_k\lambda_k}\ge 0.95\Big\}.
$$
실측 $K{=}13$(T1 rise)~$24$(T2 full). 즉 95% 변동을 ~13–24개 모드가 설명.

**3) 점수 추출·표준화**: $w_k^{(m)}=\langle\mathbf x_c^{(m)},\boldsymbol\phi_k\rangle$, $\hat w_k^{(m)}=w_k^{(m)}/\sqrt{\lambda_k}$.

**4) 생성 — 점수공간 KDE**(소량 데이터·비가우시안 보존; 실제 점수를 부트스트랩 후 작은 jitter):
$$
\hat{\mathbf z}=\hat{\mathbf w}^{(j)}+\boldsymbol\epsilon,\quad j\sim\mathrm{Unif}\{1{..}M\},\ \boldsymbol\epsilon\sim\mathcal N(0,h^2 I),\ h=0.35,
$$
$$
\hat{\mathbf x}=\bar{\mathbf x}+\sum_{k=1}^K\hat z_k\sqrt{\lambda_k}\,\boldsymbol\phi_k\ \xrightarrow{\text{unvec·역정규화}}\ \hat{\mathbf X}\in\mathbb R^{N\times d}.
$$
거상 채널은 단조화(상승 모드), full 모드는 복귀 보존(단조화 안 함).

**5) 시간워핑**: 이동시간 $T\sim\hat{\mathrm{Emp}}\{T^{(m)}\}$ 샘플 → 위상 $\varphi(t)=\min(t/T,1)$로 $\hat{\mathbf X}$를 $H$스텝에 보간 → 참조 $q^{\mathrm{ref}}_i(t)$. ($t>T$는 마지막 자세 유지=raise-and-hold.)

**왜 좋은가(검증)**: $\hat{\mathbf z}\sim\mathcal N(0,I)$이면 생성 공분산$=\sum_k\lambda_k\boldsymbol\phi_k\boldsymbol\phi_k^\top$ = 경험 공분산(절단) → **under-dispersion 없음**. 실측: 정점 실제 $94{\pm}9°$ vs fPCA $95{\pm}9°$(폭 일치), 주변분포 KS $0.07$–$0.14$, 1-NN상 비복사·현실적.

## D.3 VAE — 단계별 (비교군)
**구조**(입력=출력=정규화·평탄화 곡선 $\mathbf x\in\mathbb R^{Nd}$, $Nd{=}200$; 잠재 $\mathbf z\in\mathbb R^6$):
$$
\text{인코더}:\ \mathbf h=\mathrm{ReLU}\,\mathrm{Lin}_{128}\circ\mathrm{ReLU}\,\mathrm{Lin}_{128}(\mathbf x),\quad \boldsymbol\mu_z=\mathrm{Lin}_6(\mathbf h),\ \log\boldsymbol\sigma_z^2=\mathrm{Lin}_6(\mathbf h),
$$
$$
\text{디코더}:\ \hat{\mathbf x}=\mathrm{Lin}_{200}\circ\mathrm{ReLU}\,\mathrm{Lin}_{128}\circ\mathrm{ReLU}\,\mathrm{Lin}_{128}(\mathbf z),\quad \mathbf z=\boldsymbol\mu_z+\boldsymbol\sigma_z\odot\boldsymbol\varepsilon,\ \boldsymbol\varepsilon\sim\mathcal N(0,I).
$$
**손실(β-VAE ELBO)**:
$$
\mathcal L=\underbrace{\|\mathbf x-\hat{\mathbf x}\|_2^2}_{\text{재구성}}+\beta\cdot\underbrace{\tfrac12\sum_k(\sigma_{z,k}^2+\mu_{z,k}^2-1-\log\sigma_{z,k}^2)}_{\mathrm{KL}(q\|\mathcal N(0,I))},\quad \beta=0.5.
$$
**학습**: Adam(lr $10^{-3}$, wd $10^{-5}$), 1500 epoch, full-batch. **생성**: $\mathbf z\sim\mathcal N(0,I)$ → 디코더 → 역정규화 → 시간워핑.

**한계(fPCA에 진 이유)**: $\beta$KL 정칙화 + 소량 데이터(~200)로 **평균화** → 분포 폭 과소(정점 sd $9\to4$), 잠재 6중 ~3개만 활성(posterior collapse). fPCA는 공분산을 구조적으로 보존하므로 분포 충실.

---

# E. 식별 회귀기 (시계열 TCN)

## E.1 입력·출력
M4 롤아웃 데이터 $\{(\mathbf Z^{(n)},\boldsymbol\ell^{(n)},\mathbf y^{(n)})\}$ ($n$=에피소드):
- **입력 시계열** $\mathbf Z\in\mathbb R^{T\times C}$. **운동학(KIN)만**: $C=2(|\mathcal D|{+}1)+|\mathcal D|$ — 추적DoF+회전의 각도·속도, 그리고 추적오차. 기본 $C{=}14$(=5 pos+5 vel+4 err), 손목 $C{=}20$. (EMG/활성 미사용.) 가변길이 → $T_{\max}$ 제로패딩 + 마스크 $\mathbf m\in\{0,1\}^T$.
- **공변량** $\boldsymbol\ell\in\mathbb R^6$ (latent 5 + 과제 1; known nuisance).
- **출력/라벨** $\mathbf y=[\mathbf s_F;\mathbf s_L]\in\mathbb R^{2|\mathcal C|}$ (기본 20, 손목 24).

채널·공변량은 유효스텝 통계로 표준화. (참고: privileged 상한 측정 시 63근 활성을 입력에 추가하면 $C{+}63$.)

## E.2 구조 — dilated TCN + masked pooling
시간 동역학(상승 기울기·떨림·이탈 *시점*)을 포착하기 위한 1D 시간합성곱망:
$$
\mathbf H^{(0)}=\mathrm{ReLU}\,\mathrm{GN}\,\mathrm{Conv1d}_{k=7,\,s=2}(\mathbf Z^\top)\quad(\text{시간 1/2 다운샘플}),
$$
$$
\mathbf H^{(l)}=\mathrm{ReLU}\,\mathrm{GN}\,\mathrm{Conv1d}_{k=5,\ \mathrm{dilation}=2^l}(\mathbf H^{(l-1)}),\quad l=1,2,3\ (\text{폭 }64\to64\to128\to128),
$$
$$
\mathbf g=\Big[\underbrace{\tfrac{\sum_t m_t\mathbf H_t}{\sum_t m_t}}_{\text{masked avg}};\ \underbrace{\max_{t:\,m_t=1}\mathbf H_t}_{\text{masked max}};\ \boldsymbol\ell\Big],\qquad
\hat{\mathbf y}=\mathrm{MLP}_{[\,\cdot\to128\to128\to 2|\mathcal C|]}(\mathbf g).
$$
GroupNorm(배치·패딩 오염 없음), dilation 1·2·4·8로 수용영역 확장, 마스킹으로 패딩 무시. (왜 시계열? 요약통계 스냅샷-MLP는 KIN-only $R^2{\approx}0.36$였으나, 시계열은 동역학을 살려 $0.81$로 대폭↑.)

## E.3 손실·학습
**약화 정도 재가중**(드문 중증 표본 강조) + 이상치 강건 Huber:
$$
w_n=1+2\max_c(1-s_F^{c,n}),\qquad
\mathcal L=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}w_n\cdot\frac1{2|\mathcal C|}\sum_o\mathrm{Huber}\big(\hat y^{(n)}_o-y^{(n)}_o\big).
$$
Adam(lr $10^{-3}$, wd $10^{-5}$) + CosineAnnealing, batch 256, 100–120 epoch, **85/15 홀드아웃** best-val. 평가 = 채널별 $R^2$, MAE.

## E.4 데이터 생성(M4) 상세
학습된 정책을 $N{\approx}16$–$20$k 에피소드 롤아웃(병렬 샤드). 매 에피소드: latent·섭동을 env가 샘플(또는 고정) → 정책이 deterministic 추종(운동노이즈는 ON) → 매 스텝 KIN 채널 기록(step *전* 기록으로 자동리셋 오염 방지) → 라벨 = 그 에피소드 $(\mathbf s_F,\mathbf s_L)$. 약 15% 건강(k=0) 포함.

---

# F. 결과 종합

## F.1 KIN-only 식별성 (운동학만, fPCA 참조, 운동노이즈)
| 설정 | 입력채널 | 어깨/팔꿈치 $s_F$ $R^2$ | 손목근 $R^2$ | $s_L$ $R^2$ |
|---|---|---|---|---|
| **rise, T1+T2, 손목잠금** | KIN 14 | **0.81** | — | 0.56 |
| full-seq, 손목해제, T1 | KIN 20 | 0.60 | 0.81 | 0.28 |
| full-seq, 손목해제, T1+T2 | KIN 20 | 0.55 | 0.74 | 0.28 |

채널별(rise·최고설정 s_F): A_lowcuff .94, DELT1/2 .79–.86, SUPSP/BIClong/TRIlong .86–.87, B_latadd .88, DELT3 .75, PECM1/CORB .62–.65.

## F.2 해석 (발표 포인트)
1. **EMG 없이 움직임만으로 어깨/팔꿈치 근육 최대힘 약화를 강식별($R^2{=}0.81$)** — 임상적으로 모션캡처만으로 근력 손실 추정 가능성 시사.
2. **시계열이 핵심**: 동역학(상승 기울기·이탈 시점)을 살리는 TCN이 요약통계(0.36)보다 압도적. 사람 동작의 *시간 구조*에 약화 정보가 있음.
3. **참조 생성은 fPCA가 최적**: 실제 분포(평균+공분산)를 구조적으로 재현, VAE의 under-dispersion 회피.
4. **트레이드오프**: 손목 해제+full-sequence는 손목근 식별(~0.8)을 *추가*하나, 정보가 적은 복귀·조작 구간이 섞여 거상-rise(어깨근 최정보 구간)를 희석 → 어깨/팔꿈치 식별 저하. 한 프로토콜로 둘 다 최대화 불가.
5. **$s_L$(길이)은 $s_F$(힘)보다 어렵다**: 운동학만으론 길이 파라미터 신호가 약함(중간 수준).

## F.3 산출물
모델 `ppo_arm_{T1,M2a,M2b,M3a,M3b,M3v,wrist,wrist_mix}`, 회귀 `regressor_{G,seqG,kinvae,wrist,wrist_mix}.pt`, 생성기 `out/{fpca,fpcafull,vae}_*`. 코드 `custom_envs/`·`train.py`·`references/`·`gen_seq_data.py`·`regress_seq.py`. 그림 `results/report/fig1~11`, 비디오 `M1_T1.mp4`.
