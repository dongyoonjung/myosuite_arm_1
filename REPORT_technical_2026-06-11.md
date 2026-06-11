# myoArm 근골격 파라미터 추정 — 엄밀 기술 보고서

> 학습 구성요소(**RL 정책 · fPCA 참조생성 · VAE 참조생성 · 시계열 회귀기**)의 입출력·구조·목적함수·하이퍼파라미터·학습절차를 정리. 수식은 LaTeX. (GitHub/VSCode/Typora에서 렌더.)

---

## 0. 표기와 전역 설정

- 모델: MuJoCo myoArm. 물리 적분 500 Hz(timestep $0.002\,$s), 제어 50 Hz(frame skip $F{=}10$, $\Delta t{=}0.02\,$s).
- 독립 관절 27 + 커플 11(견갑 rhythm, equality), 근육 63.
- 한 에피소드 길이 $H$ 스텝: 거상-rise 모드 $H{=}175$($3.5\,$s), 손목·full-sequence 모드 $H{=}200$($4.0\,$s).
- 추적 자유도 집합 $\mathcal D$, 행동 근육 집합 $\mathcal A$, 섭동 채널 집합 $\mathcal C$:

| 모드 | $\mathcal D$ (추적 DoF) | $|\mathcal A|$ (행동근) | $|\mathcal C|$ (섭동채널) | obs 차원 |
|---|---|---|---|---|
| 기본(손목잠금) | elv, plane, elbow, prosup (4) + rot(자유) | 26 | 10 | 76 |
| 손목해제·full-seq | +wrist flex, wrist dev (6) | 32 | 12 | 96 |

---

## 1. 참조(track 대상) 생성 ① — fPCA / ProMP  *(최종 채택)*

실제 사람 동작(KIMHu)에서 *모방 대상 운동학 궤적*을 생성한다. 정책 학습과 분리된 **데이터→궤적** 생성기.

### 1.1 입력 (데이터셋)
KIMHu 스켈레톤에서 반복별 상승(또는 전체) 구간을 위상 $N$점으로 리샘플한 다채널 곡선:
$$
\mathcal X=\{\mathbf X^{(m)}\in\mathbb R^{N\times d}\}_{m=1}^{M},\qquad
(N,d)=\begin{cases}(50,4)&\text{rise: elev, elbow, plane, prosup}\\(80,6)&\text{full: +wrist flex, wrist dev}\end{cases}
$$
$M\approx 200$ (T1 203, T2 192). 채널별 표준화: $\tilde X_{tc}=(X_{tc}-\mu_c)/\sigma_c$, 벡터화 $\mathbf x^{(m)}=\mathrm{vec}(\tilde{\mathbf X}^{(m)})\in\mathbb R^{Nd}$.

### 1.2 학습 (functional PCA)
표본 평균 $\bar{\mathbf x}=\tfrac1M\sum_m\mathbf x^{(m)}$, 중심화 $\mathbf x_c^{(m)}=\mathbf x^{(m)}-\bar{\mathbf x}$. SVD로 주성분:
$$
\mathbf X_c = \mathbf U\mathbf S\mathbf V^\top,\qquad
\lambda_k=\frac{S_{kk}^2}{M-1},\qquad
K=\min\Big\{K:\textstyle\frac{\sum_{k\le K}\lambda_k}{\sum_k\lambda_k}\ge 0.95\Big\}.
$$
주성분 $\boldsymbol\phi_k$ (=$\mathbf V$의 열), 점수 $w_k^{(m)}=\langle\mathbf x_c^{(m)},\boldsymbol\phi_k\rangle$, 표준화 점수 $\hat w_k=w_k/\sqrt{\lambda_k}$. (실측 $K{=}13$(T1,rise)$\sim24$(T2,full).)

### 1.3 출력 (생성)
**점수공간 KDE**(소량 데이터·비가우시안 보존): 실제 점수벡터 하나를 부트스트랩 후 가우시안 jitter,
$$
\hat{\mathbf z}=\hat{\mathbf w}^{(j)}+\boldsymbol\epsilon,\quad j\sim\mathrm{Unif}\{1..M\},\ \boldsymbol\epsilon\sim\mathcal N(\mathbf 0,h^2\mathbf I),\ h=0.35,
$$
$$
\hat{\mathbf x}=\bar{\mathbf x}+\sum_{k=1}^{K}\hat z_k\sqrt{\lambda_k}\,\boldsymbol\phi_k,\qquad
\hat{\mathbf X}=\mathrm{unvec}(\hat{\mathbf x})\odot\boldsymbol\sigma+\boldsymbol\mu .
$$
시간워핑: 이동시간 $T\sim\hat{\mathrm{Emp}}\{T^{(m)}\}$ 샘플 후 위상 $\varphi(t)=\min(t/T,1)$로 $\hat{\mathbf X}$를 $H$스텝에 보간 → 참조 $q^{\mathrm{ref}}_{i}(t)$.

### 1.4 성질·결과
- $\hat{\mathbf z}\sim\mathcal N(0,I)$로 두면 생성 공분산 $=\sum_k\lambda_k\boldsymbol\phi_k\boldsymbol\phi_k^\top$ = 경험 공분산(절단) → **under-dispersion 없음**.
- 실측(생성 vs 실제): 정점 $94{\pm}9°$ vs $95{\pm}9°$, 주변분포 KS $0.07$–$0.14$, 1-NN상 비복사. VAE보다 분포 충실.

---

## 2. 참조 생성 ② — VAE  *(비교군)*

### 2.1 입출력·구조
입력/출력 = 정규화·벡터화 곡선 $\mathbf x\in\mathbb R^{Nd}$ ($Nd{=}200$, rise 4채널). 잠재 $\mathbf z\in\mathbb R^{6}$.
$$
\text{Encoder } q_\theta(\mathbf z|\mathbf x):\ \mathbf h=\mathrm{MLP}_{[200\to128\to128]}(\mathbf x),\quad
\boldsymbol\mu_z=W_\mu\mathbf h,\ \log\boldsymbol\sigma_z^2=W_\sigma\mathbf h,
$$
$$
\text{Decoder } p_\theta(\mathbf x|\mathbf z):\ \hat{\mathbf x}=\mathrm{MLP}_{[6\to128\to128\to200]}(\mathbf z).
$$
재매개화 $\mathbf z=\boldsymbol\mu_z+\boldsymbol\sigma_z\odot\boldsymbol\varepsilon,\ \boldsymbol\varepsilon\sim\mathcal N(0,I)$.

### 2.2 손실 (β-VAE ELBO)
$$
\mathcal L_{\mathrm{VAE}}=\underbrace{\big\|\mathbf x-\hat{\mathbf x}\big\|_2^2}_{\text{재구성}}+\beta\,\underbrace{\tfrac12\sum_{k}\big(\sigma_{z,k}^2+\mu_{z,k}^2-1-\log\sigma_{z,k}^2\big)}_{\mathrm{KL}(q_\theta\|\,\mathcal N(0,I))},\quad \beta=0.5.
$$
**학습**: Adam($\eta{=}10^{-3}$, wd $10^{-5}$), 1500 epoch, full-batch. **생성**: $\mathbf z\sim\mathcal N(0,I)\to$ decoder. **한계**: $\beta$KL·소량데이터로 분포 폭 과소(정점 sd $9\to4$), 잠재 6중 3 활성(붕괴).

---

## 3. 강화학습 정책 (PPO)

### 3.1 MDP — 관측·행동·전이
**관측** $\mathbf o_t\in\mathbb R^{76/96}$ (정규화 전):
$$
\mathbf o_t=\Big[\;\underbrace{\mathbf q_{\mathcal D\cup\{\text{rot}\}}/\pi}_{\text{관절각}},\ \underbrace{\dot{\mathbf q}/5}_{\text{속도}},\ \underbrace{\mathbf a^{\mathrm{act}}}_{\text{근활성}},\ \underbrace{\mathbf q^{\mathrm{ref}}/\pi,\ \mathbf e/\pi,\ \mathbf q^{\mathrm{ref}}_{+5}/\pi}_{\text{참조·오차·선행}},\ \underbrace{[\varphi,\,\mathbb 1_{\text{hold}}]}_{\text{위상}},\ \underbrace{\boldsymbol\ell}_{\text{latent }5},\ \underbrace{[\mathbf s_F,\mathbf s_L]}_{\text{섭동 }2|\mathcal C|},\ \underbrace{\tau}_{\text{과제}}\Big],
$$
$\mathbf e=\mathbf q^{\mathrm{ref}}-\mathbf q$ (추적오차), $\boldsymbol\ell=(T,\text{peak},\text{skew},\text{plane},\text{elbow})$ 정규화. **VecNormalize**(running mean/var, clip $\pm10$)로 $\mathbf o_t$ 표준화 후 정책 입력.

**행동** $\mathbf a_t\in[-1,1]^{|\mathcal A|}$ → 근육 흥분 $u=\tfrac12(\mathbf a_t+1)\in[0,1]^{|\mathcal A|}$. 비행동 근육은 $u{=}0$.

**운동노이즈(Harris–Wolpert, 신호의존)** 및 muscle 활성동역학:
$$
u\leftarrow\mathrm{clip}\big(u+\eta,\,0,1\big),\quad \eta\sim\mathcal N\!\big(0,(\sigma_{\mathrm{sd}}u+\sigma_0)^2\big),\ \sigma_{\mathrm{sd}}{=}0.1,\ \sigma_0{=}0.01.
$$
$\mathbf{ctrl}=u$ 적용 후 $F$회 `mj_step`(MuJoCo muscle dyntype이 활성→힘 변환).

**섭동(파라미터 약화)** 채널 $c$의 근육 $i$에 force scale $s_F^c$, length scale $s_L^c$:
$$
\text{gainprm}_{i,2}\!\leftarrow\! s_F^c\,\text{gainprm}^0_{i,2},\qquad
\text{gainprm}_{i,0{:}2}\!\leftarrow\!\text{gainprm}^0_{i,0{:}2}/s_L^c,
$$
($\text{gainprm}_2{=}F_{\max}$, $\text{gainprm}_{0:2}{=}$작동 길이범위). 희소-$k$: 매 에피소드 $k\!\sim\!\mathrm{Cat}(\{1,2,3\};0.5,0.3,0.2)$ 채널만 $s_F\!\sim\!U[0.05,1]$, $s_L\!\sim\!U[0.7,1.2]$.

### 3.2 보상
$$
\boxed{\,r_t=\mathrm{TASK}_t\cdot\mathrm{QUALITY}_t+w_e\,\mathrm{EFFORT}_t+\mathrm{pen}^{\mathrm{rot}}_t\,}
$$
$$
\mathrm{TASK}_t=\sum_{i\in\mathcal D}w_i\exp\!\Big(-k_i^{\mathrm{out}}\big[\max(0,|q_i-q^{\mathrm{ref}}_i|-b_i)\big]^2\Big)\quad(\text{양면 soft 밴드}),
$$
$$
\mathrm{QUALITY}_t=\exp\!\Big(-\big(c_J\overline{\ddot q^2}+c_D\overline{(\mathbf a_t-\mathbf a_{t-1})^2}+c_S\,\mathbb 1_{\text{hold}}\overline{\dot q^2}\big)\Big),\qquad
\mathrm{EFFORT}_t=-\tfrac1{|\mathcal A|}\textstyle\sum_j a^{\mathrm{act}\,p}_{j},
$$
$$
\mathrm{pen}^{\mathrm{rot}}_t=-w_r\big[\max(0,|q_{\text{rot}}|-50°)\big]^2,\qquad
\text{종료}: |e_{\text{elv}}|>50°\ \lor\ |e_{\text{elbow}}|>60°\ \lor\ \max|\dot q|>40\ \Rightarrow r_t\!-\!1.
$$
계수: $w_i=(0.15,0.15,0.25,0.20[,0.1,0.1]_{\text{wrist}})$, $b_i\approx10$–$20°$, $k^{\mathrm{out}}\!\in[12,30]$, $w_e{=}0.05$, $p{=}2$, $c_J{=}5{\times}10^{-6}$, $c_D{=}0.5$, $c_S{=}1.0$, $w_r{=}0.5$.

### 3.3 정책망·PPO 목적
가우시안 정책 $\pi_\psi(\mathbf a|\mathbf o)=\mathcal N(\boldsymbol\mu_\psi(\mathbf o),\mathrm{diag}\,e^{2\boldsymbol\sigma_\psi})$, $\boldsymbol\mu_\psi,\,V_\psi$는 분리 MLP $[256,256]$+tanh. PPO clipped surrogate:
$$
\mathcal L^{\mathrm{PPO}}(\psi)=\mathbb E\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1{-}\epsilon,1{+}\epsilon)\hat A_t\big)\Big]
-c_v\,\mathbb E\big[(V_\psi-\hat R_t)^2\big]+c_e\,\mathbb E[\mathcal H(\pi_\psi)],
$$
$\rho_t=\frac{\pi_\psi(\mathbf a_t|\mathbf o_t)}{\pi_{\psi_{\text{old}}}}$, GAE $\hat A_t$($\gamma{=}0.99,\lambda{=}0.95$). 하이퍼파라미터:

| $\epsilon$ | lr | rollout($n{\times}$envs) | minibatch | epochs | $\gamma$ | $\lambda$ | $c_v$ | $c_e$ | grad clip |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | $3{\times}10^{-4}$ | $512{\times}20$ | 4096 | 10 | 0.99 | 0.95 | 0.5 | 0 | 0.5 |

### 3.4 학습 절차 (curriculum + warm-start)
1. **M1** 건강·고정(T1) → 검증 게이트(정점 $91{\pm}11°$·단조·인간대역속도·4–12 Hz 떨림無·RMS).
2. **M2** latent 분포 ON + T2(mix). 3. **M3** 섭동 $k{=}1\!\to\!k{\le}3$. 각 단계 이전 가중치 warm-start(obs 차원 불변).
4. 참조 생성기·운동노이즈·손목해제는 플래그(`ref_gen∈{warp,vae,fpca}`, `motor_noise`, `wrist`)로 환경에 주입.

---

## 4. 식별 회귀기 (시계열 TCN)

### 4.1 입력·출력
$M4$ 롤아웃 데이터 $\{(\mathbf Z^{(n)},\boldsymbol\ell^{(n)},\,\mathbf y^{(n)})\}$:
- **입력 시계열** $\mathbf Z\in\mathbb R^{T\times C}$. KIN-only: $C=2(|\mathcal D|{+}1)+|\mathcal D|$ (관절각·속도·추종오차; 기본 14, 손목 20). 가변길이 → $T_{\max}$ 제로패딩 + 마스크 $\mathbf m\in\{0,1\}^{T}$. (privileged 변형은 +63 근활성.)
- **공변량** $\boldsymbol\ell\in\mathbb R^{6}$ (latent+과제, known nuisance).
- **출력/라벨** $\mathbf y=[\mathbf s_F;\mathbf s_L]\in\mathbb R^{2|\mathcal C|}$ (기본 20, 손목 24).

채널·공변량은 유효스텝 통계로 표준화.

### 4.2 구조 (dilated TCN + masked pooling)
$$
\mathbf H^{(0)}=\mathrm{Conv1d}_{k7,s2}(\mathbf Z^\top),\quad
\mathbf H^{(l)}=\mathrm{ReLU}\,\mathrm{GN}\,\mathrm{Conv1d}_{k5,\,\mathrm{dil}=2^{l}}(\mathbf H^{(l-1)}),\ l=1,2,3,
$$
$$
\mathbf g=\big[\,\underbrace{\textstyle\frac{\sum_t m_t\mathbf H_t}{\sum_t m_t}}_{\text{masked avg}};\ \underbrace{\max_{t:m_t=1}\mathbf H_t}_{\text{masked max}};\ \boldsymbol\ell\,\big],\qquad
\hat{\mathbf y}=\mathrm{MLP}_{[\,\cdot\,\to128\to128\to 2|\mathcal C|]}(\mathbf g).
$$
폭 64→128, GroupNorm, 스트라이드·dilation으로 시간 동역학(상승 기울기·떨림·이탈시점) 포착.

### 4.3 손실·학습
약화 정도 재가중 $w_n=1+2\max_c(1-s_F^{c,n})$, Huber(SmoothL1):
$$
\mathcal L=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}w_n\cdot\tfrac1{2|\mathcal C|}\sum_o\mathrm{Huber}\big(\hat y^{(n)}_o-y^{(n)}_o\big).
$$
Adam($10^{-3}$,wd $10^{-5}$)+cosine, batch 256, 100–120 epoch, 85/15 홀드아웃 best-val. 평가 = 채널별 $R^2,\,$MAE.

### 4.4 주요 결과 (KIN-only)
| 설정 | 입력 | 어깨/팔꿈치 $R^2$ | 손목근 $R^2$ |
|---|---|---|---|
| rise, T1+T2, 손목잠금 | KIN 14ch | **0.81** | — |
| full-seq, 손목해제, T1 | KIN 20ch | 0.60 | 0.81 |
| full-seq, 손목해제, T1+T2 | KIN 20ch | 0.55 | 0.74 |

(참고: 요약통계 스냅샷-MLP의 KIN-only는 $R^2{\approx}0.36$ → 시계열 TCN이 동역학으로 대폭 향상.)

---

## 5. 산출물
모델 `ppo_arm_{T1,M2a,M2b,M3a,M3b,M3v,wrist,wrist_mix}.zip`, 회귀 `regressor_{seqG,kinvae,wrist,wrist_mix}.pt`, 생성기 `out/{fpca,fpcafull,vae}_*.{npz,pt}`. 코드: `custom_envs/`, `train.py`, `references/{fpca_ref,fpca_full_ref,vae_ref,extract_traj*}.py`, `gen_seq_data.py`, `regress_seq.py`.
