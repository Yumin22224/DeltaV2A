# DeltaV2A — 발표 자료

---

## View of Correlation

두 모달 간 대응 관계를 분류할 때, **전후 관계(인과)를 고려하는가**가 핵심 축이다.

- **Static alignment**: 두 객체 간 유사성 (I₁↔I₂, A₁↔A₂, I↔A) — 인과 없음
- **Transition alignment**: 변화 쌍 간 대응 — **인과 있음**
  - intra-modal: (I→I'), (A→A')
  - **inter-modal: (I, I') ↔ (A, A') ← 본 연구의 타겟**

---

## Evidence of Correlation

시각-청각 두 모달을 잇는 transition alignment의 근거:

- **E1. Ground Truth Based (Physical):** 물리적 인과가 직접 성립 (ex. 악기 진동이 이미지 흔들림을 유발)
- **E2. 직관 Based (Low-level feature):** 감각 기관의 구조적 유사성에서 비롯된 직관적 대응
- **E3. 문화/경험 Based (High-level feature):** 학습된 문화적 연상 관계

---

## Feature Domain of Correlation

각 모달에서 어떤 범위(Scope), 어떤 수준(Level)의 feature를 다루는가에 따라, 어떤 근거(E1/E2/E3)가 지배적인지가 달라진다.

- **Scope:** Global / Local (semantic-aware or coordinate-based)
- **Level:** Low (color, blur, …) → Mid (texture, style) → High (semantic content)

Feature domain의 level이 높아질수록 주관적 해석의 비중이 커진다. 이는 두 가지 문제를 낳는다:
1. 예술적 해석에는 정답이 없다
2. 해당 수준의 대응 관계를 데이터셋으로 커버하기 어렵다

**→ 따라서 feature domain을 고정하고, 그 domain 안에서의 delta가 특정 증거 scope에서 어떻게 나타나는지를 탐구하는 방향으로 문제를 제한한다.** 또한 대규모 멀티모달 모델이 형성한 임베딩 공간을 활용해 데이터셋 문제를 우회한다.

---

## Target Feature Domain

- **Scope:** Global
- **Level:** Low–Mid (전문 편집 도구에서 흔히 쓰이는 효과 중심)
  - **Image Delta:** Wand 라이브러리 기반 이미지 effect (adaptive_blur, motion_blur, adaptive_sharpen, add_noise, spread, sepia_tone, solarize)
  - **Audio Delta:** Spotify Pedalboard 기반 오디오 effect (lowpass, bitcrush, reverb, highpass, distortion, delay, playback_rate)

---

## [Experiment 1] 대형 멀티모달 모델의 Target Feature Domain 표현력

**질문:** CLIP / CLAP / ImageBind가 이 low-mid level effect 적용에 얼마나 민감하게 반응하는가?

**결과 요약:**
- **ImageBind:** 이미지·오디오 두 도메인 모두에서 가장 민감하게 반응
- **CLIP:** 이미지 도메인에서 반응하나 ImageBind 대비 낮음
- **CLAP:** 오디오 도메인에서 반응

**ImageBind의 실용적 한계:** 오디오 입력을 내부적으로 2초로 제한 → 음악 단위 처리에 부적합

**→ 결론:** LAION-CLAP (음악 데이터 학습, 최대 10초 window)과 CLIP을 채택.
image↔audio를 잇는 intermodal로 **text**를 사용하기로 결정.

---

## Target Correlation Evidence Scope — Style Vocabulary

Text를 intermodal로 사용하면 대응 관계를 explicit하게 표현할 수 있어 해석 가능성이 높아진다.
단, image-audio 사이의 모든 관계를 text가 완전히 담을 수는 없다는 trade-off가 존재한다.

**단어 선정 기준:**

- **ΔE-direction:** 절대적 상태가 아닌 변화의 **방향**을 묘사. style label은 `normalize(CLIP(I') - CLIP(I))`로부터 계산되므로, vocab embedding은 I'가 도달하는 위치가 아닌 변환 벡터의 방향과 align되어야 한다.
- **Cross-modal neutrality:** 물리적/구조적 은유를 통해 시각·청각 두 도메인에 동등하게 적용 가능. 특정 모달의 지각 기관에서 직접 비롯된 단어(빛/색 기반, 음고 기반)는 배제.
- **Anti-anchor:** 어떤 단일 term도 "변화가 일어났다"는 일반 anchor로 작동하지 않도록. 세 가지 perceptual dimension (texture, space, energy)에 걸쳐 coverage를 분산.

```python
STYLE_AXES_PHASES = [
    ("phase_1_texture", [
        ("granular", "polished"),   # 표면 거칠기: 입자적/거친 vs 매끄럽고 정제된
        ("piercing", "muffled"),    # 주파수 침투력: 날카롭고 투과하는 vs 둔하고 필터된
        ("clipped", "pristine"),    # 신호 정합성: 과구동/왜곡된 vs 깨끗하고 온전한
        ("fluid", "rigid"),         # 구조적 유연성: 흐르는/적응하는 vs 굳은/고정된
    ]),
    ("phase_2_space", [
        ("expansive", "confined"),  # 공간적 스케일: 열리고 넓게 퍼지는 vs 좁고 닫힌
        ("layered", "flat"),        # 깊이 차원성: 겹쳐진 평면들 vs 단일 2D 표면
        ("dense", "hollow"),        # 질량감 충전도: 빽빽하고 충만한 vs 성긴/중심이 빈
        ("fragmented", "cohesive"), # 구조적 통일성: 흩어지고 부서진 vs 통일되고 전체적인
    ]),
    ("phase_3_energy", [
        ("agitated", "calm"),       # 운동 에너지: 격렬하고 불안한 vs 고요하고 안정된
        ("oscillating", "steady"),  # 주기적 변동: 리드미컬한 요동 vs 일정하고 안정된
        ("crisp", "muddy"),         # 경계 선명도: 날카롭고 경계 명확한 vs 흐릿하게 합쳐진
        ("deep", "thin"),           # 음조 무게: 두텁고 저역 중심인 vs 가늘고 고역 중심인
    ]),
]
```

이 24-term 집합을 **Style Vocabulary**라 부른다.

---

## System Workflow

```
[Image]  I ──→ wand effect ──→ I'
              CLIP(I), CLIP(I')
              δz = normalize(CLIP(I') - CLIP(I))
              img_style_scores = softmax(δz · V_img / T)   ← 24-dim 확률 분포
                                          ↓
                         [Cross-modal Identity Mapping]
                         aud_style_scores = img_style_scores
                                          ↓
[Audio]  A ──→ CLAP(A) ──→ Controller ──→ activity mask + effect params
                                          ↓
                                    Pedalboard render → A'
```

**Cross-modal Identity Mapping Assumption:**
추론 시 `aud_style_scores = img_style_scores`로 직접 사용.
CLIP과 CLAP은 모두 language supervision으로 학습되었기 때문에, 24개 vocab term("warm", "dark", "ethereal" 등)의 의미를 동일하게 이해한다는 가정.
즉, V_img와 V_aud는 같은 의미 개념을 표현하고 있으므로, CLIP delta로 계산한 style score와 CLAP delta로 계산한 style score는 동일한 perceptual 변화에 대해 동일한 분포를 가진다.
cross-modal alignment를 별도로 학습하지 않고, 두 모델의 **공유 language space**를 modality 간 bridge로 삼는다.

---

## Point 1. Style Label은 어떻게 만들 것인가?

**생성 방식:**

각 vocab term을 단일 template으로 임베딩하면 특정 표현 방식에 민감해질 수 있다.
→ 동일 concept을 다양한 문장으로 표현한 8개 template을 앙상블하여 term embedding을 안정화한다.

**IMG prompt templates (8개):**
```
"a {word} image"            "a {word} photograph"
"an image with a {word} mood"  "a {word} visual style"
"a {word} scene"            "a {word} picture"
"a photo that feels {word}" "artwork with a {word} atmosphere"
```

**AUD prompt templates (8개):**
```
"a {word} sound"            "a {word} music track"
"audio with a {word} mood"  "a {word} sonic texture"
"a {word} musical atmosphere"  "a {word} sounding track"
"music that feels {word}"   "a {word} audio style"
```

**앙상블 과정:**
1. 각 keyword × 8 template = 8개 문장을 CLIP/CLAP text encoder로 임베딩
2. 각 임베딩을 L2 normalize
3. 8개 평균 → 다시 L2 normalize → term embedding 1개

→ CLIP: `V_img` (24×768-dim), CLAP: `V_aud` (24×512-dim)

$$\text{style\_score} = \text{softmax}\!\left(\frac{\delta z \cdot V}{\,T\,}\right) \rightarrow \text{24-dim 확률 분포}$$

---

### [Experiment 2] Style Label Diagnosis

**목적:** Style vocab이 실제로 effect 변화를 포착하고 있는가? Identity mapping assumption은 얼마나 타당한가?

#### 진단 지표 (CLAP-side, training DB 기준)

| 지표 | 수치 | 의미 |
|------|------|------|
| Entropy (mean) | 4.34 bits (max = log₂24 = 4.59) | label이 24축 전반에 분포 |
| Effective dims | 20.5 / 24 | 실질적으로 활성화된 축 수 |
| Top-1 mass | 0.141 | 가장 강한 단일 축의 집중도 |

→ Top-1 mass 14.1% — label이 flat하다. Effect 변화의 방향성이 소수 축에 집중되지 않고 전반에 퍼진다.

#### Effect별 Style F1 (linear probe: style label → effect 분류)

**Linear probe 방법론:**
- **입력**: training DB의 style_label (24-dim)
- **레이블**: 해당 레코드 생성 시 이 effect가 활성화됐는가? (effect별 binary)
- **모델**: Logistic Regression (단일 선형 레이어) — sklearn `LogisticRegression`
- **분할**: training DB에서 80/20 train/test split
- **지표**: F1 score (class imbalance 때문에 accuracy 대신)

**Linear probe를 쓰는 이유:**
style label의 **기하학적 구조**를 진단하기 위해서다.
"이 effect가 활성화됐을 때의 style label"이 "비활성화됐을 때"와 선형적으로 분리되는가?
비선형 분류기를 쓰면 표현력의 한계가 가려지므로, 선형 분류기로 style label 공간의 선형 분리 가능성만 측정한다.
F1이 낮으면 → 해당 effect의 활성화 여부가 style label에 선형적으로 인코딩되어 있지 않다 → Controller도 style label만으로는 구분이 어렵다.

| Effect | F1 |
|--------|----|
| distortion | **0.742** |
| reverb | **0.659** |
| highpass | 0.548 |
| playback_rate | 0.521 |
| lowpass | 0.514 |
| delay | 0.275 |
| bitcrush | **0.163** |
| **평균** | **0.489** |

→ Effect에 따라 style label의 변별력이 크게 다르다.
delay, bitcrush는 현재 vocab으로 변별이 어렵다. 이는 CLIP/CLAP의 학습 objective가 이런 low-level effect 변화를 text와 align하도록 설계된 게 아니기 때문에, 임베딩 공간 자체의 표현력 한계일 가능성이 높다.

> CLAP-side style label의 일관성 측정 (consistency separation = 0.038) → **Appendix G** 참조.

#### Calibration

Controller 학습 시 본 CLAP-side label 분포와, 추론 시 CLIP-side label 분포 간의 불일치를 완화.

**캘리브레이션 과정:**

**Stage 1 — Temperature & Norm Confidence 탐색:**
Places365 원본 이미지 750장에 7종 wand effect를 각각 적용한 image 쌍 5,250개(750×7)에 CLIP을 적용해 $\delta z$를 계산한 뒤, $(T, \tau_c, s_c)$ 조합을 grid search. CLIP-side top-1 mass의 기댓값이 학습 CLAP-side 평균 top-1 mass와 일치하도록 최적화:

$$\min_{T,\,\tau_c,\,s_c} \left| E_{\text{CLIP-side}}[\text{top-1 mass}] - E_{\text{CLAP-side}}[\text{top-1 mass}] \right|$$

→ $T^* = 0.015$, $\tau_c = 0.343$, $s_c = 0.206$

**Stage 2 — Activity Threshold:**
활성화 effect 수를 특정 목표에 맞추기 위해 단일 threshold override(0.66)를 탐색했으나, 최종적으로는 학습 시 effect별로 튜닝된 per-effect threshold를 그대로 사용하기로 결정 (`activity_threshold_override: null`). Effect 수를 인위적으로 제한하지 않고 controller 예측을 그대로 따른다.

**최종 보정식:**

$$s_{\text{final}} = c \cdot \text{softmax}\!\left(\frac{\delta z \cdot V}{T^*}\right) + (1-c) \cdot \frac{1}{24}\mathbf{1}$$

$$c = \sigma\!\left(\frac{\|\delta z\| - \tau_c}{\,s_c\,}\right), \quad \tau_c = 0.343,\; s_c = 0.206$$

→ $\|\delta z\|$이 작을 때 (CLIP이 변화를 약하게 감지) style label을 uniform에 가깝게 보정.

---

## Point 2. Controller를 어떻게 학습시킬 것인가?

---

### [Experiment 3] Controller는 무엇을 봐야 하는가? — Linear Probe

**방법론:**
- **Mean baseline:** 학습 데이터 전체의 파라미터 평균값을 입력과 무관하게 항상 예측. 어떤 표현도 사용하지 않는 trivial 기준선.
- **Linear probe:** 각 입력 표현 위에 단일 선형 레이어만 추가해 파라미터를 예측. 해당 표현이 파라미터 정보를 얼마나 선형적으로 담고 있는지를 가중치 없이 진단하는 경량 실험.

| 입력 조합 | Active Param RMSE |
|----------|-------------------|
| Mean baseline (항상 평균 예측) | 0.553 |
| CLAP embedding only | **0.553** |
| Style label only | **0.420** |
| CLAP + Style (gated) | **0.398** |

→ 핵심 발견: CLAP raw embedding은 mean baseline과 동일 — 독립적으로는 무의미한 신호. Style label이 실질적 정보를 담고 있음. → Controller 설계 근거: style label primary, CLAP은 gated residual로 (실질적 기여 최소화)

---

### 데이터셋 — Inverse Mapping DB

#### 개념

Controller의 학습 목표는 **style label이 주어졌을 때 어떤 effect가 어떤 파라미터로 그 label을 만들었는지를 역추론**하는 것이다.
이를 위한 데이터셋을 **Inverse Mapping Database**라 부른다.

직접적인 (image delta, audio delta) 쌍 데이터를 수집하는 것은 불가능하다. 대신, 오디오에 무작위 effect 조합을 적용하고 결과로 발생하는 CLAP delta를 style label로 변환하여 `(style_label → effect params)`의 역방향 매핑 레코드를 대량으로 생성한다.

#### 원본 데이터셋

| 도메인 | 출처 | 구성 |
|--------|------|------|
| **Audio** | GTZAN (blues/classical/country/disco/hiphop/jazz/metal/pop/reggae/rock) + Bandcamp CC (techno) | 4,174 클립 → train 1,570 / val 1,302 / test 1,302 |
| **Image** | MIT Places365 validation set (butte, cliff, coast, creek, desert_sand, glacier, hot_spring, ice_floe, iceberg, mountain_snowy, ocean, valley, volcano, waterfall, wheat_field) | 15개 자연 scene × 50장 = 750장 |

- DB 생성: train split 1,570개 오디오 클립 사용
- Calibration: 750장 이미지 × 7종 wand effect = **5,250 image pair** 사용

#### 생성 절차

```
# [Phase 0] 원본 오디오 split 구성 (build_audio_splits.py)
for each audio file in data/original/audio/ (4,174 클립):
    track_id ← 파일명 suffix 제거로 parent track 추론
               # e.g. "blues - blues 001 - 2.mp3" → "blues::blues_blues_001"

for each parent track_id (고유 트랙):
    해당 track의 클립 목록을 shuffle
    train : val : test = 8 : 1 : 1 비율로 분배
    # 동일 트랙의 클립이 각 split에 분산 → data leakage 방지

→ train 1,570 / val 1,302 / test 1,302

# [Phase A-2] Inverse Mapping DB 생성
for each audio clip A in train split (1,570개):
    for each augmentation (60회):
        1. effect 조합 무작위 샘플링 (1~2개, 7종 중)
        2. 각 effect의 파라미터 무작위 샘플링 (param_min_intensity=0.35 이상 강도 보장)
        3. Pedalboard로 A → A' 렌더링
        4. δ_clap = normalize(CLAP(A') - CLAP(A)) 계산
        5. style_label = softmax(δ_clap · V_aud / T)
        6. 레코드 저장: (style_label, CLAP(A), active_effects, params)
```

- 총 **94,080 records** (1,570 × 60)
- Train:Val = 80:20 (75,264 / 18,816)
- 균형 샘플러: effect별 역빈도 가중치로 불균형 보정

#### 이 DB가 타당한 이유

Controller는 학습 시 CLAP-side style label을 보고, 추론 시 CLIP-side style label을 본다.
Identity mapping assumption 하에, 두 분포는 동일하다고 가정되므로 이 DB는 추론 domain과 같은 분포를 갖는다고 볼 수 있다.
또한 effect를 무작위로 조합하여 생성된 다양한 (style_label, params) 쌍은,
"특정 방향의 style delta를 만들어내는 effect 조합"의 분포를 커버한다.

#### DB 생성 방식에서 비롯되는 Controller의 한계

1. **활성 effect 수의 implicit 제한:** 학습 데이터의 active effect 수가 1~2개로 제한되어 있어, 추론 시 그 이상의 동시 활성화를 잘 일반화하지 못할 수 있다.

2. **최소 강도 편향:** `param_min_intensity=0.35`는 각 파라미터를 자신의 range와 scale(linear/log)로 `[0,1]` 정규화한 normalized 공간에서 bypass까지의 거리가 0.35 이상이 되도록 샘플링한다. 즉 raw 단위의 스케일 차이는 정규화로 보정되어 있으며, 0.35는 전체 동적 범위 중 35% 이상 강도를 보장하는 기준이다. 그 결과 subtle한 effect(normalized distance < 0.35)는 학습 데이터에 거의 없고, Controller가 약한 강도의 effect를 과소 예측할 가능성이 있다.

3. **고정된 effect chain 순서:** DB 생성 시 Pedalboard effect는 항상 lowpass → bitcrush → reverb → highpass → distortion → delay → playback_rate 순서로 렌더링된다. Controller는 이 순서로만 만들어진 (style_label, params) 쌍으로 학습되므로, 다른 체인 순서에서 발생하는 음향적 상호작용을 한 번도 본 적이 없다. 이는 MLP가 순서를 처리하지 못해서가 아니라, DB 자체가 단일 순서로 고정되어 있어 학습 분포에 순서 다양성이 없기 때문이다.

4. **Style label의 변별력 한계 상속:** DB의 style label이 effect를 잘 구분하지 못하는 경우 (delay F1=0.275, bitcrush F1=0.163), Controller도 해당 effect를 정확하게 예측하기 어렵다.

---

### Controller 구조 (총 191K params)

```
Input: style_label (24-dim)
  └─ MLP backbone:
       Linear(24  → 256) → LayerNorm(256) → ReLU → Dropout(0.3)
       Linear(256 → 256) → LayerNorm(256) → ReLU → Dropout(0.3)
       Linear(256 → 128) → LayerNorm(128) → ReLU → Dropout(0.3)
                    ↓ style_feat (128-dim)
  [Gated Residual]
       gate       = σ( Linear(128→128)(style_feat) + bias=-2.0 )
       audio_feat = Linear(512→128) → LayerNorm(128) → ReLU → Dropout(0.3)  ← CLAP(A) (512-dim)
       feat_final = style_feat + gate × audio_feat    (기본 gate≈0.12 → CLAP 기여 억제)
  ├─ Activity head: Linear(128 → 7)             → logits  [추론 시 sigmoid 적용]
  └─ Param head:   Linear(128 → 12) + Sigmoid  → 12개 파라미터 [0,1] 예측
```

**CLAP(A)의 역할:** CLAP(A)는 입력 오디오 A의 스펙트럼·음색 특성을 인코딩한 벡터다. Style label이 "어떤 방향으로 변화할지"를 결정하는 1차 신호라면, CLAP(A)는 현재 오디오의 특성에 따라 그 방향을 미세 조정하는 2차 신호다. Probe 결과(CLAP alone=0.553, CLAP+Style=0.398 vs Style alone=0.420)에서 CLAP이 단독으로는 정보가 없지만, style label과 결합 시 소폭(+5.2%) 기여함을 확인. `gate_bias=-2.0`으로 기본적으로 닫혀 있어 학습 초기 안정성을 보장하며, probe 결과와 일치하는 inductive bias를 부여한다.

---

### Loss Function

#### Activity Loss — ASL (Asymmetric Sigmoid Loss)

$$L_{\text{ASL}}(y_i, p_i) = \begin{cases} (1-p_i)^{\gamma_+} \log(p_i) & \text{if } y_i = 1 \\ (p_i^-)^{\gamma_-} \log(1 - p_i^-) & \text{if } y_i = 0 \end{cases}$$

$$p_i^- = \max(p_i - m,\; 0), \quad \gamma_+ = 0,\; \gamma_- = 5,\; m = 0.05$$

- **$\gamma_- = 5$:** 쉬운 negative sample의 gradient를 억제하여, 실제 비활성 effect를 높은 확률로 예측하는 어려운 케이스(false positive)에 집중 — 없는데 있다고 강하게 확신하는 경우에 더 강한 페널티
- **$m = 0.05$ (margin clip):** 낮은 확률의 negative sample에서 gradient를 0으로 만들어 쉬운 negative에 과도하게 집중하지 않도록

#### Param Loss — Huber Loss

$$L_{\text{Huber}}(\hat{p}, p; \delta) = \begin{cases} \dfrac{(\hat{p} - p)^2}{2\delta} & \text{if } |\hat{p} - p| \leq \delta \\[6pt] |\hat{p} - p| - \dfrac{\delta}{2} & \text{otherwise} \end{cases}, \quad \delta = 0.02$$

- Outlier에 robust하면서도 작은 오차에서 quadratic gradient를 유지
- Effect별 가중치 $w_e$ 적용:

| Effect | $w_e$ |
|--------|-------|
| delay | 3.0 |
| bitcrush | 2.5 |
| lowpass | 2.0 |
| highpass | 2.0 |
| reverb | 1.5 |
| playback_rate | 1.5 |
| distortion | 1.0 |

#### Activity Mismatch Multiplier

Mismatch penalty는 별도 항이 아니라 **Activity Loss 내부에 multiplier로 포함**된다.
ASL 각 element에 예측 불확실성 기반 가중치를 곱한다:

$$L_{\text{act}} = \text{mean}\!\left(L_{\text{ASL,elem}} \cdot \left(1 + w_{\text{mm}} \cdot (1 - p_t)^{\gamma_{\text{mm}}}\right)\right)$$

$$p_t = y \cdot p + (1-y) \cdot (1-p), \quad w_{\text{mm}} = 2.0,\; \gamma_{\text{mm}} = 2.0$$

→ $p_t$가 낮을수록 (예측이 틀렸거나 불확실할수록) multiplier가 커져 해당 element에 더 강한 페널티.

#### Total Loss

$$L_{\text{total}} = w_{\text{param}} \cdot L_{\text{param}} + w_{\text{act}} \cdot L_{\text{act}}$$

$$w_{\text{act}} = 0.6,\quad w_{\text{param}} = 1.0$$

---

### Training

- 600 epochs, cosine LR decay ($10^{-4} \to 10^{-6}$), batch=256, weight decay=$10^{-3}$
- Best epoch: 527 / 600 (selection metric: `val_active_param_rmse_gated`)

---

### Post-Training: Activity Threshold Tuning

학습 완료 후, activity head의 per-effect threshold를 validation set에서 별도 최적화한다.

**목적:** 기본 threshold=0.5는 effect별 활성화 빈도 불균형을 고려하지 않는다.
단순 F1 최대화가 아닌, threshold가 실제 파라미터 예측 품질에 미치는 영향을 직접 목적함수로 삼는다.

**과정:**
1. Validation set 전체 inference → activity logits + params_pred 수집
2. $p = \sigma(\text{logits})$
3. 탐색 공간: 0.05 ~ 0.95 균등 분할 **37개 grid**
4. **목적함수: `active_param_rmse_gated`**
   - 각 threshold 후보에서 predicted-inactive effect의 params를 bypass 값으로 gate
   - gate 적용 후 ground-truth active params에 대해서만 RMSE 계산
5. **Coordinate descent** (2 passes): 7개 effect를 순서대로 하나씩 고정하고 나머지 탐색, 2회 반복

→ effect별로 `active_param_rmse_gated`를 최소화하는 threshold가 결정됨.
이 per-effect threshold가 추론 시 `activity_threshold_override: null` 설정으로 그대로 적용된다.

---

## Results — Quantitative

**Primary metric: Active Param RMSE (gated)**
활성화로 예측된 파라미터만 대상으로, activity 예측 오류를 RMSE에 반영한 지표.

| | Active Param RMSE (gated) |
|--|--------------------------|
| Mean baseline | 0.553 |
| Style-only probe | 0.420 |
| **Final controller** | **0.177** |

→ Mean baseline 대비 68% 감소, style-only probe 대비 58% 감소.

**Activity F1:**

| Macro F1 | Micro F1 |
|----------|----------|
| 0.564 | 0.542 |

**파라미터별 Active RMSE (상위 5, 어려운 순):**

| Parameter | Active RMSE |
|-----------|------------|
| reverb.damping | 0.222 |
| delay.mix | 0.216 |
| reverb.wet_level | 0.206 |
| reverb.room_size | 0.188 |
| delay.delay_seconds | 0.181 |

→ reverb (3 params), delay (3 params) 계열이 가장 높다.
reverb의 경우 style F1이 0.659로 높음에도 파라미터 RMSE가 높은 것은, effect 자체를 구분하는 것과 그 파라미터 값을 정밀하게 예측하는 것이 별개의 난이도임을 시사한다.

---

## Results — Qualitative (Web Demo)

### Objective 기술 지표

출력의 "좋음"을 평가하는 게 아니라, 시스템의 내부 상태를 기술하는 지표:

| 지표 | 의미 |
|------|------|
| $\|\delta z\|$ (δz norm) | CLIP이 image 변화를 얼마나 강하게 감지했는가 |
| Style label entropy | 예측 방향이 얼마나 집중되어 있는가 (낮을수록 명확) |
| Max activity probability | Controller가 가장 강하게 예측한 effect의 확신도 |

### 사용자 평가 설계

**자극 구성 (case당):**
- Original image / Edited image (나란히 제시)
- Original audio / Processed audio (각각 재생 가능)

**평가 문항:**

**Q1. 방향 일치성 (5점 척도)**
> "원본→수정 이미지의 변화 방향과, 원본→처리 오디오의 변화 방향이 같은 방향을 가리킨다고 느끼는가?"
> 1 (전혀 아님) — 2 — 3 (중립) — 4 — 5 (매우 그렇다)

**Q2. 귀속 판단 (forced choice)**
> "처리된 오디오는 아래 중 어느 이미지에 더 어울리는가?"
> A (원본) / B (수정본)

→ B 선택 비율이 50%를 유의미하게 상회하면, 시스템이 perceptual하게 일관된 변화를 만들어냈다고 볼 수 있다.

**Case 선정 기준 (8~10개):**

| 그룹 | 기준 | 수량 |
|------|------|------|
| High δz norm | CLIP이 강하게 감지한 case | 4 |
| Low δz norm | CLIP이 약하게 감지한 case | 2 |
| N_active = 1 | 단일 effect 예측 (명확한 case) | 3 |
| N_active = 2 | 복합 effect 예측 | 3 |

→ "δz norm이 높을수록 Q1/Q2 점수가 높은가?"라는 correlation 분석도 가능.

**평가자:** 5~10명 (발표 청중 중 지원자)

---

## Limitations & Future Work

### Limitations

**1. Identity Mapping Assumption의 취약성**
Style label consistency separation (CLAP-side intra-modal) = 0.038.
같은 effect를 적용한 audio sample 쌍이 무작위 쌍보다 유사한 style label을 형성한다는 근거가 약하다. image↔audio cross-modal alignment는 별도로 측정되지 않으며, identity mapping assumption의 직접적인 실험적 뒷받침은 없다.

**2. Style Vocab의 낮은 변별력 (effect별 편차)**
bitcrush F1=0.163, delay F1=0.275.
CLIP/CLAP의 학습 objective가 low-level effect 변화를 text와 align하도록 설계된 게 아니기 때문에, 임베딩 공간 자체의 표현력 한계일 가능성이 높다. vocab 설계 개선만으로는 해결에 한계가 있다.

**3. DB 생성 방식에서 비롯된 한계 (Controller)**
- Active effect 수 1~2개로 implicit하게 제한
- 최소 강도 편향 (param_min_intensity=0.35)
- 고정된 effect chain 순서

**4. Style Label Domain Shift (CLAP → CLIP)**
Controller는 CLAP-side style label로 학습하지만, 추론 시 CLIP-side label이 입력된다.
Calibration으로 완화하지만 근본적인 분포 불일치는 남는다.

**5. MLP가 학습하지 못하는 것**
Effect 파라미터 간 상호 의존성 (e.g. reverb.wet_level ↔ reverb.room_size), 순서 의존적 음향 특성, 활성 effect 수의 유연한 변화 등은 현재 구조에서 표현되지 않는다.

### Future Work

- **Style Vocab Ensemble:** 현재는 단일 vocab 집합. 여러 vocab 버전을 앙상블하면 label 안정성 개선 가능
- **Autoregressive Controller:** Effect를 순차적으로 예측하여 순서 dependency를 학습하는 구조 (본 연구에서 구현됨, Appendix 참조)
- **Cross-modal Alignment 학습:** Identity mapping 대신 실제 alignment를 학습하는 별도 모듈

---

---

# Appendix

---

## A. Style Vocabulary 버전별 비교

| 버전 | 특징 | 문제점 |
|------|------|--------|
| v1 | 직관적 단어 위주 | "vintage" 등 특정 단어가 generic anchor로 작동, 변별력 낮음 |
| v2 | 보다 구조화된 축 도입 | 단어가 특정 모달에 편향되거나 절대적 semantic region을 가리키는 경우 존재 |
| v3 (현재) | ΔE-direction + Cross-modal neutrality + Anti-anchor 원칙 적용 | Style F1 평균 0.423 → 0.489 (+15.7%), cross-modal consistency는 소폭 감소 |

---

## B. Attempt별 학습 결과 비교

| Attempt | Vocab | Epochs | Best Epoch | Active RMSE (gated) | Style F1 평균 | Activity Macro F1 |
|---------|-------|--------|-----------|---------------------|--------------|------------------|
| attempt12 | v2 | 400 | — | **0.160** | 0.423 | — |
| attempt13 | v3 | 400 | 386/400 | 0.178 | 0.489 | 0.550 |
| attempt14 | v3 | 600 | 527/600 | 0.177 | 0.489 | **0.564** |

**관찰:**
- v3 vocab 전환으로 style F1이 개선됐지만 RMSE는 소폭 악화됨
- attempt14(600ep)는 attempt13(400ep) 대비 RMSE 0.001, Activity F1 +0.014 개선 — 600ep에서도 여전히 수렴 중이었음을 best_epoch=527이 시사

---

## C. 최종 Hyperparameter Config (attempt14)

### CLAP / CLIP
| 항목 | 값 |
|------|---|
| CLIP model | ViT-L-14 (OpenAI) |
| CLAP model_id | 1 (LAION-CLAP) |
| CLAP max_duration | 20.0s |
| Sample rate | 48,000 Hz |

### Inverse Mapping DB 생성
| 항목 | 값 |
|------|---|
| 오디오 파일 수 | 1,570 |
| Augmentations per audio | 60 |
| Min active effects | 1 |
| Max active effects | 2 |
| param_min_intensity | 0.35 |
| param_mid_bypass_exclusion | 0.15 |
| delta_min_norm | 0.06 |
| delta_resample_attempts | 3 |
| Temperature (label 생성) | 0.1 |

### Controller 학습
| 항목 | 값 |
|------|---|
| Hidden dims | [256, 256, 128] |
| Dropout | 0.3 |
| Audio embed dim (gated) | 512 |
| Gate bias | -2.0 |
| Learning rate | 1e-4 |
| LR scheduler | cosine (min=1e-6) |
| Batch size | 256 |
| Epochs | 600 |
| Weight decay | 1e-3 |
| Activity loss | ASL (γ_pos=0, γ_neg=5, clip=0.05) |
| Param loss | Huber (δ=0.02) |
| w_act / w_param | 0.6 / 1.0 |
| Mismatch weight / gamma | 2.0 / 2.0 |
| Effect loss weights | delay×3.0, bitcrush×2.5, lowpass×2.0, highpass×2.0, reverb×1.5, playback_rate×1.5, distortion×1.0 |
| Selection metric | val_active_param_rmse_gated |

### Inference
| 항목 | 값 |
|------|---|
| Style temperature T* | 0.015 |
| Norm confidence threshold τ_c | 0.343 |
| Norm confidence scale s_c | 0.206 |
| Activity threshold override | null (학습 시 per-effect threshold 사용) |
| Top-k | 5 (UI 표시용 — 상위 5개 vocab term을 웹 데모에 노출, 추론 로직에 무관) |

---

## D. Calibration 상세

**목적:** Controller 학습 시 본 CLAP-side style label의 분포와, 추론 시 생성되는 CLIP-side style label의 분포 간 불일치를 완화.

**문제:** 동일한 vocab에 대해 CLIP과 CLAP이 형성하는 임베딩 공간의 geometry가 다르다 (img_vocab 쌍별 유사도=0.92 vs aud_vocab=0.78). 이로 인해 CLIP-side softmax 출력이 더 broad해지는 경향.

**보정 방법:**

1. **Temperature scaling:** $T^* = 0.015$
   `scripts/calibrate_inference.py`로 5,250개의 wand-effect image 쌍에 대해 최적화.
   목표: `--target-top1 = 0.1412` (학습 CLAP-side 평균 top-1 mass, `calibrate_inference.py` 기본값으로 고정).

2. **Norm-based confidence mixing:**
   $\|\delta z\|$이 작으면 (CLIP이 변화를 약하게 감지) style label을 uniform에 가깝게 보정.
   $\tau_c, s_c$도 T*와 jointly calibrate.

3. **Activity threshold: per-effect threshold 사용 (`activity_threshold_override: null`)**
   단일 override threshold(0.66)를 탐색했으나, effect 수를 인위적으로 제한하지 않기 위해 학습 시 effect별로 튜닝된 threshold를 그대로 사용.

---

## E. Autoregressive (AR) Controller 구현

학습 데이터의 고정된 effect chain 순서와 MLP의 순서 무관 예측을 개선하기 위해, effect를 순차적으로 예측하는 AR 구조를 구현.

**구조:**
```
Step 0: condition = f(style_label, CLAP(A))   (condition_dim=256)
Step t: (이전 step에서 예측한 effect_id, params) → update condition
        → predict next effect (CE loss) + params (Huber loss)
        → max_steps=2에서 종료 또는 EOS 예측 시 종료
```

| 항목 | 값 |
|------|---|
| condition_dim | 256 |
| hidden_dim | 512 |
| dropout | 0.1 |
| max_steps | 2 |
| clap_embed_dim | 512 |
| effect CE label smoothing | 0.03 |
| param_loss_weight | 50.0 (CE ~0.83 vs Huber ~0.003 스케일 보정) |

**현재 상태:** 구현 완료, 학습 미완 (best_epoch = None). 향후 완성 시 MLP controller와 비교 가능.

---

## F. Effect 및 파라미터 전체 목록

| Effect | Parameters | 파라미터 수 |
|--------|-----------|-----------|
| lowpass | cutoff_hz | 1 |
| bitcrush | bit_depth | 1 |
| reverb | room_size, damping, wet_level, dry_level | 4 |
| highpass | cutoff_hz | 1 |
| distortion | drive_db | 1 |
| delay | delay_seconds, feedback, mix | 3 |
| playback_rate | rate | 1 |
| **합계** | | **12** |

> Note: 실제 코드에서 playback_rate는 Pedalboard 외부에서 별도 처리하며, 항상 chain의 마지막에 적용.

---

## G. Style Label 내부 일관성 (Consistency Separation)

**측정 과정:** CLAP-side training DB 내에서, 같은 effect 조합을 적용한 sample 쌍을 bootstrap으로 샘플링하고 각 쌍의 style label 간 cosine similarity를 계산. 이를 무작위 쌍의 cosine similarity 분포와 비교.

$$\text{separation} = \overline{\cos(s_i, s_j)}_{\text{same-effect}} - \overline{\cos(s_i, s_j)}_{\text{random}}$$

- Consistency separation = **0.038**
- 0.038은 낮은 수치 — style vocab이 같은 effect를 일관된 label 방향으로 매핑하지 못하고 있음을 시사. Effect별 style F1과 같은 맥락의 지표.

> 이 측정은 전적으로 CLAP-side (audio delta) 내에서만 이루어진다. CLIP-side (image delta)와 CLAP-side 간의 cross-modal alignment — identity mapping assumption의 직접적 검증 — 은 본 프레임워크에서 별도로 측정되지 않는다.

**더 직접적인 진단 방향 (미수행):** 동일한 linear probe를 CLIP-side에서 수행 — wand-effect image 쌍에서 style label을 계산한 뒤 image effect 분류. 이 경우 실제 추론 경로의 입력 측 변별력을 직접 확인할 수 있다.

---

*작성일: 2026-02-26*
