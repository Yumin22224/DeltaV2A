# Delta V2A 실험 파이프라인 데이터 흐름

## 명령어별 입력/출력 및 처리 과정

### 1. `extract` - 델타 임베딩 추출

**입력:**
- `data/images/**/*.{jpg,png}` - 원본 이미지 파일들
- `data/audio/**/*.{wav,mp3,flac}` - 원본 오디오 파일들
- `configs/experiment.yaml` - 설정 파일

**처리:**
1. CLIP (ViT-L/14) 로드 → 이미지 768차원 임베딩
2. CLAP (music, 630k+audioset) 로드 → 오디오 512차원 임베딩
3. 각 파일에 대해:
   - 원본 임베딩 추출: `e_orig`
   - 각 이펙트 × 강도 조합에 대해:
     - 이펙트 적용: `augmented = effect(original, intensity)`
     - 증강 임베딩 추출: `e_aug`
     - 델타 계산: `Δe = e_aug - e_orig`

**출력:**
- `outputs/image_deltas.json` - 이미지 델타 데이터셋
  - 구조: `[{modality, effect_type, intensity, category, delta, original_embedding, augmented_embedding}, ...]`
  - 개수: N_images × N_effects × N_intensities
  - 예: 100 images × 4 effects × 3 intensities = 1200 deltas
- `outputs/audio_deltas.json` - 오디오 델타 데이터셋
  - 구조: 동일
  - 개수: N_audio × N_effects × N_intensities

---

### 2. `sensitivity` - Phase 0-a: 민감도 검사

**입력:**
- `outputs/image_deltas.json`
- `outputs/audio_deltas.json`
- Threshold: `min_distance` (기본 0.01)

**처리:**
1. 각 (modality, effect_type, intensity) 조합에 대해:
   - 모든 델타의 노름 계산: `||Δe||`
   - 평균/표준편차/최소/최대 계산
   - 민감도 판정: `mean(||Δe||) > threshold`

**출력:**
- `outputs/sensitivity_results.json`
  - 각 조합별 통계량 및 민감도 판정
  - 콘솔: 민감하지 않은 이펙트 경고
- **의미:** 모델이 해당 이펙트를 감지할 수 있는지 확인

---

### 3. `linearity` - Phase 0-b: 선형성/일관성 검사

**입력:**
- `outputs/image_deltas.json`
- `outputs/audio_deltas.json`
- Thresholds:
  - `min_cosine`: 0.8 (방향 일관성)
  - `max_cv`: 0.3 (크기 일관성)
  - `max_variance`: 0.05 (카테고리 간 분산)

**처리:**
1. **카테고리 내 일관성:**
   - 같은 카테고리 내에서 동일 이펙트의 델타들 간 코사인 유사도 계산
   - 크기(노름) 변동계수(CV) 계산
   - 판정: `mean_cosine > 0.8 AND cv < 0.3`

2. **카테고리 간 불변성 (Context Invariance):**
   - 다른 카테고리에서 동일 이펙트의 델타 방향 비교
   - 카테고리 간 코사인 유사도 분산 계산
   - 판정: `variance < 0.05`

**출력:**
- `outputs/linearity_results.json`
  - 카테고리별 일관성 점수
  - 카테고리 간 분산 점수
- `outputs/cross_category_variance.json`
  - 각 이펙트의 컨텍스트 불변성 검증
- **의미:** 델타가 선형적이고, 소스 카테고리와 무관하게 일관된 방향을 가지는지 확인



---

### 4. `phase1` - Phase 1: 텍스트 앵커 기반 발견

**입력:**
- `outputs/image_deltas.json`
- `outputs/audio_deltas.json`
- CLIP/CLAP 모델 (텍스트 임베딩용)

**처리:**
1. **프로토타입 계산:**
   - 각 (modality, effect_type, intensity)에 대해:
   - `p(t,s) = E[Δe / ||Δe||]` (정규화된 델타의 평균)

2. **텍스트 앵커 생성 (Synonym Expansion):**
   - **이미지 텍스트 앵커:**
     - 각 이펙트별 동의어 세트 정의 (예: blur → ["blurry", "out of focus", "fuzzy", ...])
     - 템플릿 적용: "A {synonym} photo of {category}"
     - 델타 계산: embed("A blurry photo of ocean") - embed("A photo of ocean")
     - 모든 동의어 × 카테고리 조합에 대해 평균 → 텍스트 앵커 델타
   - **오디오 텍스트 앵커:**
     - 동일 방식으로 "A {synonym} {genre} song" 템플릿 사용

3. **3방향 유사도 계산 (Multiplication):**
   - 각 (이미지 이펙트, 오디오 이펙트) 쌍에 대해:
     - `sim1 = cosine(image_prototype, image_text_anchor)`
     - `sim2 = cosine(audio_prototype, audio_text_anchor)`
     - `sim3 = cosine(image_text_anchor, audio_text_anchor)`
     - `final_score = sim1 × sim2 × sim3`  (곱셈 = AND 로직)

4. **Discovery Matrix 생성:**
   - 행렬 크기: (이미지 이펙트 수) × (오디오 이펙트 수)
   - 각 셀에 3방향 유사도 점수

5. **시각화:**
   - 히트맵 생성: 이미지 이펙트 × 오디오 이펙트

**출력:**
- `outputs/discovery_matrix.npy` - 3방향 유사도 행렬
- `outputs/discovery_labels.json` - 이펙트 레이블
- `outputs/discovery_heatmap.png` - 히트맵 이미지
- **의미:** 어떤 이미지 이펙트가 어떤 오디오 이펙트와 대응되는지 발견 (텍스트 기반 검증 포함)

---

### 5. `phase3` - Phase 3: 학습 (Learning - The Decoder)

**입력:**
- `outputs/discovery_matrix.npy` - Phase 1에서 발견한 이미지-오디오 이펙트 대응 관계
- `outputs/discovery_labels.json` - 이펙트 레이블
- `data/audio/**/*.{wav,mp3,flac}` - 원본 오디오 파일들
- Discovery Matrix에서 높은 점수를 받은 대응 쌍 (예: blur ↔ lpf, brightness ↔ highshelf)

**처리:**
1. **학습 데이터셋 생성:**
   - 원본 오디오 $A_{raw}$ 로드
   - 각 샘플에 대해:
     - 랜덤하게 DSP 이펙트 선택 (Discovery Matrix 기반)
     - 랜덤 파라미터 $P$ 생성 (예: cutoff_freq, gain, decay 등)
     - 증강 오디오 생성: $A_{aug} = \text{DSP}(A_{raw}, P)$
     - 텍스트 조건 생성: 대응하는 이미지 이펙트 설명 $E_{cond}$

2. **Decoder 아키텍처:**
   - **입력:**
     - 원본 오디오 임베딩: $\text{CLAP}(A_{raw})$
     - 텍스트 조건 임베딩: $\text{CLAP}_{\text{text}}(E_{cond})$
   - **출력:**
     - DSP 파라미터 예측: $\hat{P}$
   - **구조:**
     - Cross-attention을 통해 텍스트 조건을 오디오에 반영
     - MLP로 DSP 파라미터 회귀

3. **학습:**
   - **Loss:** $\mathcal{L} = \text{MSE}(\hat{P}, P)$ (파라미터 회귀)
   - **Optimizer:** AdamW
   - **데이터:** 원본 오디오 × 랜덤 DSP 조합
   - **Validation:** 예측 파라미터로 증강 후 스펙트로그램 비교

**출력:**
- `outputs/decoder_model.pt` - 학습된 Decoder 모델
- `outputs/training_log.json` - 학습 로그 (loss, validation metrics)
- `outputs/parameter_predictions/` - 검증 샘플의 예측 파라미터
- **의미:** 이미지 이펙트 설명($E_{cond}$)을 받아서 대응하는 오디오 DSP 파라미터를 예측할 수 있는 모델

---

### 6. `all` - 전체 파이프라인

**처리 순서:**
```
extract → sensitivity → linearity → phase1 → phase3
```

**최종 출력:**
- 모든 중간 결과물
- 콘솔에 전체 실험 요약 출력
- 실패한 단계가 있으면 경고 메시지

---

## 데이터 흐름 다이어그램

```
[원본 데이터]
    ↓
[extract] → image_deltas.json, audio_deltas.json
    ↓
    ├─→ [sensitivity] → sensitivity_results.json (Phase 0-a)
    │
    ├─→ [linearity] → linearity_results.json (Phase 0-b)
    │
    └─→ [phase1] → discovery_matrix, discovery_heatmap (Phase 1: Text Anchor Discovery)
            ↑
            └── CLIP/CLAP 텍스트 임베딩 사용
            ↓
        [phase3] → decoder_model.pt (Phase 3: Learning - The Decoder)
            ↑
            ├── discovery_matrix (대응 관계)
            ├── 원본 오디오 (data/audio/)
            └── 랜덤 DSP 증강
```

---

## 주요 파라미터

### 이펙트 목록
- **이미지:** `blur`, `brightness`, `saturation`, `contrast`
- **오디오:** `lpf`, `highshelf`, `saturation`, `reverb`
- **강도:** `low`, `mid`, `high`

### 임베딩 차원
- **CLIP (이미지):** 768-dim
- **CLAP (오디오):** 512-dim
- **텍스트 임베딩:** CLIP 768-dim (이미지), CLAP 512-dim (오디오)

### 주요 Thresholds
- **Sensitivity:** `min_distance = 0.01`
- **Linearity:** `min_cosine = 0.8`, `max_cv = 0.3`
- **Cross-category:** `max_variance = 0.05`
- **Discovery:** 3-way similarity 기반 (multiplication)

### Phase 3 하이퍼파라미터 (예시)
- **Batch size:** 32
- **Learning rate:** 1e-4
- **Optimizer:** AdamW
- **Loss:** MSE (parameter regression)
- **Epochs:** 100
- **Validation split:** 20%
