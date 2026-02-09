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

### 4. `fit_alignment` - Phase 2-a: CCA 정렬 학습

**입력:**
- `data/images/**/*.{jpg,png}` - 원본 이미지 (증강 X)
- `data/audio/**/*.{wav,mp3,flac}` - 원본 오디오 (증강 X)

**처리:**
1. 원본 이미지 임베딩 추출: `(N, 768)` CLIP
2. 원본 오디오 임베딩 추출: `(N, 512)` CLAP
3. CCA(Canonical Correlation Analysis) 학습:
   - 목표: CLIP(768d) ↔ CLAP(512d) 공간 정렬
   - 방법: 512 components로 projection 학습
   - 결과: `CCA.transform_image()`, `CCA.transform_audio()`

**출력:**
- `outputs/cca_alignment.pkl` - CCA 모델 (pickle)
  - 메서드: `transform_image(X_768) → X_512`
  - 메서드: `transform_audio(X_512) → X_512`
- **의미:** 교차모달 유사도 계산을 위한 임베딩 공간 정렬

---

### 5. `phase1` - Phase 3-a: 프로토타입 발견

**입력:**
- `outputs/image_deltas.json`
- `outputs/audio_deltas.json`
- `outputs/cca_alignment.pkl`

**처리:**
1. **프로토타입 계산:**
   - 각 (modality, effect_type, intensity)에 대해:
   - `p(t,s) = E[Δe / ||Δe||]` (정규화된 델타의 평균)

2. **교차모달 유사도 행렬:**
   - 이미지 프로토타입: CCA로 정렬 → `(N_img_effects, 512)`
   - 오디오 프로토타입: CCA로 정렬 → `(N_aud_effects, 512)`
   - 유사도 계산: `cosine_similarity(p_img, p_aud)`

3. **시각화:**
   - 히트맵 생성: 이미지 이펙트 × 오디오 이펙트

**출력:**
- `outputs/image_prototypes.npy` - 이미지 프로토타입 벡터들
- `outputs/audio_prototypes.npy` - 오디오 프로토타입 벡터들
- `outputs/similarity_matrix.npy` - 교차모달 유사도 행렬
- `outputs/similarity_heatmap.png` - 히트맵 이미지
- **의미:** 어떤 이미지 이펙트가 어떤 오디오 이펙트와 대응되는지 발견

---

### 6. `phase2` - Phase 3-b: 통계적 검증

**입력:**
- `outputs/image_deltas.json`
- `outputs/audio_deltas.json`
- `outputs/cca_alignment.pkl`

**처리:**

#### 6-1. 검색 기반 검증 (Retrieval)
1. 이미지 델타를 쿼리로 사용
2. 오디오 델타 DB에서 가장 유사한 것 검색
3. 메트릭 계산:
   - **MRR (Mean Reciprocal Rank):** 정답의 평균 역순위
   - **Recall@K:** Top-K 내 정답 비율
4. 강도별로 집계

#### 6-2. 순열 검정 (Permutation Test)
1. 실제 MRR 계산
2. 라벨을 무작위로 섞어서 MRR 재계산 (n=1000회)
3. p-value 계산: `P(random_MRR >= actual_MRR)`
4. 유의성 판정: `p < 0.05`

#### 6-3. Spearman 상관계수
1. 강도(low=1, mid=2, high=3)와 MRR 간 상관계수
2. 가설: 강도가 높을수록 MRR도 높아야 함
3. 계산: `spearman(intensity_rank, MRR)`

#### 6-4. 노름 분석
1. 각 이펙트의 델타 노름과 강도 간 상관계수
2. 가설: `||Δe|| ∝ intensity`
3. 계산: `spearman(intensity, ||Δe||)`

**출력:**
- `outputs/retrieval_results.json`
  - 이펙트별, 강도별 MRR/Recall@K
- `outputs/permutation_test_results.json`
  - p-value, 유의성 판정
- `outputs/spearman_results.json`
  - 강도-MRR 상관계수
- `outputs/norm_analysis.json`
  - 강도-노름 상관계수
- `outputs/retrieval_by_effect.png` - 이펙트별 검색 성능 그래프
- **의미:** 발견된 대응이 통계적으로 유의미한지 검증

---

### 7. `all` - 전체 파이프라인

**처리 순서:**
```
extract → sensitivity → linearity → fit_alignment → phase1 → phase2
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
    └─→ [fit_alignment] → cca_alignment.pkl (Phase 2-a)
            ↓
            ├─→ [phase1] → prototypes, similarity_matrix, heatmap (Phase 3-a)
            │
            └─→ [phase2] → retrieval, permutation, spearman, norm (Phase 3-b)
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
- **CCA 정렬 후:** 모두 512-dim

### 주요 Thresholds
- **Sensitivity:** `min_distance = 0.01`
- **Linearity:** `min_cosine = 0.8`, `max_cv = 0.3`
- **Cross-category:** `max_variance = 0.05`
- **Retrieval:** `n_permutations = 1000`, `p_threshold = 0.05`
