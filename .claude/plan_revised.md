# Delta Correspondence Experiment: Implementation Plan (Phase 0-2)

## 0. 프로젝트 개요 (Overview)

**목표**: 시각적 이펙트(ΔV)와 청각적 이펙트(ΔA) 사이의 의미론적 대응(Semantic Correspondence)을 규명한다.

**핵심 접근**:
- **Model**: CLIP (ViT-L/14, 768d) + LAION-CLAP (Music, 512d)
- **Method**: 통계적 검증(Phase 0-2) → CCA 정렬 → 대응 관계 발견
- **가설**: Image Effect와 Audio Effect 사이에 Semantic Correspondence가 존재한다.

**전제 가정**:
- A-1: CLIP/CLAP 각 모달에서 이펙트가 적용된 대상의 컨텍스트와 관계없이, 이펙트가 동일 종류일 경우 동일한 방향성을 가진다 (Context Invariant, Linearity)
- B: Semantic Correspondence 자체는 전역적으로 존재하지만, 구체적 물리적 수치(Audio Parameter)로 실현하는 과정은 초기 오디오 상태에 종속적(Conditional)이다.

---

## Phase 0: 민감도 및 일관성 진단 (Diagnosis)

> **목적**: CLIP/CLAP이 이펙트를 감지하는지, 컨텍스트 불변성을 가지는지 검증

### Phase 0-a: Sensitivity Check

**질문**: CLIP, CLAP 모델이 이펙트를 감지하는가?

**방법**:
```python
# src/experiment/sensitivity.py
@dataclass
class SensitivityResult:
    modality: str  # "image" or "audio"
    effect_type: str
    intensity: str
    mean_distance: float  # E[||e(x_eff) - e(x_0)||]
    std_distance: float
    num_samples: int
    is_sensitive: bool  # mean_distance > threshold

def sensitivity_check(
    delta_dataset: DeltaDataset,
    threshold: float = 0.01,
) -> List[SensitivityResult]:
    """
    각 (modality, effect_type, intensity)에 대해:
    - Distance(e(x_0), e(x_eff)) 평균 및 표준편차 계산
    - threshold와 비교하여 민감도 판정

    Returns:
        민감도가 낮은(is_sensitive=False) 이펙트는 제외 대상
    """
```

**Action**: 변화가 너무 미미한 이펙트는 실험에서 제외

**Integration**: `scripts/run_experiment.py` → `sensitivity` 커맨드

---

### Phase 0-b: Consistency (Variance) Check

**질문**: 컨텍스트(카테고리)가 달라져도 Δ 방향이 일관한가?

**방법**:
```python
# src/experiment/linearity.py (기존 계획과 통합)
@dataclass
class LinearityResult:
    modality: str
    effect_type: str
    intensity: str
    category: str  # e.g., "ocean", "mountain"

    # Delta 방향 일관성
    mean_pairwise_cosine: float  # 정규화된 델타 간 코사인 유사도 평균
    std_pairwise_cosine: float
    min_cosine: float
    max_cosine: float

    # Delta 크기 일관성
    mean_norm: float
    cv_norm: float  # Coefficient of Variation (std/mean)

    num_samples: int
    is_consistent: bool  # mean_pairwise_cosine > threshold (e.g., 0.8)

def linearity_analysis(
    delta_dataset: DeltaDataset,
    category_mapping: Dict[str, str],  # path -> category (from folder structure)
    cosine_threshold: float = 0.8,
) -> List[LinearityResult]:
    """
    각 (modality, effect_type, intensity, category)에 대해:
    1. 같은 카테고리 내 델타들의 pairwise cosine similarity 계산
    2. 델타 norm의 CV 계산

    High cosine sim + Low CV → 선형성/일관성 확보
    """

def cross_category_variance_check(
    results: List[LinearityResult],
) -> Dict[Tuple[str, str, str], float]:
    """
    각 (modality, effect_type, intensity)에 대해:
    - 카테고리 간 mean_pairwise_cosine의 분산 계산
    - 분산이 크면 → 컨텍스트 의존적 (일관성 없음)

    Returns:
        {(modality, effect_type, intensity): cross_category_variance}
    """
```

**Scope Adjustment**:
- 특정 컨텍스트(카테고리)에서 분산이 크다면:
  - Option 1: 해당 컨텍스트 제거 (Scope 축소)
  - Option 2: Phase 3에서 비선형 모델(MLP) 채택

**Integration**: `scripts/run_experiment.py` → `linearity` 커맨드

---

## Phase 1: 기반 구축 (Foundation)

### 1.1 Embedder 모듈 구현

**파일**: `src/models/clip_embedder.py`
```python
class CLIPEmbedder:
    def __init__(self, model_name='ViT-L-14', pretrained='openai', device='mps')
    def embed_image_paths(self, paths: List[str]) -> torch.Tensor  # (B, 768)
    def embed_images(self, images: torch.Tensor) -> torch.Tensor
    @property
    def embed_dim(self) -> int  # 768
```

**파일**: `src/models/clap_embedder.py`
```python
class CLAPEmbedder:
    def __init__(self, model_id=1, enable_fusion=False, device='mps')
    def embed_audio_paths(self, paths: List[str]) -> torch.Tensor  # (B, 512)
    def embed_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor
    def preprocess_audio(self, path: str, max_duration: float = 20.0)
        # Critical: Truncate/Pad to max_duration for batch processing
    @property
    def embed_dim(self) -> int  # 512
    @property
    def sample_rate(self) -> int  # 48000
```

**파일**: `src/models/multimodal_embedder.py`
```python
class MultimodalEmbedder:
    def __init__(self, clip_embedder: CLIPEmbedder, clap_embedder: CLAPEmbedder)
    def embed_image_paths(self, paths: List[str]) -> torch.Tensor
    def embed_audio_paths(self, paths: List[str]) -> torch.Tensor
    @property
    def image_dim(self) -> int  # 768
    @property
    def audio_dim(self) -> int  # 512
```

---

### 1.2 CCA Alignment 구현

**파일**: `src/models/alignment.py`
```python
class CCAAlignment:
    """Canonical Correlation Analysis for cross-modal alignment"""

    def __init__(self, n_components: int = 512):
        self.n_components = n_components
        self.cca = None  # sklearn CCA object

    def fit(
        self,
        image_embeds: np.ndarray,  # (N, 768)
        audio_embeds: np.ndarray,  # (N, 512)
    ) -> None:
        """
        Fit CCA on original (unaugmented) embeddings

        Phase 0-b를 통과한 clean data로 학습
        동일 카테고리 내 원본 이미지/오디오 쌍 사용
        """
        from sklearn.cross_decomposition import CCA
        self.cca = CCA(n_components=self.n_components)
        self.cca.fit(image_embeds, audio_embeds)

    def transform_image(self, image_embeds: np.ndarray) -> np.ndarray:
        # (N, 768) -> (N, 512)
        return self.cca.transform(image_embeds)[0]

    def transform_audio(self, audio_embeds: np.ndarray) -> np.ndarray:
        # (N, 512) -> (N, 512)
        return self.cca.transform(None, audio_embeds)[1]

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.cca, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            self.cca = pickle.load(f)
```

---

### 1.3 Delta Extraction 업데이트

**파일**: `src/experiment/delta_extraction.py` (수정)

**Changes**:
1. `ImageBindEmbedder` → `MultimodalEmbedder`
2. Audio sample rate: `16000` → `48000`
3. `DeltaResult`에 embedding dimension 추가:
```python
@dataclass
class DeltaResult:
    modality: str
    effect_type: str
    intensity: str
    original_path: str
    category: str  # NEW: from folder structure
    delta: np.ndarray
    original_embedding: np.ndarray
    augmented_embedding: np.ndarray
    embedding_dim: int  # NEW: 768 for image, 512 for audio
```

4. 카테고리 자동 추출:
```python
def get_category_from_path(path: str) -> str:
    """
    Extract category from folder structure
    e.g., "data/experiment/images/ocean/img1.jpg" -> "ocean"
    """
    return Path(path).parent.name
```

---

## Phase 2: 프로토타입 및 정렬 (Prototype & Alignment)

### Phase 2-a: CCA Fit

**파일**: `scripts/run_experiment.py` → `fit_alignment` 커맨드

```python
def fit_alignment(config):
    """
    Phase 0-b를 통과한 clean data로 CCA 학습

    1. 원본(unaugmented) 이미지/오디오 임베딩 추출
    2. 동일 개수 샘플링 (balanced)
    3. CCA fit 및 저장
    """
    # Load embedders
    embedder = load_embedders(config)

    # Get image paths (from each category)
    image_paths = get_data_paths(config.data.root_dir, "images", IMAGE_EXTS)
    audio_paths = get_data_paths(config.data.root_dir, "audio", AUDIO_EXTS)

    # Extract original embeddings
    print("Extracting original embeddings for CCA fitting...")
    image_embeds = embedder.embed_image_paths(image_paths).cpu().numpy()
    audio_embeds = embedder.embed_audio_paths(audio_paths).cpu().numpy()

    # Fit CCA
    alignment = CCAAlignment(n_components=config.model.alignment.n_components)
    alignment.fit(image_embeds, audio_embeds)

    # Save
    output_path = Path(config.output.dir) / "alignment.pkl"
    alignment.save(str(output_path))
    print(f"CCA alignment saved to {output_path}")
```

---

### Phase 2-b: Delta Prototype 생성

**파일**: `src/experiment/prototype.py` (수정)

**Changes**: `compute_prototypes()`는 기존과 동일, 단 정규화된 델타의 평균 사용 확인:
```python
def compute_prototypes(
    delta_dataset: DeltaDataset,
    normalize_deltas: bool = True,  # Must be True for Phase 2
) -> PrototypeSet:
    """
    p(t,s) = E_n[Δe_n / ||Δe_n||]

    노이즈 제거를 위해 정규화된 델타의 평균 벡터 계산
    """
```

---

## Phase 3: 대응 관계 발견 및 검증 (Discovery & Validation)

### Phase 3-a: Discovery (Heatmap)

**파일**: `src/experiment/prototype.py` (수정)

`compute_similarity_matrix()`에 alignment 추가:
```python
def compute_similarity_matrix(
    image_prototypes: PrototypeSet,
    audio_prototypes: PrototypeSet,
    alignment: CCAAlignment,  # NEW: required
    image_effect_types: Optional[List[str]] = None,
    audio_effect_types: Optional[List[str]] = None,
    intensities: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    CCA로 정렬된 공간에서 코사인 유사도 행렬 계산

    Visual Prototype (|T|) vs Audio Prototype (|U|)
    """
    # Get matrices
    image_matrix, img_labels, _ = image_prototypes.get_matrix(
        "image", image_effect_types, intensities
    )
    audio_matrix, aud_labels, _ = audio_prototypes.get_matrix(
        "audio", audio_effect_types, intensities
    )

    # Apply CCA alignment
    image_aligned = alignment.transform_image(image_matrix)  # (N, 512)
    audio_aligned = alignment.transform_audio(audio_matrix)  # (M, 512)

    # Normalize
    image_norm = image_aligned / (np.linalg.norm(image_aligned, axis=1, keepdims=True) + 1e-8)
    audio_norm = audio_aligned / (np.linalg.norm(audio_aligned, axis=1, keepdims=True) + 1e-8)

    # Compute similarity
    sim_matrix = image_norm @ audio_norm.T

    return sim_matrix, img_labels, aud_labels
```

**Integration**: `scripts/run_experiment.py` → `phase1` 커맨드
- Heatmap 시각화 및 대각 우세(Diagonal Dominance) 분석

---

### Phase 3-b: Statistical Validation

**파일**: `src/experiment/retrieval.py` (수정)

`DeltaRetrieval`에 alignment 추가:
```python
class DeltaRetrieval:
    def __init__(
        self,
        query_dataset: DeltaDataset,  # Image deltas
        database_dataset: DeltaDataset,  # Audio deltas
        alignment: CCAAlignment,  # NEW
        effect_type_mapping: Optional[Dict[str, str]] = None,
    ):
        self.alignment = alignment
        # ... existing code ...

    def retrieve(
        self,
        query_delta: np.ndarray,  # (768,)
        top_k: int = 10,
        normalize: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        CCA 정렬 후 검색
        """
        # Align query
        query_aligned = self.alignment.transform_image(
            query_delta.reshape(1, -1)
        )[0]  # (512,)

        # Align database
        db_aligned = self.alignment.transform_audio(self.database_matrix)  # (M, 512)

        # Normalize and compute similarity
        # ... existing code ...
```

**파일**: `src/experiment/statistics.py` (추가 분석)

```python
def spearman_intensity_correlation(
    retrieval_metrics: Dict[str, RetrievalMetrics],
) -> Tuple[float, float]:
    """
    Spearman's ρ: Intensity 증가에 따라 Similarity/Matching Score가 단조증가하는지 확인

    Args:
        retrieval_metrics: {intensity: metrics} from evaluate_by_intensity()

    Returns:
        (rho, p_value)
    """
    from scipy.stats import spearmanr

    intensities = ["low", "mid", "high"]
    intensity_map = {i: idx for idx, i in enumerate(intensities)}

    x = [intensity_map[i] for i in retrieval_metrics.keys()]
    y = [m.mean_reciprocal_rank for m in retrieval_metrics.values()]

    return spearmanr(x, y)

def norm_intensity_analysis(
    delta_dataset: DeltaDataset,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Norm Analysis: 벡터 크기가 이펙트 강도와 상관관계가 있는지 확인

    Returns:
        {(modality, effect_type): (spearman_rho, p_value)}
    """
    from scipy.stats import spearmanr

    results = {}
    intensity_map = {"low": 0, "mid": 1, "high": 2}

    for modality in ["image", "audio"]:
        effect_types = set(d.effect_type for d in delta_dataset.deltas if d.modality == modality)

        for effect_type in effect_types:
            deltas = [
                d for d in delta_dataset.deltas
                if d.modality == modality and d.effect_type == effect_type
            ]

            x = [intensity_map[d.intensity] for d in deltas]
            y = [np.linalg.norm(d.delta) for d in deltas]

            rho, pval = spearmanr(x, y)
            results[(modality, effect_type)] = (rho, pval)

    return results
```

**Integration**: `scripts/run_experiment.py` → `phase2` 커맨드
- Retrieval 평가
- Permutation test (기존 `retrieval_permutation_test()` 사용)
- Spearman correlation
- Norm analysis

---

## Implementation Order

### 필수 구현 (Phase 0-2 실행에 필요)

1. ✅ **CLIPEmbedder** (`src/models/clip_embedder.py`)
2. ✅ **CLAPEmbedder** (`src/models/clap_embedder.py`)
3. ✅ **MultimodalEmbedder** (`src/models/multimodal_embedder.py`)
4. ✅ **CCAAlignment** (`src/models/alignment.py`)
5. ✅ **Sensitivity Check** (`src/experiment/sensitivity.py`)
6. ✅ **Linearity Analysis** (`src/experiment/linearity.py` - 기존과 통합)
7. ✅ **Update delta_extraction** (MultimodalEmbedder, 48kHz, category field)
8. ✅ **Update prototype** (alignment parameter)
9. ✅ **Update retrieval** (alignment parameter)
10. ✅ **Statistical analysis** (`src/experiment/statistics.py` - Spearman, Norm)
11. ✅ **Update configs** (`configs/experiment.yaml`)
12. ✅ **Update run_experiment.py** (sensitivity, linearity, fit_alignment, phase1, phase2)
13. ✅ **Move ImageBind to legacy** (`src/models/legacy/`)
14. ✅ **Update check_similarity.py** (use MultimodalEmbedder)

---

## Experiment Pipeline

### Full Pipeline
```bash
# 0. Extract deltas (CLIP + CLAP)
python scripts/run_experiment.py extract

# 1. Phase 0-a: Sensitivity check
python scripts/run_experiment.py sensitivity

# 2. Phase 0-b: Linearity/consistency check
python scripts/run_experiment.py linearity

# 3. Phase 2-a: Fit CCA alignment
python scripts/run_experiment.py fit_alignment

# 4. Phase 2-b: Compute prototypes (already done in extract)

# 5. Phase 3-a: Discovery (heatmap)
python scripts/run_experiment.py phase1

# 6. Phase 3-b: Statistical validation
python scripts/run_experiment.py phase2

# All-in-one (recommended)
python scripts/run_experiment.py all
```

---

## Configuration Update

**파일**: `configs/experiment.yaml`
```yaml
# Model configuration
model:
  # Primary embedders
  clip:
    name: "ViT-L-14"
    pretrained: "openai"
    embed_dim: 768

  clap:
    model_id: 1  # 630k+audioset non-fusion
    enable_fusion: false
    embed_dim: 512
    sample_rate: 48000
    max_duration: 20.0  # seconds (for batch processing)

  # Cross-modal alignment
  alignment:
    method: "cca"
    n_components: 512
    fit_on_originals: true

  # Legacy (weights preserved, not used)
  imagebind:
    enabled: false
    embed_dim: 1024

# Phase 0 thresholds
thresholds:
  sensitivity:
    min_distance: 0.01  # Minimum delta norm to be considered sensitive

  linearity:
    min_cosine: 0.8  # Minimum mean pairwise cosine for consistency
    max_cv: 0.3  # Maximum CV of norms

# Effects configuration
effects:
  intensities: ["low", "mid", "high"]

  image:
    types: ["brightness", "contrast", "saturation", "blur"]

  audio:
    types: ["lpf", "highshelf", "saturation", "reverb"]

# Effect mapping (hypothesis)
effect_mapping:
  blur: "lpf"
  brightness: "highshelf"
  contrast: "saturation"
  saturation: "reverb"

# Statistical analysis
n_permutations: 10000

# Output
output:
  dir: "outputs/experiment"
  save_plots: true
  plot_format: "png"

# Device
device: "mps"
```

---

## Phase 3+ (Placeholder)

### Phase 3: 모델 학습 (The Bridge)

**TODO**: Phase 2 결과를 바탕으로 최적의 '번역기' 학습

**파일**: `src/models/delta_adapter.py` (미구현)
```python
class DeltaAdapter(nn.Module):
    """
    CLIP ΔV (768d) -> CLAP ΔA (512d) 변환

    Phase 2 통계 검증 결과가 충분히 유의미할 경우에만 구현
    """
    pass
```

**Loss Functions**:
- Cosine Embedding Loss (방향 일치)
- MSE Loss (크기 일치, Norm Analysis 결과 참고)

---

### Phase 4: 도구화 및 파라미터 디코드 (Application)

**TODO**: 임베딩 → DSP 파라미터 변환

**파일**: `src/models/parameter_decoder.py` (미구현)
```python
class ParameterDecoder(nn.Module):
    """
    CLAP Δ (512d) -> DSP Params (cutoff_hz, gain_db, drive, wet_ratio, ...)

    Regression 문제로 접근
    Label: 이펙트 적용 시 사용한 실제 파라미터 값
    """
    pass
```

**Web Demo**: Gradio UI (미구현)

---

## Expected Outcomes (Phase 0-2)

### Phase 0-a: Sensitivity
- 각 이펙트가 임베딩 공간에서 감지 가능한지 확인
- 민감도가 낮은 이펙트 제외 목록 생성

### Phase 0-b: Linearity
- 카테고리별 델타 일관성 보고서
- Cross-category variance 분석
- 컨텍스트 불변성 검증

### Phase 2-a: CCA Alignment
- 768d(CLIP) ↔ 512d(CLAP) 정렬 모델 저장
- 정렬 품질 메트릭 (canonical correlation coefficients)

### Phase 3-a: Discovery
- Prototype Similarity Heatmap
- 대각 우세 패턴 시각화
- Human perception과 일치 여부 확인

### Phase 3-b: Validation
- Retrieval Top-k Accuracy
- Permutation Test p-values
- Spearman ρ (intensity monotonicity)
- Norm-Intensity Correlation

---

## Open Questions

1. ✅ CLAP audio truncation: 20초 vs 30초? → 20초로 시작, 필요시 조정
2. ⏳ Phase 0-b에서 consistency 실패 시 대응:
   - Scope 축소 (특정 카테고리 제외) vs
   - Phase 3에서 비선형 모델 채택
3. ⏳ CCA n_components: 512 고정 vs adaptive?
4. ⏳ Phase 3+ 진행 여부는 Phase 2 결과에 따라 결정
