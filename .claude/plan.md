# Implementation Plan: CLIP + CLAP Migration with Cross-Modal Alignment

## Overview
Replace ImageBind with CLIP (images) + LAION-CLAP (audio) due to ImageBind's 2-second audio window limitation. Since embedding dimensions differ (CLIP ViT-L/14 = 768d, CLAP = 512d), implement cross-modal alignment via CCA and Linear Projection.

## Configuration
- **CLIP**: ViT-L/14 from OpenAI (768-dim embeddings)
- **CLAP**: LAION music_audioset variant (512-dim embeddings, 48kHz audio)
- **Alignment**: Both CCA (unsupervised) and Linear Projection (trainable)

---

## Phase 1: New Embedder Architecture

### 1.1 Create `src/models/clip_embedder.py`
**Purpose**: CLIP wrapper for image embeddings

**Key Components**:
```python
class CLIPEmbedder:
    def __init__(self, model_name='ViT-L-14', pretrained='openai', device='mps')
    def embed_image_paths(self, image_paths: List[str]) -> torch.Tensor  # (B, 768)
    def embed_images(self, images: torch.Tensor) -> torch.Tensor  # (B, 3, 224, 224) -> (B, 768)
    def preprocess_images(self, pil_images: List[PIL.Image]) -> torch.Tensor
    @property
    def embed_dim(self) -> int  # 768
```

**API**:
- Use `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')`
- Preprocessing: automatic via `preprocess` transform
- Embedding: `model.encode_image(preprocessed_images)`

### 1.2 Create `src/models/clap_embedder.py`
**Purpose**: LAION-CLAP wrapper for audio embeddings

**Key Components**:
```python
class CLAPEmbedder:
    def __init__(self, model_id=1, enable_fusion=False, device='mps')
    def embed_audio_paths(self, audio_paths: List[str]) -> torch.Tensor  # (B, 512)
    def embed_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor  # (B, T) -> (B, 512)
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]  # Load at 48kHz
    @property
    def embed_dim(self) -> int  # 512
    @property
    def sample_rate(self) -> int  # 48000
```

**API**:
- Use `laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny', tmodel='roberta')`
- Load checkpoint: `clap_model.load_ckpt(model_id=1)`  # 630k+audioset non-fusion
- Embedding: `clap_model.get_audio_embedding_from_filelist(paths)` returns numpy
- Convert to torch: `torch.from_numpy(embeddings).float()`

### 1.3 Create `src/models/multimodal_embedder.py`
**Purpose**: Unified interface managing both CLIP and CLAP

**Key Components**:
```python
class MultimodalEmbedder:
    def __init__(self, clip_embedder: CLIPEmbedder, clap_embedder: CLAPEmbedder)
    def embed_image_paths(self, paths: List[str]) -> torch.Tensor  # (B, 768)
    def embed_audio_paths(self, paths: List[str]) -> torch.Tensor  # (B, 512)
    @property
    def image_dim(self) -> int  # 768
    @property
    def audio_dim(self) -> int  # 512
    @property
    def audio_sample_rate(self) -> int  # 48000
```

**Note**: Does NOT align embeddings internally - that's handled by alignment module.

---

## Phase 2: Cross-Modal Alignment

### 2.1 Create `src/models/alignment.py`
**Purpose**: Align CLIP and CLAP embedding spaces

**Key Components**:

#### CCA Alignment (Unsupervised)
```python
class CCAAlignment:
    def __init__(self, n_components: int = 512)  # Project to 512d
    def fit(self, image_embeds: np.ndarray, audio_embeds: np.ndarray) -> None
        # Uses sklearn.cross_decomposition.CCA
        # Learns linear projections W_img, W_aud to maximize correlation
    def transform_image(self, image_embeds: np.ndarray) -> np.ndarray  # (N, 768) -> (N, 512)
    def transform_audio(self, audio_embeds: np.ndarray) -> np.ndarray  # (M, 512) -> (M, 512)
    def save(self, path: str)
    def load(self, path: str)
```

**Requirements for fit()**:
- Need paired or unpaired samples from both modalities
- For experiment: use original (unaugmented) embeddings from each category
- Fit once before computing prototypes

#### Linear Projection (Learnable)
```python
class LinearProjection(nn.Module):
    def __init__(self, image_dim: int = 768, audio_dim: int = 512, shared_dim: int = 512)
    def forward(self, image_embeds: torch.Tensor, audio_embeds: torch.Tensor)
        -> Tuple[torch.Tensor, torch.Tensor]  # Both (*, shared_dim)
    def project_image(self, x: torch.Tensor) -> torch.Tensor  # (*, 768) -> (*, 512)
    def project_audio(self, x: torch.Tensor) -> torch.Tensor  # (*, 512) -> (*, 512)
```

**Training** (optional for phase 1):
- Contrastive loss: encourage high similarity for same effect type, low for different
- Can be added later if CCA alignment is insufficient

### 2.2 Update `src/experiment/delta_extraction.py`
**Changes**:
- Replace `ImageBindEmbedder` with `MultimodalEmbedder`
- Update sample rate: `16000` → `48000` for audio
- Store embedding dimension metadata in `DeltaResult`:
  ```python
  @dataclass
  class DeltaResult:
      # ... existing fields ...
      embedding_dim: int  # 768 for image, 512 for audio
  ```

### 2.3 Update `src/experiment/prototype.py`
**Changes**:
- Add `alignment` parameter to `compute_similarity_matrix()`:
  ```python
  def compute_similarity_matrix(
      image_prototypes: PrototypeSet,
      audio_prototypes: PrototypeSet,
      alignment: Optional[Union[CCAAlignment, LinearProjection]] = None,
      ...
  ) -> Tuple[np.ndarray, List[str], List[str]]:
      # Get matrices
      image_matrix, img_labels, _ = image_prototypes.get_matrix("image", ...)
      audio_matrix, aud_labels, _ = audio_prototypes.get_matrix("audio", ...)

      # Align if provided
      if alignment:
          if isinstance(alignment, CCAAlignment):
              image_aligned = alignment.transform_image(image_matrix)
              audio_aligned = alignment.transform_audio(audio_matrix)
          else:  # LinearProjection
              image_aligned = alignment.project_image(torch.from_numpy(image_matrix)).numpy()
              audio_aligned = alignment.project_audio(torch.from_numpy(audio_matrix)).numpy()
      else:
          raise ValueError("Cross-modal comparison requires alignment")

      # Compute similarity
      sim_matrix = cosine_similarity(image_aligned, audio_aligned)
      return sim_matrix, img_labels, aud_labels
  ```

### 2.4 Update `src/experiment/retrieval.py`
**Changes**:
- Add `alignment` parameter to `DeltaRetrieval.__init__()`
- Apply alignment in `retrieve()` before computing similarities
- Update `cosine_similarity_matrix()` to handle aligned embeddings

---

## Phase 3: Delta Linearity Analysis

### 3.1 Create `src/experiment/linearity.py`
**Purpose**: Verify that effect deltas are consistent across different source images/audio

**Key Analysis**:
```python
@dataclass
class LinearityResult:
    modality: str
    effect_type: str
    intensity: str
    category: str  # e.g., "ocean", "mountain"
    mean_pairwise_cosine: float  # Average cosine sim between normalized deltas
    std_pairwise_cosine: float
    min_cosine: float
    max_cosine: float
    num_samples: int
    delta_norm_cv: float  # Coefficient of variation of delta norms

def delta_linearity_analysis(
    delta_dataset: DeltaDataset,
    category_mapping: Dict[str, str],  # path -> category name
) -> List[LinearityResult]:
    """
    For each (modality, effect_type, intensity, category):
    - Group deltas from same category
    - Compute pairwise cosine similarity of normalized deltas
    - Compute CV of delta norms

    High cosine sim + low CV → effect is linear (context-independent)
    """
    pass

def print_linearity_report(results: List[LinearityResult]):
    """Print formatted table of linearity analysis."""
    pass
```

**Integration**:
- Add to `scripts/run_experiment.py` as `linearity` command
- Run after delta extraction, grouped by image/audio category folders

---

## Phase 4: Update Experiment Pipeline

### 4.1 Update `configs/experiment.yaml`
```yaml
# Model configuration
model:
  # Primary models
  clip:
    name: "ViT-L-14"
    pretrained: "openai"
    embed_dim: 768

  clap:
    model_id: 1  # 630k+audioset non-fusion
    enable_fusion: false
    embed_dim: 512
    sample_rate: 48000

  # Cross-modal alignment
  alignment:
    method: "cca"  # "cca" or "linear_proj"
    n_components: 512  # Shared dimension
    fit_on_originals: true  # Fit CCA on unaugmented embeddings

  # Legacy (not used, but weights preserved)
  imagebind:
    enabled: false
    embed_dim: 1024
```

### 4.2 Update `scripts/run_experiment.py`
**New Commands**:
- `fit_alignment`: Fit CCA/projection on original embeddings
- `linearity`: Run delta linearity analysis
- Updated pipeline: `extract` → `fit_alignment` → `linearity` → `consistency` → `phase1` → `phase2`

**Changes**:
```python
def load_embedders(config):
    clip_emb = CLIPEmbedder(
        model_name=config.model.clip.name,
        pretrained=config.model.clip.pretrained,
        device=config.device,
    )
    clap_emb = CLAPEmbedder(
        model_id=config.model.clap.model_id,
        enable_fusion=config.model.clap.enable_fusion,
        device=config.device,
    )
    return MultimodalEmbedder(clip_emb, clap_emb)

def fit_alignment(config):
    # Load original (unaugmented) embeddings from images and audio
    # Fit CCA and save to outputs/experiment/alignment.pkl
    pass

def run_linearity(config):
    # Load deltas, run linearity analysis, print report
    pass
```

### 4.3 Update `scripts/check_similarity.py`
**Changes**:
- Replace `ImageBindEmbedder` with `MultimodalEmbedder`
- For cross-modal mode: require `--alignment` path or auto-fit CCA on the fly
- Update audio sample rate: `16000` → `48000`

---

## Phase 5: Organize ImageBind (Legacy)

### 5.1 Move to legacy directory
```
src/models/
├── __init__.py           # Only export CLIP/CLAP/Multimodal embedders
├── clip_embedder.py      # NEW
├── clap_embedder.py      # NEW
├── multimodal_embedder.py  # NEW
├── alignment.py          # NEW
└── legacy/
    ├── __init__.py
    └── embedder.py       # Renamed from embedder.py (ImageBind)
```

### 5.2 Update imports
- `src/models/__init__.py`: Export new embedders, not ImageBindEmbedder
- Add comment: "ImageBind moved to src/models/legacy/ - weights preserved but not used"

---

## Phase 6: Testing & Validation

### 6.1 Unit tests
- Test CLIP embedder: load model, embed sample image, check shape (768,)
- Test CLAP embedder: load model, embed sample audio, check shape (512,)
- Test CCA: fit on random data, transform, check shapes
- Test linearity analysis: run on sample deltas

### 6.2 Integration test
```bash
# 1. Extract deltas with new embedders
python scripts/run_experiment.py extract

# 2. Fit alignment
python scripts/run_experiment.py fit_alignment

# 3. Run linearity analysis
python scripts/run_experiment.py linearity

# 4. Run consistency check
python scripts/run_experiment.py consistency

# 5. Run Phase 1 (with alignment)
python scripts/run_experiment.py phase1

# 6. Run Phase 2 (with alignment)
python scripts/run_experiment.py phase2
```

---

## Implementation Order

1. ✅ **CCAAlignment + LinearProjection** (`src/models/alignment.py`)
2. ✅ **CLIPEmbedder** (`src/models/clip_embedder.py`)
3. ✅ **CLAPEmbedder** (`src/models/clap_embedder.py`)
4. ✅ **MultimodalEmbedder** (`src/models/multimodal_embedder.py`)
5. ✅ **Delta linearity module** (`src/experiment/linearity.py`)
6. ✅ **Update delta_extraction** (use MultimodalEmbedder, 48kHz audio)
7. ✅ **Update prototype** (add alignment parameter)
8. ✅ **Update retrieval** (add alignment parameter)
9. ✅ **Update configs** (new model config)
10. ✅ **Update run_experiment.py** (fit_alignment, linearity commands)
11. ✅ **Move ImageBind to legacy** (preserve weights)
12. ✅ **Update check_similarity.py** (use new embedders)
13. ✅ **Test end-to-end**

---

## Expected Outcomes

### Before (ImageBind):
- ✅ Unified 1024d space (no alignment needed)
- ❌ 2-second audio window (too short for music)
- ❌ Limited audio quality for music

### After (CLIP + CLAP):
- ✅ Full-length audio support (CLAP handles variable length)
- ✅ Music-specific embeddings (CLAP trained on music)
- ✅ Better image embeddings (CLIP ViT-L/14)
- ✅ Cross-modal alignment via CCA/projection
- ✅ Delta linearity verification (context-independence check)

---

## Open Questions
1. ✅ CCA n_components: 512 (match CLAP dim)
2. ✅ Fit CCA on: original (unaugmented) embeddings from each category
3. ⏳ If CCA insufficient: train LinearProjection with contrastive loss
4. ⏳ Audio duration handling: CLAP auto-handles via fusion, but verify with long audio

## Notes
- ImageBind weights preserved in `~/.cache/imagebind/` (don't delete)
- CLIP weights: `~/.cache/open_clip/`
- CLAP weights: auto-download to HuggingFace cache
