# DeltaV2A: Visual-to-Audio Delta Transformation

시각적 편집(이미지 변화)을 청각적 변화(오디오 스타일 변환)로 매핑하는 연구 프로젝트입니다.

## 프로젝트 개요

본 시스템은 (I_init, A_init) 초기 쌍이 주어졌을 때, 사용자의 이미지 편집 I_edit을 받아 오디오의 구조는 보존하면서 스타일을 변환(A_edit)하는 것을 목표로 합니다.

### 핵심 특징
- **Prior-Guided Learning**: Type-II rules와 ImageBind 기반의 soft prior를 결합
- **Delta-Conditioned Mapping**: 시각적 변화(ΔV)를 청각적 제어 신호로 변환
- **Structure Preservation**: 오디오의 리듬, 하모니, 에너지 등 구조적 요소 보존
- **Style Steering**: Timbre, Space, Texture 등 스타일 요소 변환

## 프로젝트 구조

```
DeltaV2A/
├── configs/               # 설정 파일들
│   ├── default.yaml      # 기본 설정
│   ├── stage0_config.yaml # Prior 구성
│   ├── stage1_config.yaml # Audio control 학습
│   └── stage2_config.yaml # Cross-modal mapping
├── src/
│   ├── data/             # 데이터 로딩 및 전처리
│   │   ├── dataset.py    # Dataset 클래스들
│   │   └── transforms.py # Audio/Image 변환
│   ├── models/           # 모델 구현
│   │   ├── prior.py      # Hard/Soft Prior
│   │   ├── visual_encoder.py   # Visual Delta Encoder
│   │   ├── delta_mapping.py    # Delta Mapping Module
│   │   ├── s_encoder.py        # S Encoder (Stage 1)
│   │   └── audio_generator.py  # Audio Generator
│   ├── losses/           # Loss functions
│   ├── training/         # 학습 로직 (TODO)
│   └── utils/            # 유틸리티 함수들
├── scripts/              # 학습/추론 스크립트
│   ├── train_stage0.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── inference.py
├── notebooks/            # Jupyter notebooks
└── requirements.txt      # 의존성
```

## 설치

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv_DeltaV2A
source venv_DeltaV2A/bin/activate  # Mac/Linux
# venv_DeltaV2A\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 주요 의존성

- PyTorch >= 2.0.0
- torchaudio, torchvision
- librosa, soundfile (오디오 처리)
- open_clip_torch, laion-clap (멀티모달)
- diffusers (AudioLDM)
- pytorch-lightning (학습)

## 시스템 아키텍처

### Stage 0: Prior 구성
Valid coupling space 정의 및 C_prior estimator 구축

**입력**: 고유사도 (I, A) 쌍
**출력**: Prior estimator, Type-II rules 검증

### Stage 1: Audio-only Control 학습
S_proxy 공간 학습 및 head 특화

**Phase 1-A**: Synthetic pairs (DSP 기반)
**Phase 1-B**: Remix pairs (실제 remix)
**출력**: S_encoder, tuned Audio Generator, S_proxy 통계

### Stage 2: Cross-Modal Mapping
시각적 delta를 오디오 제어로 매핑

**Phase 2-A**: g 모듈만 학습 (generation 없이)
**Phase 2-B**: End-to-end with LoRA fine-tuning
**출력**: Visual encoder, Delta mapping module, LoRA weights

## 학습 단계

### Stage 0 실행
```bash
python scripts/train_stage0.py --config configs/stage0_config.yaml
```

### Stage 1 실행
```bash
# Phase 1-A: Synthetic warmup
python scripts/train_stage1.py --config configs/stage1_config.yaml --phase 1a

# Phase 1-B: Remix fine-tuning
python scripts/train_stage1.py --config configs/stage1_config.yaml --phase 1b
```

### Stage 2 실행
```bash
# Phase 2-A: g only
python scripts/train_stage2.py --config configs/stage2_config.yaml --phase 2a

# Phase 2-B: End-to-end
python scripts/train_stage2.py --config configs/stage2_config.yaml --phase 2b
```

## 추론

```bash
python scripts/inference.py \
    --image_init path/to/init_image.jpg \
    --image_edit path/to/edited_image.jpg \
    --audio_init path/to/init_audio.wav \
    --output path/to/output.wav \
    --noise_level 0.5
```

## 데이터 준비

각 stage별로 필요한 데이터:

### Stage 0: High-Similarity Pairs
- 형식: `{image_path, audio_path, similarity}`
- 개수: ~10,000
- 요구사항: 상위 5% 유사도

### Stage 1-A: Synthetic Pairs
- 형식: `{audio_init, audio_edit, head_target, effect_name}`
- 개수: ~2,000
- 생성: DSP 효과 적용 (reverb, EQ, compression 등)

### Stage 1-B: Remix Pairs
- 형식: `{original_path, remix_path, genre}`
- 개수: ~1,000
- 요구사항: 구조 유지, 스타일 변경

### Stage 2: Cross-Modal Triplets
- 형식: `{image_init, image_edit, audio_init, edit_type}`
- 개수: ~20,000
- 주의: A_edit 없음 (zero-shot)

## 현재 상태 및 TODO

### ✅ 완료
- [x] 프로젝트 구조 설계
- [x] 기본 환경 설정 (requirements, configs)
- [x] 데이터 처리 모듈 (transforms, datasets)
- [x] Prior 구성 모듈 (Hard/Soft Prior)
- [x] Visual Delta Encoder
- [x] Delta Mapping Module
- [x] S Encoder
- [x] Audio Generator wrapper (placeholder)

### 🚧 진행 중
- [ ] Loss functions 구현
- [ ] Training loops (Stage 0, 1, 2)
- [ ] Evaluation metrics
- [ ] Inference pipeline

### 📝 다음 단계
1. **Loss Functions 완성**
   - Reconstruction, Structure Preservation
   - Rank Consistency, Coherence
   - Manifold projection losses

2. **Training Scripts 구현**
   - Stage별 학습 로직
   - Logging, checkpointing
   - Validation loops

3. **데이터 파이프라인 구축**
   - Synthetic pair 생성 스크립트
   - 데이터 전처리 도구
   - Metadata 생성

4. **AudioLDM 통합**
   - 실제 AudioLDM 모델 로드
   - FiLM conditioning 구현
   - LoRA adaptation

5. **평가 및 실험**
   - Metrics 구현
   - Ablation studies
   - User study 준비

## 서버 학습 가이드

로컬 GPU가 부족한 경우:

### Google Colab
```python
# Colab에서 실행
!git clone https://github.com/your-repo/DeltaV2A.git
%cd DeltaV2A
!pip install -r requirements.txt

# 데이터 업로드 (Google Drive 연동)
from google.colab import drive
drive.mount('/content/drive')

# 학습 실행
!python scripts/train_stage1.py --config configs/stage1_config.yaml
```

### 학교 서버
```bash
# SSH 접속
ssh username@server.address

# 프로젝트 클론
git clone <your-repo>
cd DeltaV2A

# 가상환경 설정
module load python/3.9  # 서버 환경에 따라
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 백그라운드 학습 (tmux/screen 사용)
tmux new -s deltav2a
python scripts/train_stage1.py --config configs/stage1_config.yaml
# Ctrl+B, D로 detach
```

## 참고 문서

- `DeltaV2A 개요.md`: 프로젝트 동기 및 이론적 배경
- `System Specification v1.md`: 상세 시스템 명세
- `configs/*.yaml`: 각 stage별 하이퍼파라미터

## 라이선스

[MIT License](LICENSE)

## Contact

궁금한 점이나 문제가 있으면 이슈를 등록해주세요.
