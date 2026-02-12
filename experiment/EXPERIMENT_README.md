# Delta Correspondence Experiment

## 목표

Image effect (ΔV)와 Audio effect (ΔA) 사이에 semantic correspondence가 존재하는지를 pretrained multimodal embedding 공간에서 검증하고, 이를 활용하여 DSP 파라미터를 예측하는 decoder를 학습한다.

## 프로젝트 구조

```
DeltaV2A/
├── experiment/
│   ├── configs/
│   │   └── experiment.yaml          # 실험 설정
│   ├── scripts/
│   │   ├── run_experiment.py        # 실험 실행 스크립트
│   │   └── check_similarity.py      # CLIP/CLAP 유사도 검사 도구
│   ├── src/
│   │   ├── delta_extraction.py      # Delta 추출 (Δe = e(aug) - e(orig))
│   │   ├── sensitivity.py           # Phase 0-a: 민감도 검사
│   │   ├── linearity.py             # Phase 0-b: 선형성/일관성 검사
│   │   ├── text_anchor.py           # Phase 1: 텍스트 앵커 생성 (SBERT cross-modal)
│   │   ├── discovery.py             # Phase 1: 3-way 유사도 기반 대응 발견
│   │   ├── phase3_dataset.py        # Phase 3: 학습 데이터셋
│   │   ├── phase3_training.py       # Phase 3: Decoder 학습 루프
│   │   ├── prototype.py             # 프로토타입 계산 (참고용)
│   │   ├── retrieval.py             # Retrieval 평가 (참고용)
│   │   ├── statistics.py            # 통계 분석 (참고용)
│   │   └── effects/
│   │       ├── image_effects.py     # 실험용 이미지 이펙트
│   │       └── audio_effects.py     # 실험용 오디오 이펙트
│   ├── outputs/                     # 실험 출력
│   ├── EXPERIMENT_FLOW.md
│   └── EXPERIMENT_README.md
├── src/
│   ├── effects/
│   │   └── pedalboard_effects.py    # 파이프라인용 오디오 DSP
│   └── models/
│       ├── clip_embedder.py         # CLIP (ViT-L/14) 임베딩
│       ├── clap_embedder.py         # CLAP (630k+audioset) 임베딩
│       ├── multimodal_embedder.py   # CLIP+CLAP 통합 래퍼
│       ├── alignment.py             # CCA 정렬
│       └── decoder.py               # DSP Parameter Decoder (CrossAttention + MLP)
└── data/
    └── original/
        ├── images/                  # 랜드스케이프 이미지 (카테고리별 폴더)
        └── audio/                   # 전자음악 클립
```

---

## 파이프라인

### Phase 0: 검증
- **0-a Sensitivity:** 임베딩 모델이 이펙트를 감지하는지 확인
- **0-b Linearity:** 델타 방향이 카테고리와 무관하게 일관적인지 확인

### Phase 1: 발견 (Text Anchor Ensemble)
- 동의어 확장 + 템플릿 기반 텍스트 델타 생성
- 3-way 유사도 (곱셈): 프로토타입 ↔ 텍스트 앵커 ↔ 크로스모달 SBERT
- Discovery Matrix 출력

### Phase 3: 학습 (The Decoder)
- Discovery Matrix 기반 학습 데이터 생성
- CrossAttention + MLP로 DSP 파라미터 예측
- 입력: CLAP(audio) + CLAP_text(condition) → 출력: DSP parameters

---

## 실험 실행

```bash
# 전체 파이프라인
python experiment/scripts/run_experiment.py all --config experiment/configs/experiment.yaml

# 개별 단계
python experiment/scripts/run_experiment.py extract --config experiment/configs/experiment.yaml
python experiment/scripts/run_experiment.py sensitivity --config experiment/configs/experiment.yaml
python experiment/scripts/run_experiment.py linearity --config experiment/configs/experiment.yaml
python experiment/scripts/run_experiment.py phase1 --config experiment/configs/experiment.yaml
python experiment/scripts/run_experiment.py phase3 --config experiment/configs/experiment.yaml
```

---

## Effect Type Mapping

현재 가설적 매핑 (`experiment/configs/experiment.yaml`):

| Image Effect | Audio Effect | 가설 |
|--------------|--------------|------|
| blur | lpf | 둘 다 고주파 감소 |
| brightness | highshelf | 둘 다 "밝기" 증가 |
| contrast | saturation | 둘 다 dynamic range/richness 증가 |
| saturation | reverb | (검토 필요) |

이 매핑은 예시로, Phase 1 결과를 보고 조정해야 함.
