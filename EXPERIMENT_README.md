# Delta Correspondence Experiment

## 목표

Image effect (ΔV)와 Audio effect (ΔA) 사이에 semantic correspondence가 존재하는지를 pretrained multimodal embedding 공간에서 검증한다.

## 프로젝트 구조

```
DeltaV2A/
├── configs/
│   └── experiment.yaml          # 실험 설정
├── src/
│   ├── effects/
│   │   ├── image_effects.py     # 이미지 이펙트 (brightness, contrast, saturation, blur)
│   │   └── audio_effects.py     # 오디오 이펙트 (lpf, highshelf, saturation, reverb)
│   ├── experiment/
│   │   ├── delta_extraction.py  # Delta 추출 (Δe = e(aug) - e(orig))
│   │   ├── prototype.py         # Prototype 계산
│   │   ├── retrieval.py         # Delta-to-delta retrieval
│   │   └── statistics.py        # 통계 분석 (permutation test, baseline)
│   └── models/
│       └── embedder.py          # ImageBind wrapper
├── scripts/
│   ├── run_experiment.py        # 실험 실행 스크립트
│   └── download_bandcamp_cc.py  # Bandcamp CC 음악 다운로더
└── data/
    └── experiment/              # 실험 데이터 (TODO: 준비 필요)
        ├── images/              # 랜드스케이프 이미지
        └── audio/               # 4bar 전자음악 클립
```

---

## 미정의 부분 (TODO)

### 1. 데이터셋 준비

#### 이미지 데이터
- **내용**: Landscape 이미지
- **형식**: JPG, PNG 등
- **위치**: `data/experiment/images/`
- **수량**: 50-100장 권장

**옵션**:
- Unsplash/Pexels에서 landscape 태그 이미지 다운로드
- CC 라이선스 이미지 사용

#### 오디오 데이터
- **내용**: 4-bar 전자음악 (techno, house 등)
- **조건**: 4/4 박자, 120-140 BPM, no vocal
- **형식**: WAV 또는 MP3
- **길이**: 4-8초 (4 bars at 120-140 BPM)
- **위치**: `data/experiment/audio/`
- **수량**: 50-100개 권장

**옵션**:
- Bandcamp CC 다운로더 사용: `python scripts/download_bandcamp_cc.py --auto --limit 50`
- MTG-Jamendo에서 electronic 태그 음악 다운로드
- 직접 4-bar 세그먼트로 잘라서 준비

---

### 2. Effect Type Mapping 검토

현재 가설적 매핑 (`configs/experiment.yaml`):

| Image Effect | Audio Effect | 가설 |
|--------------|--------------|------|
| blur | lpf | 둘 다 고주파 감소 |
| brightness | highshelf | 둘 다 "밝기" 증가 |
| contrast | saturation | 둘 다 dynamic range/richness 증가 |
| saturation | reverb | (검토 필요) |

이 매핑은 Phase 1 결과를 보고 조정해야 할 수 있음.

---

### 3. Content Dependency 처리 옵션

실험 문서에서 언급된 3가지 옵션:

**Option C1: Within-content 먼저**
- 같은 content 내에서만 delta 비교
- 현재 구현: 지원하지 않음 (TODO)

**Option C2: Content-stratified prototype**
- Content 그룹별로 평균 → 그 다음 그룹 평균
- 현재 구현: 부분 지원 (모든 샘플 평균)

**Option C3: Matched-pair analysis**
- e(I0), e(A0) 유사도가 높은 쌍만 선택
- 현재 구현: 지원하지 않음 (TODO)

---

## 실험 실행 방법

### 1. 데이터 준비
```bash
# 데이터 디렉토리 생성
mkdir -p data/experiment/images
mkdir -p data/experiment/audio

# 이미지와 오디오 파일을 각 디렉토리에 배치
```

### 2. Delta 추출
```bash
python scripts/run_experiment.py extract --config configs/experiment.yaml
```

### 3. Phase 1: Prototype 유사도 분석
```bash
python scripts/run_experiment.py phase1 --config configs/experiment.yaml
```

출력:
- `outputs/experiment/phase1_results.json`: 유사도 매트릭스
- `outputs/experiment/phase1_heatmap.png`: 히트맵 시각화

### 4. Phase 2: Retrieval 평가
```bash
python scripts/run_experiment.py phase2 --config configs/experiment.yaml
```

출력:
- Retrieval metrics (MRR, Top-k accuracy)
- Permutation test p-value
- Trivial confound baseline
- Norm monotonicity analysis

### 전체 파이프라인 실행
```bash
python scripts/run_experiment.py all --config configs/experiment.yaml
```

---

## 추가 구현 필요 사항

1. **Cross-model replication**
   - AudioCLIP 또는 CLIP+CLAP 조합으로 동일 실험 반복
   - 현재: ImageBind만 구현됨

2. **4-bar segment 자동 분할**
   - 긴 음악 파일을 자동으로 4-bar 세그먼트로 분할
   - BPM 감지 → bar 계산 → 분할

3. **Visualization 개선**
   - Intensity별 상세 분석 시각화
   - Cross-intensity alignment 시각화
