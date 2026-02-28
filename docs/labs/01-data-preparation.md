# 1. 데이터 준비

> 노트북: `1_data_preparation/prepare_data.ipynb`

## 개요

KoDF(Korean DeepFake) 샘플 데이터를 준비하고 S3에 업로드합니다.

## 학습 내용

- KoDF 데이터셋 구조 이해
- 딥페이크 탐지를 위한 데이터 전처리
- S3 데이터 업로드

## KoDF 데이터셋

Korean DeepFake 데이터셋은 AI Hub에서 제공하는 한국인 얼굴 기반 딥페이크 데이터입니다.

### 데이터 구조

```
data/
├── real/           # 실제 영상에서 추출한 프레임
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
└── fake/           # 딥페이크 영상에서 추출한 프레임
    ├── frame_001.jpg
    ├── frame_002.jpg
    └── ...
```

### 데이터 분할

| 구분 | 비율 | 용도 |
|------|------|------|
| Train | 70% | 모델 학습 |
| Validation | 15% | 학습 중 검증 |
| Test | 15% | 최종 평가 |

## 주요 코드

### 데이터 다운로드 및 전처리

```python
import os
from sklearn.model_selection import train_test_split

# 데이터 경로 설정
data_dir = "data"
real_dir = os.path.join(data_dir, "real")
fake_dir = os.path.join(data_dir, "fake")

# 이미지 파일 목록
real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]

# 레이블 생성 (0: real, 1: fake)
all_images = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

# 데이터 분할
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    all_images, labels, test_size=0.3, stratify=labels, random_state=42
)
```

### S3 업로드

```python
import sagemaker

session = sagemaker.Session()
bucket = session.default_bucket()
prefix = "deepfake-detection"

# 학습 데이터 업로드
train_s3_path = session.upload_data(
    path="data/train",
    bucket=bucket,
    key_prefix=f"{prefix}/train"
)

print(f"Training data uploaded to: {train_s3_path}")
```

## 체크포인트

- [ ] 데이터 다운로드 완료
- [ ] 전처리 완료 (리사이즈, 정규화)
- [ ] Train/Val/Test 분할 완료
- [ ] S3 업로드 완료

## 다음 단계

데이터 준비가 완료되면 [2. Before 평가](02-before-evaluation.md)로 이동합니다.
