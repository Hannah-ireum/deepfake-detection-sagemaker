# 1. 데이터 준비

> 노트북: `1_data_preparation/prepare_data.ipynb`

## 개요

딥페이크 탐지 실습을 위한 데이터를 준비합니다.

## 학습 내용

- 딥페이크 탐지용 데이터셋 구조 이해
- 사전 준비된 데이터 다운로드
- S3 버킷에 데이터 업로드
- config.json 생성

## 데이터 소스

강사가 미리 준비한 딥페이크 샘플 데이터를 S3에서 다운로드합니다.

### 데이터 구성

| 유형 | 설명 |
|------|------|
| **Real** | 실제 얼굴 이미지 |
| **Fake** | 딥페이크로 생성된 얼굴 이미지 |

### 데이터 분할

| 구분 | 비율 | 용도 |
|------|------|------|
| Train | 70% | 모델 학습 |
| Validation | 15% | 학습 중 검증 |
| Test | 15% | 최종 평가 (Before/After 비교) |

## 주요 코드

### 환경 설정

```python
import os
import sagemaker
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd()).parent
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()
```

### 데이터 다운로드

```python
# ⚠️ 강사가 안내한 버킷명으로 변경
DATA_SOURCE_BUCKET = "workshop-deepfake-data"
DATA_SOURCE_PREFIX = "kodf-sample"

# S3에서 데이터 다운로드
!aws s3 cp s3://{DATA_SOURCE_BUCKET}/{DATA_SOURCE_PREFIX}/ ./data/ --recursive
```

### 데이터 구조 확인

```python
for split in ['train', 'val', 'test']:
    split_dir = data_dir / split
    real_count = len(list((split_dir / 'real').glob('*')))
    fake_count = len(list((split_dir / 'fake').glob('*')))
    print(f"{split}: Real={real_count}, Fake={fake_count}")
```

### 내 S3 버킷에 업로드

```python
s3_data_path = sagemaker_session.upload_data(
    path='./data',
    bucket=bucket,
    key_prefix=f'{prefix}/data'
)
```

### config.json 저장

```python
config = {
    'bucket': bucket,
    'prefix': prefix,
    's3_train_path': f's3://{bucket}/{prefix}/data/train',
    's3_val_path': f's3://{bucket}/{prefix}/data/val',
    's3_test_path': f's3://{bucket}/{prefix}/data/test',
    'local_test_path': str(data_dir.absolute() / 'test'),
    'role': role,
    'region': region
}

config_path = PROJECT_ROOT / 'config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
```

## 데이터 구조

```
data/
├── train/           # 학습 데이터
│   ├── real/        # 실제 얼굴 이미지
│   └── fake/        # 딥페이크 얼굴 이미지
├── val/             # 검증 데이터
│   ├── real/
│   └── fake/
└── test/            # 테스트 데이터 (Before/After 비교용)
    ├── real/
    └── fake/
```

## Before vs After 비교 원리

```
동일한 Test 데이터 사용:

Before (Fine-tuning 전):
  └─ FF++ 모델 → 한국인 테스트 데이터 → ~70% 정확도

After (Fine-tuning 후):
  └─ KoDF 학습 모델 → 동일 테스트 데이터 → ~90% 정확도

비교:
  └─ 동일 데이터에서 +20% 성능 향상 확인
```

## 체크포인트

- [ ] 강사가 안내한 `DATA_SOURCE_BUCKET` 설정
- [ ] 데이터 다운로드 완료
- [ ] 데이터 구조 확인 (train/val/test)
- [ ] 내 S3 버킷에 업로드 완료
- [ ] config.json 저장 완료

## 다음 단계

데이터 준비가 완료되면 [2. Before 평가](02-before-evaluation.md)로 이동합니다.
