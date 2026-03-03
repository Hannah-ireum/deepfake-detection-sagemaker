# 1. 데이터 준비

> 노트북: `1_data_preparation/prepare_data.ipynb`

## 개요

KoDF(Korean DeepFake) 샘플 데이터를 준비하고 S3에 업로드합니다.

## 학습 내용

- KoDF 데이터셋 구조 이해
- 딥페이크 탐지를 위한 데이터 전처리
- S3 데이터 업로드
- **config.json 생성** (이후 노트북에서 사용)

## KoDF 데이터셋

Korean DeepFake 데이터셋은 AI Hub에서 제공하는 한국인 얼굴 기반 딥페이크 데이터입니다.

### 데이터 구조

```
data/
├── train/          # 학습 데이터 (70%)
│   ├── real/       # 실제 영상 프레임
│   └── fake/       # 딥페이크 영상 프레임
├── val/            # 검증 데이터 (15%)
│   ├── real/
│   └── fake/
└── test/           # 테스트 데이터 (15%)
    ├── real/
    └── fake/
```

### 데이터 분할

| 구분 | 비율 | 용도 |
|------|------|------|
| Train | 70% | 모델 학습 |
| Validation | 15% | 학습 중 검증 |
| Test | 15% | 최종 평가 |

## 주요 코드

### 환경 설정

```python
import os
import sagemaker
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(os.getcwd()).parent

# SageMaker 세션 설정
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()
prefix = 'deepfake-detection'
```

### 샘플 데이터 생성

실습에서는 샘플 이미지를 생성합니다. 실제로는 KoDF 데이터를 사용합니다.

```python
from pathlib import Path

data_dir = Path('./data')
for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        (data_dir / split / label).mkdir(parents=True, exist_ok=True)
```

### S3 업로드

```python
# 데이터 업로드
s3_data_path = sagemaker_session.upload_data(
    path='./data',
    bucket=bucket,
    key_prefix=f'{prefix}/data'
)

print(f"데이터 업로드 완료: {s3_data_path}")
```

### config.json 생성

이후 노트북에서 사용할 설정 파일을 **프로젝트 루트**에 저장합니다.

```python
import json

config = {
    'bucket': bucket,
    'prefix': prefix,
    'project_root': str(PROJECT_ROOT),
    's3_train_path': f's3://{bucket}/{prefix}/data/train',
    's3_val_path': f's3://{bucket}/{prefix}/data/val',
    's3_test_path': f's3://{bucket}/{prefix}/data/test',
    'local_test_path': str(data_dir / 'test'),
    'role': role,
    'region': region
}

# 프로젝트 루트에 저장
config_path = PROJECT_ROOT / 'config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"설정 저장: {config_path}")
```

## config.json 구조

```json
{
  "bucket": "sagemaker-us-east-1-123456789",
  "prefix": "deepfake-detection",
  "s3_train_path": "s3://bucket/deepfake-detection/data/train",
  "s3_val_path": "s3://bucket/deepfake-detection/data/val",
  "s3_test_path": "s3://bucket/deepfake-detection/data/test",
  "local_test_path": "/home/.../data/test",
  "role": "arn:aws:iam::...",
  "region": "us-east-1"
}
```

> 이 파일은 이후 모든 노트북에서 `PROJECT_ROOT / 'config.json'`으로 접근합니다.

## 체크포인트

- [ ] 샘플 데이터 생성 완료
- [ ] Train/Val/Test 분할 완료
- [ ] S3 업로드 완료
- [ ] **config.json 생성 확인**

## 다음 단계

데이터 준비가 완료되면 [2. Before 평가](02-before-evaluation.md)로 이동합니다.
