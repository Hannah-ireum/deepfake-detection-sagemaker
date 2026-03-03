# 1. 데이터 준비

> 노트북: `1_data_preparation/prepare_data.ipynb`

## 개요

딥페이크 탐지 실습을 위한 데이터를 준비합니다.

## 학습 내용

- 딥페이크 탐지용 데이터셋 구조 이해
- Kaggle에서 데이터 다운로드
- S3 버킷에 데이터 업로드
- config.json 생성

## 데이터 소스

**140k Real and Fake Faces** (Kaggle) 데이터셋을 사용합니다.

- **출처**: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- **크기**: 약 4GB (140,000장)
- **구성**: Real 70,000장 + Fake 70,000장
- **장점**: 신청 없이 바로 다운로드 가능

### 데이터 구성

| 유형 | 설명 |
|------|------|
| **Real** | StyleGAN으로 생성되지 않은 실제 얼굴 이미지 |
| **Fake** | StyleGAN으로 생성된 가짜 얼굴 이미지 |

### Workshop용 샘플 크기

| 구분 | 클래스당 | 총 개수 | 용도 |
|------|----------|---------|------|
| Train | 1,000장 | 2,000장 | 모델 학습 |
| Validation | 200장 | 400장 | 학습 중 검증 |
| Test | 200장 | 400장 | Before/After 비교 |

## 사전 준비: Kaggle API

1. [Kaggle](https://www.kaggle.com) 계정 생성
2. **Account Settings** → **API** → **Create New Token** 클릭
3. 다운로드된 `kaggle.json` 파일 확인

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

### Kaggle API 설정

```python
!pip install -q kaggle
import os
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
```

### 데이터 다운로드

```python
# Kaggle 데이터셋 다운로드 (약 4GB)
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces --unzip -p ./kaggle_data
```

### 데이터 구조 변환

```python
# Workshop용 샘플 크기
TRAIN_SIZE = 1000  # 클래스당
VAL_SIZE = 200
TEST_SIZE = 200

# Kaggle → Workshop 형식 변환
for label in ['real', 'fake']:
    # train, val, test 폴더로 복사
    ...
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

- [ ] Kaggle 계정 생성 및 API Token 발급
- [ ] kaggle.json 설정 완료
- [ ] 140k Real and Fake Faces 데이터셋 다운로드 완료
- [ ] Workshop 형식으로 데이터 변환 완료 (train/val/test)
- [ ] 내 S3 버킷에 업로드 완료
- [ ] config.json 저장 완료

## 다음 단계

데이터 준비가 완료되면 [2. Before 평가](02-before-evaluation.md)로 이동합니다.
