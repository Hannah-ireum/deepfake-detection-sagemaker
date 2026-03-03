# 강사용 데이터 준비 가이드

## 개요

Workshop 전에 **한 번만** 데이터를 Public S3 버킷에 업로드하면 됩니다.
참가자들은 노트북 실행만으로 자동으로 데이터를 다운로드합니다.

## Workshop 흐름

```
강사 (1회 준비)              참가자 (실행만)
──────────────              ──────────────
1. Kaggle에서 데이터 다운로드
2. Public S3에 업로드
                            1. 노트북 실행 → 자동 다운로드
                            2. Fine-tuning 전 평가
                            3. Fine-tuning 실행
                            4. Fine-tuning 후 평가
                            5. 성능 비교
                            6. 데모 배포
```

## Step 1: 데이터 다운로드 (Kaggle)

### Kaggle 계정 준비

1. [Kaggle](https://www.kaggle.com) 회원가입
2. Account Settings → API → Create New Token
3. `kaggle.json` 다운로드

### 데이터 다운로드

```bash
# Kaggle CLI 설치
pip install kaggle

# kaggle.json 설정
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 140k Real and Fake Faces 다운로드
kaggle datasets download -d xhlulu/140k-real-and-fake-faces --unzip -p ./kaggle_data
```

### 데이터 샘플링

전체 140k는 너무 크므로 Workshop용 샘플을 추출합니다.

```python
import shutil
import random
from pathlib import Path

# 원본 경로
kaggle_dir = Path('./kaggle_data/real_vs_fake/real_vs_fake')

# Workshop 데이터 디렉토리
data_dir = Path('./workshop-data')

# 샘플 크기
TRAIN_SIZE = 1000  # 클래스당
VAL_SIZE = 200
TEST_SIZE = 200

for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        (data_dir / split / label).mkdir(parents=True, exist_ok=True)

for label in ['real', 'fake']:
    # Train
    src = kaggle_dir / 'train' / label
    images = list(src.glob('*.jpg'))
    random.shuffle(images)
    for img in images[:TRAIN_SIZE]:
        shutil.copy(img, data_dir / 'train' / label / img.name)

    # Val
    src = kaggle_dir / 'valid' / label
    images = list(src.glob('*.jpg'))[:VAL_SIZE]
    for img in images:
        shutil.copy(img, data_dir / 'val' / label / img.name)

    # Test
    src = kaggle_dir / 'train' / label
    images = list(src.glob('*.jpg'))
    random.shuffle(images)
    for img in images[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]:
        shutil.copy(img, data_dir / 'test' / label / img.name)

print("샘플링 완료!")
```

## Step 2: Public S3 버킷 설정

### 버킷 생성

```bash
# 버킷 생성 (전역 고유 이름 필요)
aws s3 mb s3://deepfake-detection-workshop-public --region ap-northeast-2
```

### Public Read 정책 설정

```bash
# 버킷 정책 파일 생성
cat > bucket-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::deepfake-detection-workshop-public/*"
        },
        {
            "Sid": "PublicListBucket",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::deepfake-detection-workshop-public"
        }
    ]
}
EOF

# 정책 적용
aws s3api put-bucket-policy --bucket deepfake-detection-workshop-public --policy file://bucket-policy.json

# Block Public Access 해제
aws s3api put-public-access-block --bucket deepfake-detection-workshop-public --public-access-block-configuration "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
```

### 데이터 업로드

```bash
# Workshop 데이터 업로드
aws s3 cp ./workshop-data/ s3://deepfake-detection-workshop-public/sample-data/ --recursive

# 업로드 확인
aws s3 ls s3://deepfake-detection-workshop-public/sample-data/ --recursive --summarize
```

### 접근 테스트

```bash
# 익명 접근 테스트 (--no-sign-request)
aws s3 ls s3://deepfake-detection-workshop-public/sample-data/ --no-sign-request
```

## Step 3: 노트북 버킷명 확인

`1_data_preparation/prepare_data.ipynb`에서 버킷명이 맞는지 확인:

```python
DATA_SOURCE = "s3://deepfake-detection-workshop-public/sample-data"
```

> 버킷명을 변경한 경우 이 값도 수정해야 합니다.

## 데이터 구조

업로드 후 S3 구조:

```
s3://deepfake-detection-workshop-public/sample-data/
├── train/
│   ├── real/     (1,000장)
│   └── fake/     (1,000장)
├── val/
│   ├── real/     (200장)
│   └── fake/     (200장)
└── test/
    ├── real/     (200장)
    └── fake/     (200장)
```

**총 2,800장** (약 50-100MB)

## 체크리스트

Workshop 전 확인:

- [ ] Kaggle에서 데이터 다운로드 완료
- [ ] Workshop용 샘플 추출 완료 (2,800장)
- [ ] Public S3 버킷 생성 완료
- [ ] Public Read 정책 적용 완료
- [ ] 데이터 업로드 완료
- [ ] 익명 접근 테스트 성공 (`--no-sign-request`)
- [ ] 노트북 DATA_SOURCE 경로 확인

## 비용

- S3 저장소: ~$0.01/월 (100MB 기준)
- 데이터 전송: 첫 100GB 무료 (AWS → 인터넷)

## 문제 해결

### Public Access 차단됨

```bash
# Block Public Access 설정 확인
aws s3api get-public-access-block --bucket deepfake-detection-workshop-public

# 모두 false로 설정
aws s3api put-public-access-block --bucket deepfake-detection-workshop-public --public-access-block-configuration "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
```

### 참가자 다운로드 실패

```bash
# 버킷 정책 확인
aws s3api get-bucket-policy --bucket deepfake-detection-workshop-public

# 직접 테스트
curl -I https://deepfake-detection-workshop-public.s3.ap-northeast-2.amazonaws.com/sample-data/train/real/0001.jpg
```

## Workshop 당일 안내사항

참가자에게 안내할 내용:

1. Workshop Studio URL 공유
2. 노트북 셀 순서대로 실행하면 됨
3. 데이터는 자동으로 다운로드됨
4. 질문은 언제든 환영
