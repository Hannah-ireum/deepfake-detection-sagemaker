# 강사용 데이터 준비 가이드

## 개요

Workshop 진행 전 딥페이크 샘플 데이터를 S3에 업로드해야 합니다.

## 데이터 요구사항

### 데이터 구조

```
kodf-sample/
├── train/
│   ├── real/     # 실제 얼굴 이미지 (100-200장)
│   └── fake/     # 딥페이크 얼굴 이미지 (100-200장)
├── val/
│   ├── real/     # (20-50장)
│   └── fake/     # (20-50장)
└── test/
    ├── real/     # (20-50장)
    └── fake/     # (20-50장)
```

### 이미지 사양

| 항목 | 요구사항 |
|------|----------|
| 크기 | 224x224 픽셀 (권장) |
| 포맷 | JPG 또는 PNG |
| 내용 | 얼굴 이미지 (크롭됨) |

## 데이터 소스 옵션

### Option 1: KoDF 데이터 (권장)

1. [AI Hub](https://www.aihub.or.kr/)에서 KoDF 데이터셋 신청
2. 비디오에서 프레임 추출
3. 얼굴 크롭 및 리사이즈

### Option 2: FaceForensics++ 데이터

1. [FaceForensics++](https://github.com/ondyari/FaceForensics) 데이터 신청
2. 다양한 딥페이크 방식 포함 (DeepFakes, Face2Face 등)

### Option 3: 공개 샘플 데이터

- Kaggle Deepfake Detection Challenge 샘플
- 연구용 공개 데이터셋

## S3 업로드 방법

### Step 1: S3 버킷 생성

```bash
# 버킷 생성 (리전에 맞게 수정)
aws s3 mb s3://workshop-deepfake-data --region ap-northeast-2
```

### Step 2: 버킷 정책 설정

참가자들이 읽을 수 있도록 정책 설정:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "WorkshopReadAccess",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::workshop-deepfake-data/kodf-sample/*",
            "Condition": {
                "StringEquals": {
                    "aws:PrincipalAccount": ["WORKSHOP_ACCOUNT_ID"]
                }
            }
        }
    ]
}
```

또는 Workshop Studio 환경에서는 같은 계정 내이므로 별도 정책 불필요.

### Step 3: 데이터 업로드

```bash
# 로컬 데이터를 S3에 업로드
aws s3 cp ./kodf-sample/ s3://workshop-deepfake-data/kodf-sample/ --recursive
```

### Step 4: 업로드 확인

```bash
# 파일 개수 확인
aws s3 ls s3://workshop-deepfake-data/kodf-sample/ --recursive --summarize
```

## 노트북 설정 변경

참가자 노트북에서 데이터 소스 버킷명 안내:

```python
# prepare_data.ipynb에서 이 값을 안내
DATA_SOURCE_BUCKET = "workshop-deepfake-data"
DATA_SOURCE_PREFIX = "kodf-sample"
```

## 체크리스트

Workshop 전 확인사항:

- [ ] S3 버킷 생성 완료
- [ ] 데이터 업로드 완료 (train/val/test)
- [ ] 참가자 계정에서 접근 가능 확인
- [ ] 노트북에 버킷명 안내 준비
- [ ] 데이터 개수 확인 (각 클래스별 최소 50장)

## 대안: Workshop Studio Asset

Workshop Studio를 사용하는 경우, Asset으로 데이터를 미리 포함시킬 수 있습니다:

1. Workshop Studio 콘솔 → Assets
2. S3 Asset 추가
3. 참가자 환경에 자동 배포

## 문제 해결

### 참가자가 데이터 다운로드 실패 시

```bash
# 권한 확인
aws s3 ls s3://workshop-deepfake-data/kodf-sample/

# 직접 복사 시도
aws s3 cp s3://workshop-deepfake-data/kodf-sample/ ./data/ --recursive
```

### 데이터 용량 문제

- 이미지당 약 20-50KB
- 전체 약 50-100MB
- 다운로드 시간: 1-2분
