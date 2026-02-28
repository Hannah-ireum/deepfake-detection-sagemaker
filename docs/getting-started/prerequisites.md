# 사전 준비

## 필수 요구사항

### AWS 계정
- AWS 계정이 필요합니다
- SageMaker 서비스 접근 권한이 있어야 합니다
- 비용 발생에 대한 권한 확인 (약 $2.5 예상)

### 로컬 환경
- Python 3.8 이상
- pip 패키지 관리자
- Git

### AWS CLI 설정
AWS CLI가 설치되어 있고 자격 증명이 설정되어 있어야 합니다.

```bash
# AWS CLI 설치 확인
aws --version

# 자격 증명 설정 확인
aws sts get-caller-identity
```

## AWS 리소스

### S3 버킷
학습 데이터와 모델 아티팩트를 저장할 S3 버킷이 필요합니다.

```bash
# 버킷 생성 예시
aws s3 mb s3://your-bucket-name --region ap-northeast-2
```

### IAM Role
SageMaker 실행을 위한 IAM Role이 필요합니다. 다음 정책이 포함되어야 합니다:

- `AmazonSageMakerFullAccess`
- S3 버킷 접근 권한

### SageMaker 환경
다음 중 하나를 사용할 수 있습니다:
- SageMaker Studio
- SageMaker 노트북 인스턴스
- 로컬 환경 (AWS CLI 설정 완료 시)

## 권장 인스턴스 타입

| 용도 | 인스턴스 | 이유 |
|------|----------|------|
| 노트북 | ml.t3.medium | 비용 효율적 |
| Training | ml.g4dn.xlarge | GPU 가속 |
| Endpoint | ml.g4dn.xlarge | 실시간 추론 |
