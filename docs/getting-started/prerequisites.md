# 사전 준비

## Workshop Studio 사용 시

> 실습용 AWS 계정이 제공되므로 별도 준비가 필요 없습니다.

### 필요한 것

| 항목 | 설명 |
|------|------|
| **노트북/PC** | 인터넷 연결 가능 |
| **웹 브라우저** | Chrome 권장 |
| **Workshop URL** | 강사가 제공 |

### 제공되는 것

- AWS 임시 계정 (자동 생성)
- SageMaker Studio 접근 권한
- S3 버킷 (자동 생성)
- IAM Role (자동 설정)
- 필요한 서비스 할당량

```
✅ AWS 계정, CLI 설정, IAM 권한 등은 모두 자동으로 준비됩니다!
```

---

## 개인 AWS 계정 사용 시 (선택)

Workshop Studio 없이 개인 계정으로 실습하려면 아래 준비가 필요합니다.

### AWS 계정 요구사항

- AWS 계정
- SageMaker 서비스 접근 권한
- 비용 발생 (약 $1.7 ~ $2.5)

### IAM Role

SageMaker 실행을 위한 IAM Role에 다음 정책이 필요합니다:

- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (또는 특정 버킷 권한)

### 서비스 할당량

| 서비스 | 리소스 | 필요량 |
|--------|--------|--------|
| SageMaker | ml.g4dn.xlarge (Training) | 1 |
| SageMaker | ml.g4dn.xlarge (Endpoint) | 1 |

> 할당량 부족 시 AWS 콘솔에서 증가 요청이 필요합니다.

---

## 실습 환경 옵션

| 환경 | 장점 | 단점 |
|------|------|------|
| **SageMaker Studio** (권장) | 설정 간편, GPU 사용 가능 | - |
| SageMaker 노트북 인스턴스 | 전통적 방식 | Studio보다 기능 적음 |
| 로컬 환경 | 자유로운 환경 | AWS CLI 설정 필요, GPU 없음 |

---

## 권장 인스턴스 타입

| 용도 | 인스턴스 | 비용 (On-Demand) |
|------|----------|------------------|
| Studio JupyterLab | ml.t3.medium | ~$0.05/hr |
| Training Job | ml.g4dn.xlarge | ~$0.74/hr |
| Inference Endpoint | ml.g4dn.xlarge | ~$0.74/hr |

---

## 다음 단계

준비가 완료되면 [환경 설정](setup.md)으로 이동합니다.
