# 환경 설정

## 1. 저장소 클론

```bash
git clone https://github.com/Hannah-ireum/deepfake-detection-sagemaker.git
cd deepfake-detection-sagemaker
```

## 2. 가상환경 설정 (권장)

```bash
# 가상환경 생성
python -m venv venv

# 활성화 (macOS/Linux)
source venv/bin/activate

# 활성화 (Windows)
venv\Scripts\activate
```

## 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 주요 패키지

| 패키지 | 버전 | 용도 |
|--------|------|------|
| sagemaker | 2.x | SageMaker SDK |
| torch | 2.x | PyTorch 딥러닝 |
| timm | 0.9.x | 사전학습 모델 |
| gradio | 4.x | 데모 UI |
| boto3 | 1.x | AWS SDK |

## 4. AWS 자격 증명 확인

```bash
# 현재 설정된 AWS 계정 확인
aws sts get-caller-identity

# SageMaker 리전 확인
aws configure get region
```

## 5. S3 버킷 설정

노트북 실행 전에 S3 버킷 이름을 설정합니다.

```python
# 노트북 상단에서 설정
BUCKET_NAME = "your-bucket-name"
PREFIX = "deepfake-detection"
```

## 6. 실습 시작

모든 설정이 완료되면 `1_data_preparation/prepare_data.ipynb`부터 순서대로 실행합니다.

## 문제 해결

### 권한 오류 발생 시
IAM Role에 필요한 정책이 연결되어 있는지 확인합니다.

### 패키지 설치 오류 시
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```
