# 트러블슈팅

## 일반적인 문제

### 1. AWS 자격 증명 오류

**증상**:
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**해결**:
```bash
# AWS CLI 설정 확인
aws configure list

# 자격 증명 재설정
aws configure
```

### 2. IAM 권한 부족

**증상**:
```
AccessDeniedException: User is not authorized to perform: sagemaker:CreateTrainingJob
```

**해결**:
- IAM Role에 `AmazonSageMakerFullAccess` 정책 추가
- S3 버킷 접근 권한 확인

### 3. S3 버킷 접근 오류

**증상**:
```
S3UploadFailedError: Failed to upload file to S3
```

**해결**:
```python
# 버킷 존재 확인
import boto3
s3 = boto3.client('s3')
s3.head_bucket(Bucket='your-bucket-name')

# 권한 확인
aws s3 ls s3://your-bucket-name/
```

## SageMaker Training 문제

### 4. Training Job 실패

**증상**:
```
AlgorithmError: Training job failed
```

**해결**:
1. CloudWatch 로그 확인:
```bash
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name <training-job-name>/algo-1-*
```

2. 일반적인 원인:
   - CUDA Out of Memory → batch size 줄이기
   - 데이터 경로 오류 → S3 경로 확인
   - 의존성 오류 → requirements.txt 확인

### 5. GPU 메모리 부족

**증상**:
```
CUDA out of memory. Tried to allocate X MiB
```

**해결**:
```python
# 배치 사이즈 줄이기
hyperparameters={
    'batch-size': 16,  # 32 → 16
    ...
}

# 또는 더 큰 인스턴스 사용
instance_type='ml.g4dn.2xlarge'
```

### 6. 학습이 너무 느림

**원인**: CPU 인스턴스 사용 중

**해결**:
```python
# GPU 인스턴스 사용
instance_type='ml.g4dn.xlarge'  # ml.m5.xlarge 대신
```

## SageMaker Endpoint 문제

### 7. Endpoint 배포 실패

**증상**:
```
ModelError: Model container exited with a status of 1
```

**해결**:
1. inference.py 문법 확인
2. model_fn 반환값 확인
3. 필요 패키지 requirements.txt에 포함 확인

### 8. Endpoint 추론 오류

**증상**:
```
ModelError: Received client error (400) from model
```

**해결**:
```python
# 입력 형식 확인
import json
payload = json.dumps({'image': base64_string})

response = predictor.predict(
    payload,
    initial_args={'ContentType': 'application/json'}
)
```

### 9. Endpoint Timeout

**증상**:
```
ModelError: Inference timed out
```

**해결**:
```python
# Endpoint 설정에서 timeout 증가
predictor = model.deploy(
    ...
    model_server_timeout=120  # 기본 60초
)
```

## Gradio 문제

### 10. Gradio 실행 안됨

**증상**:
```
Could not find any running Gradio servers
```

**해결**:
```bash
# 포트 확인
lsof -i :7860

# 다른 포트 사용
demo.launch(server_port=7861)
```

### 11. Public URL 생성 실패

**증상**:
```
Could not create share link
```

**해결**:
```python
# 로컬에서만 실행
demo.launch(share=False, server_name="0.0.0.0")
```

## 데이터 관련 문제

### 12. 이미지 로드 오류

**증상**:
```
PIL.UnidentifiedImageError: cannot identify image file
```

**해결**:
```python
from PIL import Image

def safe_load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
```

### 13. 데이터 불균형

**증상**: 모델이 한 클래스만 예측

**해결**:
```python
# 클래스 가중치 적용
from torch.utils.data import WeightedRandomSampler

class_counts = [len(real_images), len(fake_images)]
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

## 도움 요청

문제가 지속되면:
1. 에러 메시지 전체 복사
2. CloudWatch 로그 확인
3. AWS 문서 참조
4. AWS Support 문의
