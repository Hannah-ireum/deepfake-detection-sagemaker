# 비용 최적화

## 예상 비용 상세

### 기본 실습 비용

| 리소스 | 인스턴스 타입 | 사용 시간 | 시간당 비용 | 총 비용 |
|--------|---------------|-----------|-------------|---------|
| Training Job | ml.g4dn.xlarge | 1시간 | $0.736 | ~$0.74 |
| Endpoint | ml.g4dn.xlarge | 2시간 | $0.736 | ~$1.47 |
| S3 Storage | - | 1GB | $0.023/GB | ~$0.02 |
| Data Transfer | - | 1GB | $0.00 | $0.00 |
| **총합** | | | | **~$2.23** |

### 인스턴스 옵션 비교

| 인스턴스 | GPU | 메모리 | 시간당 비용 | 학습 시간 | 총 비용 |
|----------|-----|--------|-------------|-----------|---------|
| ml.g4dn.xlarge | T4 | 16GB | $0.736 | ~1시간 | ~$0.74 |
| ml.g4dn.2xlarge | T4 | 32GB | $0.958 | ~50분 | ~$0.80 |
| ml.p3.2xlarge | V100 | 61GB | $3.825 | ~30분 | ~$1.91 |

**권장**: ml.g4dn.xlarge (비용 대비 성능 최적)

## 비용 절감 전략

### 1. Spot Instance 활용

Training Job에 Spot Instance 사용 시 최대 90% 절감:

```python
estimator = PyTorch(
    ...
    use_spot_instances=True,
    max_wait=3600,  # 최대 대기 시간 (초)
    max_run=3600,   # 최대 실행 시간 (초)
    checkpoint_s3_uri=f's3://{bucket}/checkpoints'
)
```

| 모드 | 비용 | 절감율 |
|------|------|--------|
| On-Demand | $0.736/hr | - |
| Spot | ~$0.22/hr | ~70% |

### 2. Endpoint 자동 스케일링

사용량에 따른 자동 스케일링 설정:

```python
# 최소 인스턴스 수를 0으로 설정
autoscaling_client = boto3.client('application-autoscaling')

autoscaling_client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=0,
    MaxCapacity=2
)
```

### 3. Serverless Inference

실습 후 Serverless로 전환 시 유휴 비용 제거:

```python
model.deploy(
    serverless_inference_config={
        'MemorySizeInMB': 4096,
        'MaxConcurrency': 10
    }
)
```

| 모드 | 유휴 비용 | 추론 비용 |
|------|-----------|-----------|
| Real-time | $0.736/hr | 포함 |
| Serverless | $0.00 | $0.00012/req |

### 4. 사용 후 리소스 정리

**중요**: 실습 완료 후 반드시 삭제

```python
# Endpoint 삭제
predictor.delete_endpoint()
predictor.delete_model()

# S3 데이터 정리 (선택)
!aws s3 rm s3://{bucket}/{prefix} --recursive
```

## 비용 모니터링

### AWS Cost Explorer 활용

1. AWS Console → Cost Explorer
2. Filter by Service: SageMaker
3. Group by: Usage Type

### 비용 알림 설정

```bash
# AWS Budgets 생성
aws budgets create-budget \
    --account-id YOUR_ACCOUNT_ID \
    --budget file://budget.json \
    --notifications-with-subscribers file://notifications.json
```

## 비용 체크리스트

실습 완료 후 확인:

- [ ] SageMaker Endpoint 삭제
- [ ] SageMaker Model 삭제
- [ ] 불필요한 S3 데이터 삭제
- [ ] CloudWatch 로그 보존 기간 확인
- [ ] Training Job 아티팩트 정리
