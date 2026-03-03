# 3. Fine-tuning

> 노트북: `3_fine_tuning/run_finetuning.ipynb`

## 개요

Amazon SageMaker를 활용하여 KoDF 데이터셋으로 모델을 Fine-tuning합니다.

## 학습 내용

- SageMaker Training Job 구성
- **SageMaker Experiments로 실험 추적**
- **Spot Instance로 비용 절감**
- PyTorch Estimator 사용법
- 학습 모니터링

## SageMaker Training Job

### 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│              SageMaker Training + Experiments            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│  │   S3    │───▶│ Training │───▶│   S3    │            │
│  │  Input  │    │Container │    │  Output │            │
│  └─────────┘    └─────────┘    └─────────┘            │
│                      │                                  │
│         ┌───────────┼───────────┐                      │
│         ▼           ▼           ▼                      │
│  ┌───────────┐ ┌─────────┐ ┌─────────────┐            │
│  │Experiments│ │CloudWatch│ │   Model     │            │
│  │  (추적)   │ │  Logs    │ │  Registry   │            │
│  └───────────┘ └─────────┘ └─────────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 주요 코드

### SageMaker Experiments 설정

```python
from sagemaker.experiments.run import Run
from datetime import datetime

EXPERIMENT_NAME = "deepfake-detection-kodf"
RUN_NAME = f"finetuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
```

### Estimator 설정 (Spot Instance 포함)

```python
from sagemaker.pytorch import PyTorch

# Spot Instance 사용 여부 (비용 ~70% 절감)
USE_SPOT = True

estimator = PyTorch(
    entry_point='train.py',
    source_dir='3_fine_tuning',
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 5,
        'batch-size': 32,
        'learning-rate': 0.0001,
        'model-name': 'efficientnet_b0'
    },
    # Spot Instance 설정
    use_spot_instances=USE_SPOT,
    max_wait=7200 if USE_SPOT else None,
    max_run=3600,
)
```

### Experiments와 함께 학습 실행

```python
with Run(
    experiment_name=EXPERIMENT_NAME,
    run_name=RUN_NAME,
    sagemaker_session=sagemaker_session
) as run:
    # 하이퍼파라미터 로깅
    run.log_parameters(hyperparameters)
    run.log_parameter("instance_type", "ml.g4dn.xlarge")
    run.log_parameter("use_spot", USE_SPOT)

    # Training 실행
    estimator.fit(data_channels, wait=True, logs='All')

    # 모델 경로 로깅
    run.log_parameter("model_data", estimator.model_data)
```

## Spot Instance

### 비용 비교

| 모드 | 시간당 비용 | 절감율 |
|------|-------------|--------|
| On-Demand | $0.736 | - |
| Spot | ~$0.22 | ~70% |

### 주의사항

- Spot Instance는 중단될 수 있음
- `max_wait`: 최대 대기 시간 설정
- 체크포인트 설정 권장 (긴 학습 시)

## SageMaker Experiments

### 장점

- 실험 이력 자동 기록
- 하이퍼파라미터 비교
- Before/After 성능 추적
- SageMaker Studio에서 시각화

### 기록되는 정보

| 항목 | 설명 |
|------|------|
| Parameters | 하이퍼파라미터, 인스턴스 타입 등 |
| Metrics | Loss, Accuracy 등 |
| Artifacts | 모델 경로, 데이터 경로 |

## 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| epochs | 5 | 전체 데이터 반복 횟수 |
| batch-size | 32 | 배치 크기 |
| learning-rate | 0.0001 | 학습률 |
| model-name | efficientnet_b0 | 백본 모델 |

## Fine-tuning 전략

1. **Feature Extraction**: Backbone은 고정, Classifier만 학습
2. **작은 Learning Rate**: 사전 학습 가중치 보존
3. **Early Stopping**: Validation Loss 기반

## 체크포인트

- [ ] Experiments 이름 설정
- [ ] Spot Instance 옵션 확인
- [ ] Estimator 설정 완료
- [ ] Training Job 실행
- [ ] 학습 완료 확인
- [ ] Experiments에서 결과 확인

## 다음 단계

학습이 완료되면 [4. After 평가](04-after-evaluation.md)로 이동합니다.
