# 3. Fine-tuning

> 노트북: `3_fine_tuning/run_finetuning.ipynb`

## 개요

Amazon SageMaker를 활용하여 KoDF 데이터셋으로 모델을 Fine-tuning합니다.

## 학습 내용

- SageMaker Training Job 구성
- PyTorch Estimator 사용법
- 하이퍼파라미터 설정
- 학습 모니터링

## SageMaker Training Job

### 아키텍처

```
┌─────────────────────────────────────────────────┐
│              SageMaker Training                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│  │   S3    │───▶│ Training │───▶│   S3    │    │
│  │  Input  │    │Container │    │  Output │    │
│  └─────────┘    └─────────┘    └─────────┘    │
│                      │                         │
│                      ▼                         │
│              ┌─────────────┐                   │
│              │ CloudWatch  │                   │
│              │    Logs     │                   │
│              └─────────────┘                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 주요 코드

### Estimator 설정

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='3_fine_tuning',
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 0.0001,
        'weight-decay': 0.01
    }
)
```

### 학습 실행

```python
# 학습 시작
estimator.fit({
    'train': train_s3_path,
    'validation': val_s3_path
})
```

### 학습 스크립트 (train.py)

```python
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train(args):
    # 모델 초기화
    model = create_model()
    model.load_state_dict(torch.load('ff_pretrained.pth'))

    # 마지막 레이어만 학습
    for param in model.parameters():
        param.requires_grad = False
    model.classifier.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2%}")

    # 모델 저장
    torch.save(model.state_dict(), '/opt/ml/model/model.pth')
```

## 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| epochs | 10 | 전체 데이터 반복 횟수 |
| batch-size | 32 | 배치 크기 |
| learning-rate | 0.0001 | 학습률 |
| weight-decay | 0.01 | L2 정규화 |

## Fine-tuning 전략

1. **Feature Extraction**: Backbone은 고정, Classifier만 학습
2. **작은 Learning Rate**: 사전 학습 가중치 보존
3. **Early Stopping**: Validation Loss 기반

## 학습 모니터링

SageMaker 콘솔 또는 CloudWatch에서 실시간 모니터링:
- Training Loss
- Validation Loss
- Validation Accuracy

## 체크포인트

- [ ] Estimator 설정 완료
- [ ] Training Job 실행
- [ ] 학습 완료 확인
- [ ] 모델 아티팩트 S3 저장 확인

## 다음 단계

학습이 완료되면 [4. After 평가](04-after-evaluation.md)로 이동합니다.
