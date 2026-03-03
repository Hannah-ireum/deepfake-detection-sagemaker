# 4. After 평가

> 노트북: `4_after_evaluation/evaluate_after.ipynb`

## 개요

Fine-tuning 후 모델의 한국인 얼굴 탐지 성능을 평가하고 **Model Registry에 등록**합니다.

## 학습 내용

- Fine-tuned 모델 로드
- 동일 테스트셋으로 평가
- 성능 개선 확인
- **SageMaker Model Registry 등록**

## Model Registry란?

모델의 버전을 관리하고, 승인 프로세스를 거쳐 배포하는 중앙 저장소입니다.

### 장점

- 모델 버전 관리
- 메타데이터 저장 (정확도, 학습일 등)
- 승인 워크플로우 (PendingManualApproval → Approved)
- 배포 이력 추적

### 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                  Model Registry Flow                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │Fine-tune│───▶│  평가       │───▶│  Registry   │    │
│  │ Model   │    │ (85%+ ?)   │    │  등록       │    │
│  └─────────┘    └─────────────┘    └─────────────┘    │
│                       │                    │            │
│                       │ No                 ▼            │
│                       ▼            ┌─────────────┐     │
│                    건너뜀          │  승인 대기   │     │
│                                    └──────┬──────┘     │
│                                           ▼            │
│                                    ┌─────────────┐     │
│                                    │   배포      │     │
│                                    └─────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 주요 코드

### Fine-tuned 모델 로드

```python
import torch
from sagemaker.pytorch import PyTorchModel

# S3에서 모델 아티팩트 다운로드
model_path = estimator.model_data

# 로컬에서 모델 로드
model = create_model()
model.load_state_dict(torch.load('model/model.pth'))
model.eval()
```

### 성능 평가

```python
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 테스트 데이터로 추론
after_results = evaluate_model(model_after, test_loader, device)

print(f"Accuracy:  {after_results['accuracy']*100:.1f}%")
print(f"Precision: {after_results['precision']*100:.1f}%")
print(f"Recall:    {after_results['recall']*100:.1f}%")
print(f"F1 Score:  {after_results['f1_score']*100:.1f}%")
```

### Model Package Group 생성

```python
MODEL_PACKAGE_GROUP = "deepfake-detection-kodf"
ACCURACY_THRESHOLD = 0.85

sm_client = boto3.client('sagemaker')

# Model Package Group 생성 (처음 한 번만)
sm_client.create_model_package_group(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelPackageGroupDescription="한국인 딥페이크 탐지 모델"
)
```

### 조건부 Model Registry 등록

```python
if after_results['accuracy'] >= ACCURACY_THRESHOLD:
    print(f"✅ 정확도 {after_results['accuracy']*100:.1f}% >= 85% 기준 충족!")

    # PyTorch 모델 정의
    pytorch_model = PyTorchModel(
        model_data=config['model_data'],
        role=config['role'],
        framework_version='2.0.0',
        py_version='py310',
        entry_point='inference.py',
        source_dir='../6_demo'
    )

    # Model Registry 등록
    model_package = pytorch_model.register(
        model_package_group_name=MODEL_PACKAGE_GROUP,
        inference_instances=['ml.g4dn.xlarge', 'ml.m5.large'],
        transform_instances=['ml.m5.large'],
        content_types=['application/json'],
        response_types=['application/json'],
        approval_status='PendingManualApproval',
        description=f"KoDF Fine-tuned (Accuracy: {after_results['accuracy']*100:.1f}%)"
    )

    print(f"✅ Model Package ARN: {model_package.model_package_arn}")

else:
    print(f"❌ 정확도 미달 - Registry 등록 건너뜀")
```

## 예상 결과

Fine-tuning 후 한국인 얼굴 데이터 성능:

| 메트릭 | Before | After | 개선 |
|--------|--------|-------|------|
| Accuracy | ~70% | ~90% | +20% |
| AUC | ~0.75 | ~0.95 | +0.20 |
| Precision | ~0.68 | ~0.91 | +0.23 |
| Recall | ~0.72 | ~0.89 | +0.17 |

## Model Registry 상태

| 상태 | 설명 |
|------|------|
| `PendingManualApproval` | 등록 직후, 승인 대기 |
| `Approved` | 승인 완료, 배포 가능 |
| `Rejected` | 거부됨 |

### 모델 승인하기

```python
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus='Approved'
)
```

또는 SageMaker Console에서 수동 승인 가능합니다.

## 체크포인트

- [ ] Fine-tuned 모델 로드 완료
- [ ] 테스트 데이터 추론 완료
- [ ] 성능 메트릭 기록 완료
- [ ] Before 대비 개선 확인
- [ ] **Model Registry 등록 완료**

## 다음 단계

After 평가가 완료되면 [5. 성능 비교](05-comparison.md)로 이동합니다.
