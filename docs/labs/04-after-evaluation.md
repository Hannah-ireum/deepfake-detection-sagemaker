# 4. After 평가

> 노트북: `4_after_evaluation/evaluate_after.ipynb`

## 개요

Fine-tuning 후 모델의 한국인 얼굴 탐지 성능을 평가합니다.

## 학습 내용

- Fine-tuned 모델 로드
- 동일 테스트셋으로 평가
- 성능 개선 확인

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
predictions = []
probabilities = []

for img_path in test_images:
    prob = predict(img_path)
    probabilities.append(prob)
    predictions.append(1 if prob > 0.5 else 0)

# 메트릭 계산
metrics = {
    'accuracy': accuracy_score(test_labels, predictions),
    'auc': roc_auc_score(test_labels, probabilities),
    'precision': precision_score(test_labels, predictions),
    'recall': recall_score(test_labels, predictions),
    'f1': f1_score(test_labels, predictions)
}

print("After Fine-tuning")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

### Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (After Fine-tuning)')
plt.show()
```

## 예상 결과

Fine-tuning 후 한국인 얼굴 데이터 성능:

| 메트릭 | Before | After | 개선 |
|--------|--------|-------|------|
| Accuracy | ~70% | ~90% | +20% |
| AUC | ~0.75 | ~0.95 | +0.20 |
| Precision | ~0.68 | ~0.91 | +0.23 |
| Recall | ~0.72 | ~0.89 | +0.17 |

## 분석 포인트

### 성능 개선 이유
1. 한국인 얼굴 특징 학습
2. 도메인 특화 미세 조정
3. 데이터 분포 매칭

### 여전히 어려운 케이스
- 고품질 딥페이크
- 저해상도 이미지
- 극단적 조명 조건

## 체크포인트

- [ ] Fine-tuned 모델 로드 완료
- [ ] 테스트 데이터 추론 완료
- [ ] 성능 메트릭 기록 완료
- [ ] Before 대비 개선 확인

## 다음 단계

After 평가가 완료되면 [5. 성능 비교](05-comparison.md)로 이동합니다.
