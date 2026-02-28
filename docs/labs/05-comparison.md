# 5. 성능 비교

> 노트북: `5_comparison/compare_results.ipynb`

## 개요

Before와 After 모델의 성능을 시각적으로 비교합니다.

## 학습 내용

- 성능 메트릭 시각화
- ROC Curve 비교
- 실패 케이스 분석

## 주요 코드

### 메트릭 비교 차트

```python
import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
before_scores = [0.70, 0.68, 0.72, 0.70]
after_scores = [0.90, 0.91, 0.89, 0.90]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, before_scores, width, label='Before', color='#ff7f7f')
bars2 = ax.bar(x + width/2, after_scores, width, label='After', color='#7fbf7f')

ax.set_ylabel('Score')
ax.set_title('Before vs After Fine-tuning')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1)

# 값 표시
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### ROC Curve 비교

```python
from sklearn.metrics import roc_curve, auc

# Before ROC
fpr_before, tpr_before, _ = roc_curve(test_labels, probs_before)
auc_before = auc(fpr_before, tpr_before)

# After ROC
fpr_after, tpr_after, _ = roc_curve(test_labels, probs_after)
auc_after = auc(fpr_after, tpr_after)

plt.figure(figsize=(8, 8))
plt.plot(fpr_before, tpr_before, 'r-', label=f'Before (AUC={auc_before:.3f})')
plt.plot(fpr_after, tpr_after, 'g-', label=f'After (AUC={auc_after:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

### 개선율 요약

```python
improvements = {
    'Accuracy': (0.90 - 0.70) / 0.70 * 100,
    'AUC': (0.95 - 0.75) / 0.75 * 100,
    'Precision': (0.91 - 0.68) / 0.68 * 100,
    'Recall': (0.89 - 0.72) / 0.72 * 100
}

print("=" * 40)
print("        성능 개선율 요약")
print("=" * 40)
for metric, improvement in improvements.items():
    print(f"{metric:12}: +{improvement:.1f}%")
print("=" * 40)
```

## 시각화 결과

### 성능 비교표

| 메트릭 | Before | After | 개선율 |
|--------|--------|-------|--------|
| Accuracy | 70% | 90% | +28.6% |
| AUC | 0.75 | 0.95 | +26.7% |
| Precision | 68% | 91% | +33.8% |
| Recall | 72% | 89% | +23.6% |

### 실패 케이스 분석

Fine-tuning 후에도 어려운 케이스:

1. **고품질 Face Swap**
   - 매우 정교한 합성
   - 아티팩트가 거의 없음

2. **저해상도 이미지**
   - 세부 특징 파악 어려움
   - 압축 아티팩트와 혼동

3. **극단적 포즈/조명**
   - 학습 데이터에 적은 케이스
   - 일반화 한계

## 결론

- Fine-tuning으로 **20%p 이상** 성능 향상
- 한국인 얼굴에 특화된 모델 확보
- 추가 개선을 위해 더 많은 데이터 필요

## 체크포인트

- [ ] 메트릭 비교 차트 생성
- [ ] ROC Curve 비교
- [ ] 개선율 계산
- [ ] 실패 케이스 분석

## 다음 단계

비교 분석이 완료되면 [6. 데모 배포](06-demo.md)로 이동합니다.
