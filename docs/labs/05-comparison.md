# 5. 성능 비교 (Before vs After × 3가지 기법)

> 노트북: `5_comparison/compare_results.ipynb`

## 개요

Before 모델과 **3가지 Fine-tuning 기법**의 성능을 종합 비교합니다.
각 기법의 장단점을 분석하고 적합한 기법 선택 기준을 제시합니다.

## 학습 내용

- **Domain Shift 문제**와 Fine-tuning의 효과
- 각 **Fine-tuning 기법의 장단점** 분석
- **적합한 기법 선택 기준** 이해

## 비교 관점

### 1. Before vs After (Domain Adaptation)

```
Before (Pretrained)          After (Fine-tuned)
    기존 데이터셋         →     타겟 데이터 특화
        ~70%                     ~90%
```

### 2. Fine-tuning 기법 비교

| 관점 | Full | Freeze | LoRA |
|------|------|--------|------|
| 성능 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 속도 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 메모리 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 과적합 위험 | 높음 | 낮음 | 낮음 |

## 주요 코드

### 결과 로드

```python
# Before 결과
before_path = PROJECT_ROOT / '2_before_evaluation' / 'before_results.json'
with open(before_path, 'r') as f:
    before = json.load(f)

# After 결과 (다중 기법)
after_path = PROJECT_ROOT / '4_after_evaluation' / 'after_results.json'
with open(after_path, 'r') as f:
    after_data = json.load(f)

after_methods = after_data['methods']  # ['full', 'freeze', 'lora']
after_results = after_data['results']
```

### 종합 비교 테이블

```python
print("=" * 80)
print("  📊 Fine-tuning 효과 종합 비교")
print("=" * 80)

# Before 결과
print(f"Before (FF++): {before['accuracy']*100:.1f}%")

# After 결과 (각 기법별)
for method in after_methods:
    result = after_results[method]
    improvement = result['accuracy']*100 - before['accuracy']*100
    print(f"After ({method.upper()}): {result['accuracy']*100:.1f}% (+{improvement:.1f}%p)")

# 최고 성능 기법
best_method = max(after_methods, key=lambda m: after_results[m]['accuracy'])
print(f"\n🏆 최고 성능: {best_method.upper()}")
```

### 시각화 차트

```python
import matplotlib.pyplot as plt
import numpy as np

# Accuracy 비교 막대 그래프
models = ['Before'] + [f'After\n({m.upper()})' for m in after_methods]
accuracies = [before['accuracy']*100] + [after_results[m]['accuracy']*100 for m in after_methods]

colors = ['#ff6b6b'] + ['#4ecdc4', '#45b7d1', '#96ceb4']
plt.bar(models, accuracies, color=colors)
plt.ylabel('Accuracy (%)')
plt.title('Fine-tuning 기법별 정확도 비교')
plt.axhline(y=70, color='gray', linestyle='--', label='Before 기준')
plt.axhline(y=90, color='green', linestyle='--', label='목표')
plt.show()
```

## 성능 비교표

| 모델 | Accuracy | Precision | Recall | F1 | 개선 |
|------|----------|-----------|--------|----|----|
| **Before (FF++)** | ~70% | ~68% | ~72% | ~70% | - |
| **After (FULL)** | ~90% | ~91% | ~89% | ~90% | +20%p |
| **After (FREEZE)** | ~82% | ~83% | ~81% | ~82% | +12%p |
| **After (LORA)** | ~87% | ~88% | ~86% | ~87% | +17%p |

## 기법 선택 가이드

### 언제 Full Fine-tuning?

- 데이터가 충분할 때 (1000+ 샘플)
- 성능이 가장 중요할 때
- 과적합 방지 기법 적용 가능할 때 (Dropout, Augmentation)

### 언제 Layer Freezing?

- 데이터가 적을 때 (100~500 샘플)
- 빠른 실험이 필요할 때
- 계산 자원이 제한적일 때

### 언제 LoRA?

- 대규모 모델 (LLM, ViT-Large 등)
- 여러 태스크에 동시 적용 시 (Adapter 저장)
- 효율성과 성능 모두 중요할 때

## 효율성 분석

| 기법 | 학습 파라미터 | 성능 | 효율성 (성능/파라미터) |
|------|-------------|------|---------------------|
| Full | 4,000,000 (100%) | 90% | 낮음 |
| Freeze | 2,000 (0.05%) | 82% | 매우 높음 |
| LoRA | 10,000 (0.25%) | 87% | 높음 |

> LoRA는 적은 파라미터로 높은 성능을 달성하여 효율성이 뛰어남

## 결론

### 1. Domain Shift 문제 해결
- **Before** (Pretrained): ~70%
- **After** (Fine-tuned): ~90%
- Fine-tuning으로 **+20%p 향상!**

### 2. Fine-tuning 기법 선택 가이드
| 상황 | 추천 기법 |
|------|----------|
| 성능 최우선 | Full Fine-tuning |
| 빠른 실험 / 적은 데이터 | Layer Freezing |
| 대규모 모델 / 효율성 | LoRA |

### 3. 이 워크샵에서는...
- **Full Fine-tuning**이 가장 높은 성능 달성
- **LoRA**도 적은 파라미터로 좋은 성능 달성 (효율적)

## 실패 케이스 분석

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

## 체크포인트

- [ ] Before vs After 성능 비교 완료
- [ ] 3가지 기법 비교 차트 생성
- [ ] 최고 성능 기법 확인
- [ ] 기법 선택 기준 이해

## 다음 단계

비교 분석이 완료되면 [6. 데모 배포](06-demo.md)로 이동합니다.
최고 성능 모델을 배포하고 실시간 데모를 실행합니다.
