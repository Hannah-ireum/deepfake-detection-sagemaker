# 2. Before 평가

> 노트북: `2_before_evaluation/evaluate_before.ipynb`

## 개요

Fine-tuning 전, ImageNet 사전 학습 모델의 딥페이크 탐지 성능을 평가합니다.

## 학습 내용

- 사전 학습 모델 로드
- 추론 파이프라인 구성
- 성능 메트릭 계산

## ImageNet Pretrained 모델

ImageNet은 1,400만장의 이미지로 1,000개 객체를 분류하는 대규모 데이터셋입니다.
이 데이터로 학습된 모델은 **객체 분류**에 특화되어 있으며, **딥페이크 탐지 학습은 되어 있지 않습니다.**

### 모델 아키텍처

- **Backbone**: EfficientNet-B0
- **Task**: Binary Classification (Real vs Fake)
- **Input**: 224x224 RGB 이미지
- **Pretrained**: ImageNet (객체 분류용)

## 주요 코드

### 모델 로드

```python
import torch
import timm

# ImageNet Pretrained 모델 로드
# pretrained=True: ImageNet 가중치 사용 (객체 분류용, 딥페이크 학습 안 됨)
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model.eval()

print("ImageNet Pretrained 모델 로드 완료")
print("⚠️ 이 모델은 객체 분류용으로 학습되어 딥페이크 탐지 성능이 낮습니다.")
```

### 추론 함수

```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)

    return prob[0][1].item()  # fake 확률
```

### 성능 평가

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 테스트 데이터로 평가
predictions = []
for img_path in test_images:
    pred = predict(img_path)
    predictions.append(1 if pred > 0.5 else 0)

# 메트릭 계산
accuracy = accuracy_score(test_labels, predictions)

print(f"Before Fine-tuning")
print(f"Accuracy: {accuracy:.2%}")
```

## 예상 결과

ImageNet Pretrained 모델의 딥페이크 탐지 성능:

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| Accuracy | ~50% | 무작위 수준 |
| Precision | ~50% | 무작위 수준 |
| Recall | ~50% | 무작위 수준 |
| F1 Score | ~50% | 무작위 수준 |

## 분석

ImageNet Pretrained 모델이 딥페이크 탐지를 못하는 이유:

| 구분 | ImageNet 학습 | 딥페이크 탐지 |
|------|--------------|--------------|
| **목적** | 객체 분류 (고양이, 개, 자동차 등) | 합성 여부 판별 (Real vs Fake) |
| **특징** | 전체적인 형태, 색상, 텍스처 | 미세한 합성 흔적, 아티팩트 |
| **학습 데이터** | 일반 사물 이미지 | 얼굴 딥페이크 이미지 |

> 💡 **결론**: 딥페이크 탐지를 위해서는 해당 태스크에 맞는 **Fine-tuning**이 필수입니다!

## 체크포인트

- [ ] ImageNet Pretrained 모델 로드 완료
- [ ] 테스트 데이터 추론 완료
- [ ] 성능 메트릭 기록 완료 (~50%)

## 다음 단계

Before 성능을 기록했으면 [3. Fine-tuning](03-fine-tuning.md)으로 이동합니다.
