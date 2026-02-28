# 2. Before 평가

> 노트북: `2_before_evaluation/evaluate_before.ipynb`

## 개요

Fine-tuning 전, FaceForensics++ 사전 학습 모델의 한국인 얼굴 탐지 성능을 평가합니다.

## 학습 내용

- 사전 학습 모델 로드
- 추론 파이프라인 구성
- 성능 메트릭 계산

## FaceForensics++ 모델

FaceForensics++는 대표적인 딥페이크 탐지 벤치마크입니다. 주로 서양인 얼굴 데이터로 학습되었습니다.

### 모델 아키텍처

- **Backbone**: EfficientNet-B4
- **Task**: Binary Classification (Real vs Fake)
- **Input**: 224x224 RGB 이미지

## 주요 코드

### 모델 로드

```python
import torch
import timm

# 사전 학습 모델 로드
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)

# FF++ 가중치 로드
model.load_state_dict(torch.load('weights/ff_pretrained.pth'))
model.eval()
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# 테스트 데이터로 평가
predictions = []
for img_path in test_images:
    pred = predict(img_path)
    predictions.append(1 if pred > 0.5 else 0)

# 메트릭 계산
accuracy = accuracy_score(test_labels, predictions)
auc = roc_auc_score(test_labels, predictions)

print(f"Before Fine-tuning")
print(f"Accuracy: {accuracy:.2%}")
print(f"AUC: {auc:.4f}")
```

## 예상 결과

한국인 얼굴 데이터에서의 Before 성능:

| 메트릭 | 값 |
|--------|-----|
| Accuracy | ~70% |
| AUC | ~0.75 |
| Precision | ~0.68 |
| Recall | ~0.72 |

## 분석

서양인 얼굴로 학습된 모델이 한국인 얼굴에서 성능이 저하되는 이유:
- 얼굴 특징(눈, 코, 입 비율)의 차이
- 피부톤 및 텍스처 차이
- 메이크업 스타일 차이

## 체크포인트

- [ ] 사전 학습 모델 로드 완료
- [ ] 테스트 데이터 추론 완료
- [ ] 성능 메트릭 기록 완료

## 다음 단계

Before 성능을 기록했으면 [3. Fine-tuning](03-fine-tuning.md)으로 이동합니다.
