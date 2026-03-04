# 6. 데모 배포

> 노트북: `6_demo/deploy_and_demo.ipynb`

## 개요

**최고 성능 Fine-tuned 모델**을 SageMaker Endpoint로 배포하고 Gradio UI로 데모합니다.

## 학습 내용

- SageMaker Endpoint 배포 과정
- 실시간 추론 아키텍처
- Gradio 데모 UI 구축

## 배포 아키텍처

![SageMaker Endpoint Deployment](../images/03_deployment_architecture.png)

## Part 1: 최고 성능 모델 배포

### 배포할 모델 선택

config.json에서 **최고 성능 기법**의 모델을 자동 선택합니다:

```python
# 최고 성능 기법의 모델 사용
best_method = config.get('best_method', 'full')
training_results = config.get('training_results', {})

if training_results and best_method in training_results:
    model_data = training_results[best_method]['model_data']
else:
    model_data = config['model_data']

print(f"배포할 모델: {best_method.upper()} Fine-tuned")
print(f"모델 경로: {model_data}")
```

### Endpoint 이름 생성

여러 사용자가 동시에 실습할 때 충돌을 방지하기 위해 **타임스탬프**를 추가합니다.

```python
from datetime import datetime

# 고유한 Endpoint 이름 생성 (기법명 포함)
ENDPOINT_NAME = f"deepfake-{best_method}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
print(f"Endpoint Name: {ENDPOINT_NAME}")
# 예: deepfake-full-20240304-143052
```

### 배포 과정 이해

1. **PyTorchModel 생성**: S3의 model.tar.gz와 inference.py 지정
2. **deploy() 호출**: SageMaker가 자동으로:
   - EC2 인스턴스 프로비저닝 (ml.g4dn.xlarge)
   - Docker 컨테이너 시작
   - 모델 로드 및 웜업
3. **Endpoint 생성**: HTTPS 엔드포인트 URL 제공

### Endpoint 배포 코드

```python
from sagemaker.pytorch import PyTorchModel

# 모델 정의
pytorch_model = PyTorchModel(
    model_data=model_data,  # 최고 성능 기법의 모델
    role=role,
    entry_point='inference.py',
    source_dir='.',
    framework_version='2.0.0',
    py_version='py310'
)

# Endpoint 배포 (약 5-10분 소요)
print(f"배포할 모델: {best_method.upper()} Fine-tuned")
print("Endpoint 배포 중...")
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    endpoint_name=ENDPOINT_NAME
)

print(f"✅ Endpoint 배포 완료: {predictor.endpoint_name}")
```

### inference.py 역할

```python
def model_fn(model_dir):     # 모델 로드
def input_fn(data, type):    # 입력 전처리
def predict_fn(data, model): # 추론 실행
def output_fn(pred, type):   # 출력 후처리
```

## Part 2: Gradio 데모 UI

### Gradio란?

- ML 모델을 위한 **웹 UI 프레임워크**
- Python 코드 몇 줄로 데모 인터페이스 생성
- `share=True`로 외부 공유 가능한 URL 생성

### 데모 흐름

```
사용자가 이미지 업로드
      ↓
이미지 → JPEG 바이트로 변환
      ↓
SageMaker Endpoint 호출
      ↓
결과 (REAL/FAKE + 확신도) 표시
```

### 노트북 내에서 직접 실행

```python
import gradio as gr
import boto3
from io import BytesIO

runtime = boto3.client('sagemaker-runtime')

def detect_deepfake(image):
    """딥페이크 탐지 함수"""
    if image is None:
        return "이미지를 업로드해주세요."

    # 이미지를 바이트로 변환
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # Endpoint 호출
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/x-image',
        Body=img_bytes
    )

    result = json.loads(response['Body'].read().decode())

    prediction = result.get('prediction', 'Unknown')
    confidence = result.get('confidence', 0)

    if prediction == 'FAKE':
        return f"🚨 FAKE 탐지!\n확신도: {confidence:.1%}"
    else:
        return f"✅ REAL\n확신도: {confidence:.1%}"

# Gradio 인터페이스
demo = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Image(type="pil", label="이미지 업로드"),
    outputs=gr.Textbox(label="탐지 결과"),
    title="🎭 딥페이크 탐지 데모",
    description="이미지를 업로드하면 딥페이크 여부를 판별합니다.\n(KoDF Fine-tuned 모델 사용)"
)

# 실행 (share=True로 공개 URL 생성)
demo.launch(share=True)
```

### 데모 UI 화면

```
┌────────────────────────────────────────┐
│     🎭 딥페이크 탐지 데모              │
├────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐   │
│  │              │  │              │   │
│  │   이미지     │  │   탐지 결과  │   │
│  │   업로드     │  │              │   │
│  │              │  │  ✅ REAL     │   │
│  │              │  │  확신도: 95% │   │
│  └──────────────┘  └──────────────┘   │
│                                        │
│         [Submit]  [Clear]              │
└────────────────────────────────────────┘
```

## 비용 관리

### Endpoint 비용

| 인스턴스 | 시간당 비용 | 비고 |
|---------|-----------|------|
| ml.g4dn.xlarge | ~$0.74 | GPU (추론용) |

> ⚠️ **중요**: Endpoint는 **실행 중인 동안 계속 과금**됩니다!

### Endpoint 삭제 (중요!)

실습 완료 후 **반드시** Endpoint를 삭제하세요:

```python
# 주석 해제 후 실행
predictor.delete_endpoint()
print(f"✅ Endpoint '{ENDPOINT_NAME}' 삭제 완료!")
print("더 이상 비용이 발생하지 않습니다.")
```

## 체크포인트

- [ ] 배포할 모델 확인 (최고 성능 기법)
- [ ] Endpoint 이름 확인 (타임스탬프 포함)
- [ ] SageMaker Endpoint 배포 완료
- [ ] Gradio 데모 실행
- [ ] 이미지 업로드 후 탐지 결과 확인
- [ ] **⚠️ Endpoint 삭제 완료**

## Workshop 완료!

축하합니다! 모든 실습을 완료했습니다.

### 배운 내용 정리

| 단계 | 내용 |
|------|------|
| **1. 데이터 준비** | S3에서 데이터 다운로드, 구조 이해 |
| **2. Before 평가** | Pretrained 모델의 한계 확인 (~70%) |
| **3. Fine-tuning** | Full, Freeze, LoRA 세 가지 기법 비교 |
| **4. After 평가** | Fine-tuned 모델 성능 확인 (~90%) |
| **5. 성능 비교** | 기법별 장단점 분석 |
| **6. 데모 배포** | SageMaker Endpoint + Gradio UI |

### 핵심 학습 포인트

1. **Domain Shift 문제**: Pretrained 모델은 다른 도메인에서 성능 저하
2. **Fine-tuning 효과**: 타겟 도메인 데이터로 학습 시 성능 크게 향상
3. **기법 선택**: 상황에 따라 Full, Freeze, LoRA 중 선택
4. **SageMaker 활용**: Experiments, Model Registry, Spot Instance

### 다음 단계 (심화)

- 더 많은 한국인 데이터로 학습
- 다른 모델 아키텍처 실험 (ViT, ConvNeXt)
- A/B 테스트 및 프로덕션 배포

감사합니다!
