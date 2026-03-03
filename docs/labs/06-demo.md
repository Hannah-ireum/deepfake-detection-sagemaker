# 6. 데모 배포

> 노트북: `6_demo/deploy_and_demo.ipynb`

## 개요

Fine-tuned 모델을 SageMaker Endpoint로 배포하고 Gradio UI로 데모합니다.

## 학습 내용

- SageMaker Endpoint 배포
- 실시간 추론 API 호출
- Gradio 데모 UI (노트북 내 실행)

## Part 1: SageMaker Endpoint 배포

### 아키텍처

```
┌─────────────────────────────────────────────────┐
│              SageMaker Endpoint                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│  │ Gradio  │───▶│Endpoint │───▶│ Model   │    │
│  │   UI    │    │  API    │    │Container│    │
│  └─────────┘    └─────────┘    └─────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Endpoint 이름 생성

여러 사용자가 동시에 실습할 때 충돌을 방지하기 위해 **타임스탬프**를 추가합니다.

```python
from datetime import datetime

# 고유한 Endpoint 이름 생성
ENDPOINT_NAME = f"deepfake-detector-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
print(f"Endpoint Name: {ENDPOINT_NAME}")
# 예: deepfake-detector-20240304-143052
```

### Endpoint 배포 코드

```python
from sagemaker.pytorch import PyTorchModel

# 모델 정의
pytorch_model = PyTorchModel(
    model_data=config['model_data'],
    role=role,
    entry_point='inference.py',
    source_dir='.',
    framework_version='2.0.0',
    py_version='py310'
)

# Endpoint 배포 (약 5-10분 소요)
print("Endpoint 배포 중...")
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    endpoint_name=ENDPOINT_NAME
)

print(f"✅ Endpoint 배포 완료: {predictor.endpoint_name}")
```

## Part 2: Gradio 데모 UI

### 노트북 내에서 직접 실행

별도 Python 파일 실행 없이 노트북에서 바로 Gradio를 실행합니다.

```python
import gradio as gr
import boto3
from io import BytesIO

runtime = boto3.client('sagemaker-runtime')

def detect_deepfake(image):
    """딥페이크 탐지 함수"""
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
    description="이미지를 업로드하면 딥페이크 여부를 판별합니다."
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

### Endpoint 삭제 (중요!)

실습 완료 후 **반드시** Endpoint를 삭제하세요:

```python
# 주석 해제 후 실행
predictor.delete_endpoint()
print(f"✅ Endpoint '{ENDPOINT_NAME}' 삭제 완료!")
```

### 비용 참고

| 상태 | 시간당 비용 |
|------|-------------|
| Endpoint 실행 중 | ~$0.74/hr |
| Endpoint 삭제 후 | $0.00 |

## 체크포인트

- [ ] Endpoint 이름 확인 (타임스탬프 포함)
- [ ] SageMaker Endpoint 배포 완료
- [ ] Gradio 데모 실행
- [ ] 이미지 업로드 후 탐지 결과 확인
- [ ] **⚠️ Endpoint 삭제 완료**

## 실습 완료

축하합니다! 모든 실습을 완료했습니다.

### 배운 내용 정리

1. ✅ 딥페이크 탐지 모델의 도메인 특화 Fine-tuning
2. ✅ SageMaker Experiments로 실험 추적
3. ✅ Model Registry로 모델 버전 관리
4. ✅ Spot Instance로 비용 절감
5. ✅ 실시간 추론 Endpoint 배포
6. ✅ Gradio를 활용한 데모 UI 구축
