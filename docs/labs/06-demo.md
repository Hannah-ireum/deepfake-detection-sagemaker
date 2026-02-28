# 6. 데모 배포

> 노트북: `6_demo/deploy_and_demo.ipynb`

## 개요

Fine-tuned 모델을 SageMaker Endpoint로 배포하고 Gradio UI로 데모합니다.

## 학습 내용

- SageMaker Endpoint 배포
- 실시간 추론 API 호출
- Gradio 데모 UI 구축

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

### Endpoint 배포 코드

```python
from sagemaker.pytorch import PyTorchModel

# 모델 정의
model = PyTorchModel(
    model_data=estimator.model_data,
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='6_demo'
)

# Endpoint 배포
predictor = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1,
    endpoint_name='deepfake-detection-endpoint'
)
```

### Inference 스크립트 (inference.py)

```python
import torch
import json
import base64
from io import BytesIO
from PIL import Image

def model_fn(model_dir):
    """모델 로드"""
    model = create_model()
    model.load_state_dict(torch.load(f'{model_dir}/model.pth'))
    model.eval()
    return model

def input_fn(request_body, content_type):
    """입력 전처리"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return transform(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """추론"""
    with torch.no_grad():
        output = model(input_data)
        prob = torch.softmax(output, dim=1)
    return prob

def output_fn(prediction, accept):
    """출력 포맷"""
    fake_prob = prediction[0][1].item()
    result = {
        'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL',
        'confidence': fake_prob if fake_prob > 0.5 else 1 - fake_prob,
        'fake_probability': fake_prob
    }
    return json.dumps(result)
```

## Part 2: Gradio 데모 UI

### Gradio 앱 코드 (gradio_app.py)

```python
import gradio as gr
import boto3
import base64
import json

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = 'deepfake-detection-endpoint'

def detect_deepfake(image):
    # 이미지를 base64로 인코딩
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Endpoint 호출
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps({'image': img_str})
    )

    result = json.loads(response['Body'].read().decode())

    # 결과 포맷팅
    label = result['prediction']
    confidence = result['confidence']

    if label == 'FAKE':
        return f"🚨 FAKE 탐지 (확신도: {confidence:.1%})"
    else:
        return f"✅ REAL (확신도: {confidence:.1%})"

# Gradio 인터페이스
demo = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Image(type="pil", label="이미지 업로드"),
    outputs=gr.Textbox(label="탐지 결과"),
    title="딥페이크 탐지 데모",
    description="이미지를 업로드하면 딥페이크 여부를 판별합니다.",
    examples=[
        ["examples/real_sample.jpg"],
        ["examples/fake_sample.jpg"]
    ]
)

demo.launch(share=True)
```

### 데모 실행

```bash
cd 6_demo
python gradio_app.py
```

## 비용 관리

### Endpoint 삭제 (중요!)

실습 완료 후 반드시 Endpoint를 삭제하세요:

```python
predictor.delete_endpoint()
print("Endpoint deleted successfully!")
```

## 체크포인트

- [ ] SageMaker Endpoint 배포 완료
- [ ] Endpoint 추론 테스트
- [ ] Gradio 데모 실행
- [ ] 실시간 탐지 확인
- [ ] **Endpoint 삭제 완료**

## 실습 완료

축하합니다! 모든 실습을 완료했습니다.

### 배운 내용 정리

1. 딥페이크 탐지 모델의 도메인 특화 Fine-tuning
2. SageMaker Training Job 활용
3. 성능 평가 및 비교 분석
4. 실시간 추론 Endpoint 배포
5. Gradio를 활용한 데모 UI 구축
