# 6. 영상 데모 배포

> 노트북: `6_demo/deploy_and_demo.ipynb`

## 개요

**최고 성능 Fine-tuned 모델**을 SageMaker Endpoint로 배포하고 **숏폼 영상 분석 데모**를 실행합니다.

## 학습 내용

- SageMaker Endpoint 배포 과정
- **영상 분석 파이프라인** (프레임 추출 → CNN 분석 → 결과 종합)
- Gradio 영상/이미지 데모 UI 구축

## 영상 딥페이크 탐지 파이프라인

<img src="../images/00_video_detection_pipeline.png" alt="Video Detection Pipeline" width="1000">

## 배포 아키텍처

<img src="../images/03_deployment_architecture.png" alt="Endpoint Deployment" width="1000">

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

## Part 2: 영상 분석 데모 UI

### Gradio란?

- ML 모델을 위한 **웹 UI 프레임워크**
- Python 코드 몇 줄로 데모 인터페이스 생성
- `share=True`로 외부 공유 가능한 URL 생성

### 영상 분석 흐름

```
숏폼 영상 업로드
      ↓
프레임 추출 (3fps)
      ↓
각 프레임 → Endpoint 호출
      ↓
다수결 투표로 최종 판정
      ↓
결과 표시 (REAL/FAKE + 상세 정보)
```

### 핵심 코드: 프레임 추출

```python
import cv2

def extract_frames(video_path, fps=3):
    """영상에서 프레임 추출 (3fps 기본)"""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > fps else 1

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames
```

### 핵심 코드: 다수결 투표

```python
def detect_deepfake_video(video):
    # 프레임 추출
    frames = extract_frames(video, fps=3)

    # 각 프레임 분석
    fake_count = 0
    for frame in frames:
        prediction, confidence = analyze_single_frame(frame)
        if prediction == 'FAKE':
            fake_count += 1

    # 다수결 투표
    fake_ratio = fake_count / len(frames)
    final_prediction = "FAKE" if fake_ratio > 0.5 else "REAL"

    return final_prediction
```

### 데모 UI (탭 구성)

- **🎬 영상 분석**: 숏폼 영상 업로드 → 프레임 분석 → 결과 종합
- **🖼️ 이미지 분석**: 단일 이미지 분석

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
- [ ] **영상 업로드 후 탐지 결과 확인**
- [ ] 이미지 업로드 후 탐지 결과 확인
- [ ] **⚠️ Endpoint 삭제 완료**

## Workshop 완료!

축하합니다! **숏폼 딥페이크 영상 판별** 워크샵을 완료했습니다.

### 배운 내용 정리

| 단계 | 내용 |
|------|------|
| **1. 데이터 준비** | S3에서 데이터 다운로드, 구조 이해 |
| **2. Before 평가** | ImageNet Pretrained 모델의 한계 확인 (~50%) |
| **3. Fine-tuning** | Full, Freeze, LoRA 세 가지 기법 비교 |
| **4. After 평가** | Fine-tuned 모델 성능 확인 (~85%) |
| **5. 성능 비교** | 기법별 장단점 분석 |
| **6. 영상 데모** | SageMaker Endpoint + 영상 분석 UI |

### 영상 딥페이크 탐지 파이프라인

```
숏폼 영상 → 프레임 추출(3fps) → CNN 분석 → 다수결 투표 → REAL/FAKE
```

### 핵심 학습 포인트

1. **태스크 전이의 한계**: 객체 분류 모델은 딥페이크 탐지에 적합하지 않음
2. **Fine-tuning 효과**: 딥페이크 데이터로 학습 시 성능 크게 향상 (+35%p)
3. **프레임 기반 분석**: 숏폼 영상에 효과적이고 빠름
4. **SageMaker 활용**: Experiments, Model Registry, Spot Instance

### 프로덕션 적용 가이드

| 항목 | 워크샵 (데모) | 프로덕션 |
|------|-------------|----------|
| 추론 방식 | 실시간 Endpoint | 비동기 Inference |
| 스케일링 | 고정 1대 | Auto Scaling, Scale to Zero |
| 콜백 | 즉시 응답 | SNS 알림 |

### 다음 단계 (심화)

- **시공간 모델**: ViViT, TimeSformer로 시간적 패턴 분석
- **데이터 플라이휠**: 유저 투표 데이터로 모델 재학습 (RLHF)
- **비동기 추론**: Scale to Zero로 비용 최적화

감사합니다!
