# 아키텍처 상세

## 영상 딥페이크 탐지 파이프라인

<img src="../images/00_video_detection_pipeline.png" alt="Video Detection Pipeline" width="1000">

**숏폼 영상 분석 흐름:**
1. **영상 업로드**: 사용자가 숏폼 영상(15~60초) 업로드
2. **프레임 추출**: 3fps로 프레임 샘플링
3. **얼굴 탐지**: MTCNN으로 각 프레임에서 얼굴 영역 추출
4. **CNN 분류**: Fine-tuned 모델로 각 프레임 REAL/FAKE 판별
5. **결과 종합**: 다수결 투표로 최종 영상 판정

## 전체 워크샵 아키텍처

<img src="../images/01_overall_architecture.png" alt="Overall Architecture" width="1000">

## 모델 아키텍처

### EfficientNet-B4 기반 분류기

**모델 흐름:**
1. **Input Image** (224x224x3)
2. **EfficientNet-B4 Backbone** (Frozen) - Stem Conv → MBConv Blocks → Head Conv
3. **Global Average Pooling**
4. **Classifier** (Trainable) - Dropout(0.4) → Linear(1792→2)
5. **Softmax Output** - [Real, Fake]

### 레이어 구성

| 레이어 | 파라미터 수 | 학습 여부 |
|--------|-------------|-----------|
| Stem | ~5K | Frozen |
| MBConv 1-7 | ~17M | Frozen |
| Head | ~2M | Frozen |
| Classifier | ~3.5K | **Trainable** |

## 데이터 파이프라인

### 전처리 흐름

**전처리 순서:**
1. Raw Video
2. Frame Extraction (fps=1)
3. Face Detection (MTCNN)
4. Face Alignment & Crop
5. Resize (224x224)
6. Normalize (ImageNet stats)
7. Training Ready Tensor

### 데이터 증강

학습 시 적용되는 증강:

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
])
```

## 추론 파이프라인

### 실시간 추론 흐름

<img src="../images/03_deployment_architecture.png" alt="Deployment Architecture" width="800">

**SageMaker Endpoint 함수:**
- `input_fn()`: Base64 decode, Preprocess
- `predict_fn()`: Model inference
- `output_fn()`: Format response

**응답 예시:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.95,
  "fake_probability": 0.95
}
```

## 보안 고려사항

### IAM 권한

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateEndpoint",
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket/*"
    }
  ]
}
```

### VPC 설정 (선택)

프로덕션 환경에서는 VPC 내에서 실행 권장:

```python
estimator = PyTorch(
    ...
    subnets=['subnet-xxxxx'],
    security_group_ids=['sg-xxxxx']
)
```

## 프로덕션 아키텍처 (비동기 추론)

실제 게임 플랫폼 등에 적용 시, **비동기 추론(Async Inference)** 아키텍처를 권장합니다:

<img src="../images/06_production_architecture.png" alt="Production Architecture" width="1000">

### 비동기 추론의 장점

| 항목 | 실시간 Endpoint | 비동기 Endpoint |
|------|----------------|-----------------|
| 응답 시간 | 즉시 (타임아웃 위험) | 비동기 (SNS 콜백) |
| 스케일링 | Auto Scaling | Scale to Zero 가능 |
| 비용 | 항상 실행 | 트래픽 없으면 $0 |
| 적합 워크로드 | 짧은 추론 | 영상 분석 등 긴 추론 |

### 비동기 추론 코드 예시

```python
import boto3

sagemaker_runtime = boto3.client('sagemaker-runtime')

def analyze_video_async(endpoint_name, s3_input_uri):
    """비동기 영상 분석 요청"""
    response = sagemaker_runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=s3_input_uri  # S3에 업로드된 영상 경로
    )

    inference_id = response['InferenceId']
    output_location = response['OutputLocation']

    return inference_id, output_location
```

### SNS 콜백 설정

분석 완료 시 게임 서버로 알림:

```python
# Endpoint 생성 시 SNS 토픽 설정
async_config = AsyncInferenceConfig(
    output_path=f"s3://{bucket}/async-output/",
    notification_config=NotificationConfig(
        success_topic="arn:aws:sns:region:account:success-topic",
        error_topic="arn:aws:sns:region:account:error-topic"
    )
)
```

## 심화: 시공간 분석 모델

본 워크샵은 **프레임 기반 분석** 방식입니다. 더 정교한 시공간 분석이 필요한 경우:

### 시공간 모델 비교

| 모델 | 입력 형태 | 특징 | 적합한 경우 |
|------|----------|------|------------|
| **ViViT** | [B,C,T,H,W] | Video Vision Transformer | 시간적 패턴 분석 |
| **TimeSformer** | [B,C,T,H,W] | 시공간 어텐션 분리 | 긴 영상 분석 |
| **X3D** | [B,C,T,H,W] | 효율적 3D CNN | 실시간 분석 |

### 프레임 기반 vs 시공간 분석

| 구분 | 프레임 기반 (본 워크샵) | 시공간 분석 |
|------|----------------------|------------|
| **탐지 대상** | 공간적 아티팩트 | 시간적 불일치 |
| **모델** | 2D CNN (EfficientNet) | 3D CNN, Video Transformer |
| **입력** | [B, C, H, W] | [B, C, **T**, H, W] |
| **GPU 메모리** | 낮음 | 높음 |
| **적합 영상** | 숏폼 (15~60초) | 긴 영상, 정교한 딥페이크 |

### 시공간 분석 탐지 특징

시공간 모델은 다음과 같은 **시간적 불일치**를 탐지할 수 있습니다:

- 눈 깜빡임의 부자연스러운 주기
- 입술 움직임과 오디오 싱크 불일치
- 프레임 간 미세한 떨림 (Flickering)
- 표정 변화의 비연속성

### LoRA 적용 가능

시공간 모델에도 LoRA 기법을 동일하게 적용할 수 있습니다:

```python
# TimeSformer의 Temporal Attention에 LoRA 적용
class TemporalLoRALayer(nn.Module):
    def __init__(self, temporal_attention, rank=8):
        super().__init__()
        self.temporal_attention = temporal_attention

        # Q, V에 LoRA 어댑터 추가
        self.lora_Q = LoRAAdapter(temporal_attention.query, rank)
        self.lora_V = LoRAAdapter(temporal_attention.value, rank)
```

## 데이터 플라이휠 (비즈니스 확장)

게임 플랫폼에서 유저 투표 데이터를 활용한 모델 고도화:

```
유저 투표 데이터 수집
        ↓
    데이터 검증
        ↓
   S3에 적재
        ↓
SageMaker Pipelines
  (주기적 재학습)
        ↓
   모델 업데이트
        ↓
  탐지 성능 향상
        ↓
    (반복)
```

이 **데이터 플라이휠**을 통해 시간이 지날수록 탐지 성능이 향상되며, 경쟁사가 쉽게 따라올 수 없는 비즈니스 해자(Moat)가 됩니다.
