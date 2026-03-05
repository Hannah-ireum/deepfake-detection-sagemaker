# 숏폼 딥페이크 영상 판별 Workshop

## Workshop 소개

**숏폼 영상(틱톡, 릴스, 쇼츠 등)의 딥페이크 여부를 판별하는 AI 모델**을 Fine-tuning하는 실습입니다.

게임 플랫폼, 콘텐츠 검증 서비스 등에서 활용할 수 있는 딥페이크 영상 탐지 모델을 Amazon SageMaker 기반으로 학습하고 배포합니다.

## 영상 딥페이크 탐지 파이프라인

<img src="images/00_video_detection_pipeline.png" alt="Video Detection Pipeline" width="1000">

**숏폼 영상 분석 흐름:**
1. **영상 업로드**: 사용자가 숏폼 영상(15~60초) 업로드
2. **프레임 추출**: 3fps로 프레임 샘플링
3. **얼굴 탐지**: MTCNN으로 각 프레임에서 얼굴 영역 추출
4. **CNN 분류**: Fine-tuned 모델로 각 프레임 REAL/FAKE 판별
5. **결과 종합**: 다수결 투표로 최종 영상 판정

> 💡 **이 워크샵에서는** CNN 분류기(4번)를 Fine-tuning합니다. 이것이 영상 탐지의 핵심 엔진입니다.

## 핵심 목표

| 구분 | Before | After |
|------|--------|-------|
| 모델 | Pretrained | Fine-tuned |
| 학습 데이터 | 기존 데이터셋 | + 타겟 도메인 데이터 |
| 탐지 정확도 | ~70% | **~90%+** |

## Fine-tuning 기법 비교 (핵심!)

이 Workshop에서는 **세 가지 Fine-tuning 기법**을 직접 비교합니다:

| 기법 | 학습 파라미터 | 장점 | 단점 | 적합한 경우 |
|------|--------------|------|------|------------|
| **Full Fine-tuning** | 전체 (~4M) | 최고 성능 | 과적합 위험, 느림 | 데이터 충분, 성능 중요 |
| **Layer Freezing** | Classifier만 (~2K) | 빠름, 안정적 | 성능 제한 | 데이터 적음, 빠른 실험 |
| **LoRA** | Adapter (~10K) | 효율적, 좋은 성능 | 구현 복잡 | 대규모 모델, LLM |

## 실습 흐름

```
Step 1: 데이터 준비
    ↓
Step 2: Before 평가 (Fine-tuning 전)
    ↓
Step 3: 3가지 Fine-tuning 기법 비교 ★
    ↓
Step 4: After 평가 (기법별 성능 비교)
    ↓
Step 5: Before vs After 종합 비교
    ↓
Step 6: 영상 데모 배포 ★
```

## 워크샵 아키텍처

<img src="images/01_overall_architecture.png" alt="Workshop Architecture" width="1000">

## 프로덕션 적용 가이드

실제 게임 플랫폼 등에 적용 시, **비동기 추론(Async Inference)** 아키텍처를 권장합니다:

<img src="images/06_production_architecture.png" alt="Production Architecture" width="1000">

**비동기 추론의 장점:**
- 영상 분석(수 초 소요)에 적합
- Scale to Zero: 트래픽 없을 때 비용 0원
- SNS 콜백으로 결과 전달

## 예상 소요 시간

| 단계 | 내용 | 시간 |
|------|------|------|
| 1 | 데이터 준비 | 15분 |
| 2 | Before 평가 | 10분 |
| 3 | Fine-tuning (3가지 기법) | 45분 |
| 4 | After 평가 | 15분 |
| 5 | 성능 비교 | 10분 |
| 6 | 영상 데모 배포 | 15분 |
| **총합** | | **~2시간** |

## 예상 비용

| 리소스 | 인스턴스 | Spot 사용 시 |
|--------|----------|-------------|
| Training (3가지 기법) | ml.g4dn.xlarge × 3 | ~$0.66 |
| Endpoint | ml.g4dn.xlarge | ~$0.74/hr |
| S3 | - | ~$0.02 |
| **총합** | | **~$2.5** |

## 시작하기

👉 [사전 준비](getting-started/prerequisites.md)를 확인하세요.
