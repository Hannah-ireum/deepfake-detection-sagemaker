# 딥페이크 탐지 모델 Fine-tuning 실습

## 개요

글로벌 딥페이크 탐지 모델을 **한국인 얼굴에 특화**되도록 Fine-tuning하는 실습입니다.

Amazon SageMaker를 활용하여 FaceForensics++ 기반의 사전 학습 모델을 KoDF(Korean DeepFake) 데이터셋으로 파인튜닝하고, 성능 향상을 직접 확인합니다.

## 학습 목표

이 실습을 통해 다음을 배웁니다:

- SageMaker Training Job을 활용한 딥러닝 모델 파인튜닝
- 사전 학습 모델의 전이 학습(Transfer Learning) 기법
- 모델 성능 평가 및 비교 분석
- SageMaker Endpoint를 통한 모델 배포
- Gradio를 활용한 데모 UI 구축

## Before vs After

| 구분 | Before | After |
|------|--------|-------|
| 모델 | FaceForensics++ Pretrained | KoDF Fine-tuned |
| 학습 데이터 | 서양인 얼굴 위주 | 한국인 얼굴 추가 |
| 한국인 영상 정확도 | ~70% | ~90%+ |

## 대상

- AWS 서비스에 관심 있는 ML/AI 개발자
- 딥페이크 탐지 기술을 배우고 싶은 분
- SageMaker 활용법을 실습으로 익히고 싶은 분

## 예상 소요 시간

전체 실습: 약 2시간 30분

## 예상 비용

약 $2.5 (온디맨드 기준)
