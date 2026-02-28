# 🎭 딥페이크 탐지 모델 Fine-tuning 실습

## 개요

**글로벌 딥페이크 탐지 모델을 한국인 얼굴에 특화되도록 Fine-tuning하는 실습입니다.**

| 구분 | Before | After |
|------|--------|-------|
| 모델 | FaceForensics++ Pretrained | KoDF Fine-tuned |
| 학습 데이터 | 서양인 얼굴 위주 | 한국인 얼굴 추가 |
| 한국인 영상 정확도 | ~70% | ~90%+ |

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    실습 전체 흐름                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [KoDF 샘플 데이터]                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  Amazon S3  │  ← Train/Val/Test 데이터 저장               │
│  └──────┬──────┘                                            │
│         │                                                   │
│    ┌────┴────┐                                              │
│    ▼         ▼                                              │
│ [Before]  [SageMaker Training]                              │
│ FF++ 모델      │                                            │
│    │          ▼                                             │
│    │     [After]                                            │
│    │     Fine-tuned 모델                                    │
│    │          │                                             │
│    └────┬─────┘                                             │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │  성능 비교 평가   │  ← Before vs After                    │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  SageMaker      │                                        │
│  │  Endpoint 배포   │                                        │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  Gradio 데모 UI  │  ← 실시간 영상 판별 체험               │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 실습 단계

| 단계 | 폴더 | 내용 | 예상 시간 |
|------|------|------|----------|
| 1 | `1_data_preparation/` | KoDF 샘플 데이터 준비 & S3 업로드 | 30분 |
| 2 | `2_before_evaluation/` | FF++ Pretrained 모델 성능 평가 | 20분 |
| 3 | `3_fine_tuning/` | SageMaker Fine-tuning 실행 | 40분 |
| 4 | `4_after_evaluation/` | Fine-tuned 모델 성능 평가 | 20분 |
| 5 | `5_comparison/` | Before vs After 비교 시각화 | 15분 |
| 6 | `6_demo/` | Endpoint 배포 & Gradio 데모 | 30분 |

## 사전 준비

### 필수 요구사항
- AWS 계정 (SageMaker 접근 권한)
- Python 3.8+
- AWS CLI 설정 완료

### AWS 리소스
- S3 버킷
- SageMaker 노트북 인스턴스 또는 Studio
- IAM Role (SageMaker 실행 권한)

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 노트북 순서대로 실행
# 1_data_preparation/prepare_data.ipynb
# 2_before_evaluation/evaluate_before.ipynb
# 3_fine_tuning/run_finetuning.ipynb
# 4_after_evaluation/evaluate_after.ipynb
# 5_comparison/compare_results.ipynb
# 6_demo/deploy_and_demo.ipynb
```

## 예상 비용

| 리소스 | 인스턴스 | 사용량 | 예상 비용 |
|--------|----------|--------|----------|
| SageMaker Training | ml.g4dn.xlarge | 1시간 | ~$0.7 |
| SageMaker Endpoint | ml.g4dn.xlarge | 2시간 | ~$1.4 |
| S3 Storage | - | 1GB | ~$0.02 |
| **총합** | | | **~$2.5** |

## 참고 자료

- [KoDF 데이터셋](https://www.aihub.or.kr/)
- [FaceForensics++ 벤치마크](https://github.com/ondyari/FaceForensics)
- [Amazon SageMaker 문서](https://docs.aws.amazon.com/sagemaker/)

## 라이선스

이 실습 자료는 교육 목적으로 제공됩니다.
