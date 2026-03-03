# 강사용 가이드

## 개요

이 Workshop은 참가자가 직접 Kaggle에서 데이터를 다운로드하는 방식으로 진행됩니다.
강사가 별도로 데이터를 준비할 필요가 없습니다.

## Workshop 흐름

```
1. 데이터 준비 (Kaggle 다운로드)
   ↓
2. Before 평가 (사전 학습된 모델)
   ↓
3. Fine-tuning (SageMaker Training Job)
   ↓
4. After 평가 (Fine-tuned 모델)
   ↓
5. 성능 비교 (Before vs After)
   ↓
6. 데모 배포 (Gradio)
```

## 참가자 사전 준비 안내

Workshop 시작 전 참가자에게 안내할 사항:

### 1. Kaggle 계정

- [Kaggle](https://www.kaggle.com) 가입
- **Account Settings** → **API** → **Create New Token**
- `kaggle.json` 파일 다운로드

### 2. Workshop Studio 접속 정보

- Workshop URL 공유
- 리전 안내 (예: ap-northeast-2)

## 데이터셋 정보

**140k Real and Fake Faces**
- URL: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- 크기: 약 4GB
- 구성: Real 70,000장 + Fake 70,000장 (StyleGAN 생성)

### Workshop용 샘플 크기

전체 140k를 사용하면 시간이 오래 걸리므로, 노트북에서 자동으로 샘플링합니다:

| 구분 | 클래스당 | 총 개수 |
|------|----------|---------|
| Train | 1,000장 | 2,000장 |
| Validation | 200장 | 400장 |
| Test | 200장 | 400장 |

## 예상 소요 시간

| 단계 | 예상 시간 | 비고 |
|------|-----------|------|
| 1. 데이터 준비 | 15-20분 | Kaggle 다운로드 포함 |
| 2. Before 평가 | 10-15분 | 추론만 수행 |
| 3. Fine-tuning | 20-30분 | Spot Instance 사용 |
| 4. After 평가 | 10-15분 | 추론만 수행 |
| 5. 성능 비교 | 5분 | 결과 시각화 |
| 6. 데모 배포 | 10분 | Gradio 실행 |

**총 예상 시간: 약 1.5-2시간**

## 비용 안내

Workshop Studio 환경에서는 비용이 청구되지 않습니다.

개인 계정 사용 시 예상 비용:
- SageMaker Training (ml.g4dn.xlarge, Spot): ~$0.50-1.00
- SageMaker Endpoint (ml.g4dn.xlarge): ~$0.70/hour
- S3 저장소: ~$0.01

## 트러블슈팅

### Kaggle 다운로드 실패

```bash
# 권한 확인
ls -la ~/.kaggle/kaggle.json

# 권한 수정
chmod 600 ~/.kaggle/kaggle.json
```

### 데이터 다운로드가 느린 경우

Workshop Studio 네트워크에 따라 다운로드 속도가 달라질 수 있습니다.
5-10분 정도 소요될 수 있음을 참가자에게 안내하세요.

### SageMaker 용량 부족

Spot Instance 사용 시 간헐적으로 용량 부족이 발생할 수 있습니다:

```python
# Spot 비활성화
USE_SPOT = False
```

## 체크리스트

Workshop 시작 전:
- [ ] Workshop URL 준비
- [ ] 참가자에게 Kaggle 계정 생성 안내
- [ ] SageMaker Studio 환경 정상 동작 확인
- [ ] 네트워크 속도 확인 (Kaggle 다운로드 테스트)

Workshop 진행 중:
- [ ] 각 노트북 실행 결과 확인
- [ ] 참가자 진행 상황 모니터링
- [ ] Q&A 대응
