# 4. After 평가 (기법별 성능 비교)

> 노트북: `4_after_evaluation/evaluate_after.ipynb`

## 개요

Fine-tuning 후 **각 기법별 모델의 성능**을 평가하고 **Model Registry에 등록**합니다.
동일한 테스트 데이터로 3가지 기법을 비교합니다.

## 학습 내용

- **3가지 Fine-tuned 모델 로드** (Full, Freeze, LoRA)
- 동일 테스트셋으로 기법별 평가
- 성능 비교 분석
- **SageMaker Model Registry 등록** (최고 성능 모델)

## 평가 프로세스

**평가 단계:**
1. **모델 다운로드**: S3에서 full, freeze, lora 모델 다운로드
2. **동일 테스트셋 평가**: 공정한 비교를 위해 같은 데이터 사용
3. **기법별 결과 비교**: Full ~90% | Freeze ~82% | LoRA ~87%
4. **Model Registry 등록**: 정확도 85% 이상인 최고 성능 모델 등록

## 주요 코드

### 다중 모델 다운로드

```python
# 학습된 모델 정보 로드
training_results = config.get('training_results', {})

# 각 기법별 모델 다운로드
for method, info in training_results.items():
    model_data = info['model_data']
    local_tar = f'./models/{method}_model.tar.gz'
    local_dir = f'./models/{method}'

    print(f"{method.upper()} 모델 다운로드 중...")
    !aws s3 cp {model_data} {local_tar}
    !tar -xzf {local_tar} -C {local_dir}
```

### 모든 모델 로드

```python
# 각 기법별 모델 로드
models = {}

for method in training_results.keys():
    model_path = f'./models/{method}/best_model.pth'

    model = DeepfakeDetector(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    models[method] = model
    print(f"✅ {method.upper()} 모델 로드 완료")
```

### 기법별 평가 실행

```python
# 각 기법별 평가
all_results = {}

for method, model in models.items():
    print(f"🔹 {method.upper()} 모델 평가 중...")
    results = evaluate_model(model, test_loader, device)
    all_results[method] = results
    print(f"   Accuracy: {results['accuracy']*100:.1f}%")

# 최고 성능 기법 찾기
best_method = max(all_results.keys(), key=lambda m: all_results[m]['accuracy'])
print(f"\n🏆 최고 성능: {best_method.upper()}")
```

### 결과 비교 테이블

```python
print("=" * 70)
print("  📊 Fine-tuning 기법별 성능 비교")
print("=" * 70)
print(f"{'기법':<12} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}")
print("-" * 70)

for method, results in all_results.items():
    print(f"{method.upper():<12} {results['accuracy']*100:>11.1f}% ...")

print("=" * 70)
```

## Model Registry 등록

### Model Registry란?

모델의 버전을 관리하고, 승인 프로세스를 거쳐 배포하는 중앙 저장소입니다.

### 등록 조건

- 정확도 **85% 이상**인 모델만 등록
- **최고 성능 기법**의 모델을 등록

```python
MODEL_PACKAGE_GROUP = "deepfake-detection-kodf"
ACCURACY_THRESHOLD = 0.85

# 최고 성능 기법의 모델 경로
best_model_data = training_results[best_method]['model_data']
best_accuracy = all_results[best_method]['accuracy']

if best_accuracy >= ACCURACY_THRESHOLD:
    print(f"✅ 정확도 {best_accuracy*100:.1f}% >= 85% 기준 충족!")

    # Model Registry 등록
    model_package = pytorch_model.register(
        model_package_group_name=MODEL_PACKAGE_GROUP,
        inference_instances=['ml.g4dn.xlarge', 'ml.m5.large'],
        approval_status='PendingManualApproval',
        description=f"{best_method.upper()} Fine-tuned (Accuracy: {best_accuracy*100:.1f}%)"
    )

    print(f"✅ Model Package ARN: {model_package.model_package_arn}")
else:
    print(f"❌ 정확도 미달 - Registry 등록 건너뜀")
```

## 예상 결과

| 기법 | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| **Full** | ~90% | ~91% | ~89% | ~90% |
| **Freeze** | ~82% | ~83% | ~81% | ~82% |
| **LoRA** | ~87% | ~88% | ~86% | ~87% |

### 기법별 특징 분석

| 기법 | 성능 | 학습 파라미터 | 특징 |
|------|------|--------------|------|
| **Full** | 최고 | 4M (100%) | 과적합 위험 있음 |
| **Freeze** | 제한적 | 2K (0.05%) | 매우 빠름, 안정적 |
| **LoRA** | 좋음 | 10K (0.25%) | 효율적, LLM에 인기 |

## Model Registry 상태

| 상태 | 설명 |
|------|------|
| `PendingManualApproval` | 등록 직후, 승인 대기 |
| `Approved` | 승인 완료, 배포 가능 |
| `Rejected` | 거부됨 |

### 모델 승인하기

```python
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus='Approved'
)
```

또는 SageMaker Console에서 수동 승인 가능합니다.

## 체크포인트

- [ ] 모든 기법 모델 다운로드 완료
- [ ] 테스트 데이터 추론 완료
- [ ] 기법별 성능 비교 완료
- [ ] **최고 성능 모델 Model Registry 등록**
- [ ] config.json 업데이트 확인

## 다음 단계

After 평가가 완료되면 [5. 성능 비교](05-comparison.md)로 이동합니다.
Before 모델과 각 Fine-tuning 기법의 성능을 종합 비교합니다.
