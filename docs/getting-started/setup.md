# 환경 설정

## Workshop Studio 접속

> 실습용 임시 AWS 계정이 제공됩니다. 개인 AWS 계정은 필요 없습니다.

### Step 1: Workshop Studio 접속

1. 강사가 제공한 **Workshop URL** 접속
2. **이메일** 입력 후 참가
3. 약관 동의 후 **Join event** 클릭

### Step 2: AWS 콘솔 접속

1. 왼쪽 메뉴에서 **Open AWS Console** 클릭
2. 새 탭에서 AWS 콘솔이 열림
3. 리전이 **서울(ap-northeast-2)** 또는 지정된 리전인지 확인

```
⚠️ 주의: Workshop Studio 계정은 실습 종료 후 자동 삭제됩니다.
         중요한 결과물은 로컬에 다운로드하세요.
```

---

## SageMaker Studio 접속

### Step 1: SageMaker Studio 열기

1. AWS 콘솔 상단 검색창에 **SageMaker** 입력
2. **Amazon SageMaker** 클릭
3. 왼쪽 메뉴에서 **Studio** 클릭
4. **Open Studio** 클릭

### Step 2: JupyterLab 열기

1. Studio 홈에서 **JupyterLab** 클릭
2. Space 생성 화면이 나오면:
   - Name: `deepfake-workshop`
   - Instance: `ml.t3.medium` (기본값)
3. **Run Space** 클릭
4. JupyterLab이 열릴 때까지 대기

---

## 실습 코드 가져오기

### 방법 1: Git Clone (권장)

JupyterLab 터미널에서:

```bash
git clone https://github.com/Hannah-ireum/deepfake-detection-sagemaker.git
cd deepfake-detection-sagemaker
```

### 방법 2: 직접 업로드

1. GitHub에서 ZIP 다운로드
2. JupyterLab에 드래그 앤 드롭

---

## 설정 파일 생성

첫 번째 노트북 실행 전에 `config.json`을 생성합니다.

```python
import json
import sagemaker

session = sagemaker.Session()
config = {
    "bucket": session.default_bucket(),
    "prefix": "deepfake-detection",
    "role": sagemaker.get_execution_role(),
    "region": session.boto_region_name
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print("설정 완료!")
print(f"Bucket: {config['bucket']}")
print(f"Region: {config['region']}")
```

---

## 환경 확인 체크리스트

| 항목 | 확인 방법 |
|------|----------|
| Workshop 접속 | AWS 콘솔 열림 |
| SageMaker Studio | JupyterLab 열림 |
| Git Clone | `deepfake-detection-sagemaker` 폴더 존재 |
| config.json | 버킷, 역할 정보 출력됨 |

---

## 문제 해결

### Studio가 열리지 않을 때
- 브라우저 팝업 차단 해제
- 다른 브라우저로 시도 (Chrome 권장)

### 권한 오류 발생 시
- Workshop Studio 세션이 만료되었을 수 있음
- 이벤트 페이지로 돌아가서 다시 콘솔 열기

### Git Clone 오류 시
```bash
# HTTPS로 시도
git clone https://github.com/Hannah-ireum/deepfake-detection-sagemaker.git
```

---

## 다음 단계

환경 설정이 완료되면 [1. 데이터 준비](../labs/01-data-preparation.md)로 이동합니다.
