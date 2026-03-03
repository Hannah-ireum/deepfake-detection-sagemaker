# 환경 설정

## Workshop Studio 접속

> 실습용 임시 AWS 계정이 제공됩니다. 개인 AWS 계정은 필요 없습니다.

### Step 1: Workshop Studio 접속

1. 강사가 제공한 **Workshop URL** 접속
   - 예: `https://catalog.workshops.aws/join?access-code=xxxx-xxxx-xx`
2. **이메일** 입력
3. **I agree to the Terms and Conditions** 체크
4. **Join event** 클릭

### Step 2: AWS 콘솔 접속

1. 왼쪽 메뉴에서 **Open AWS Console** 클릭 (주황색 버튼)
2. 새 탭에서 AWS 콘솔이 열림
3. 우측 상단에서 리전 확인
   - **서울(ap-northeast-2)** 또는 강사가 안내한 리전

```
⚠️ 주의: Workshop Studio 계정은 실습 종료 후 자동 삭제됩니다.
         중요한 결과물은 로컬에 다운로드하세요.
```

---

## SageMaker Studio 접속

### Step 1: SageMaker 서비스 찾기

1. AWS 콘솔 상단 **검색창** 클릭
2. `SageMaker` 입력
3. **Amazon SageMaker** 클릭

### Step 2: Studio 열기

1. 왼쪽 메뉴에서 **Studio** 클릭
2. **Open Studio** 버튼 클릭
3. 새 탭에서 SageMaker Studio가 열림

### Step 3: JupyterLab 실행

1. Studio 홈 화면에서 **JupyterLab** 카드 찾기
2. **JupyterLab** 클릭
3. Space 생성 화면이 나오면:
   - **Name**: `deepfake-workshop`
   - **Instance**: `ml.t3.medium` (기본값 유지)
4. **Run Space** 클릭
5. 상태가 `Running`이 될 때까지 대기 (1-2분)
6. **Open JupyterLab** 클릭

---

## 실습 코드 가져오기

### Step 1: 터미널 열기

1. JupyterLab 상단 메뉴에서 **File** 클릭
2. **New** → **Terminal** 클릭
3. 검은색 터미널 창이 열림

### Step 2: Git Clone 실행

터미널에 아래 명령어를 **한 줄씩** 입력하고 Enter:

```bash
git clone https://github.com/Hannah-ireum/deepfake-detection-sagemaker.git
```

성공하면 아래와 같이 출력됩니다:
```
Cloning into 'deepfake-detection-sagemaker'...
remote: Enumerating objects: 50, done.
remote: Counting objects: 100% (50/50), done.
...
```

### Step 3: 폴더 확인

왼쪽 파일 브라우저에서 `deepfake-detection-sagemaker` 폴더가 보이는지 확인합니다.

```
📁 deepfake-detection-sagemaker/
├── 📁 1_data_preparation/
├── 📁 2_before_evaluation/
├── 📁 3_fine_tuning/
├── 📁 4_after_evaluation/
├── 📁 5_comparison/
├── 📁 6_demo/
├── 📁 docs/
├── 📄 README.md
└── 📄 requirements.txt
```

---

## 첫 번째 노트북 실행

### Step 1: 노트북 열기

1. 왼쪽 파일 브라우저에서 `deepfake-detection-sagemaker` 폴더 더블클릭
2. `1_data_preparation` 폴더 더블클릭
3. `prepare_data.ipynb` 더블클릭

### Step 2: 커널 선택

노트북이 열리면 우측 상단에 커널 선택 창이 나올 수 있습니다:
- **Python 3 (ipykernel)** 또는 **Python 3** 선택
- **Select** 클릭

### Step 3: 셀 실행

1. 첫 번째 코드 셀 클릭
2. **Shift + Enter** 또는 상단 **▶ Run** 버튼 클릭
3. 셀 왼쪽에 `[1]` 숫자가 나오면 실행 완료
4. 다음 셀도 순서대로 **Shift + Enter**로 실행

> `config.json`은 첫 번째 노트북의 마지막 셀에서 **자동 생성**됩니다.

---

## 환경 확인 체크리스트

| 단계 | 확인 사항 | 완료 |
|------|----------|------|
| 1 | Workshop Studio 접속됨 | ☐ |
| 2 | AWS 콘솔 열림 | ☐ |
| 3 | SageMaker Studio 열림 | ☐ |
| 4 | JupyterLab 실행됨 | ☐ |
| 5 | 터미널에서 git clone 성공 | ☐ |
| 6 | 폴더 구조 확인됨 | ☐ |
| 7 | 첫 번째 노트북 열림 | ☐ |

---

## 문제 해결

### Studio가 열리지 않을 때
- 브라우저 **팝업 차단 해제**
- 다른 브라우저로 시도 (**Chrome 권장**)
- 페이지 새로고침 (F5)

### JupyterLab이 시작되지 않을 때
- Space 상태가 `Starting`이면 1-2분 대기
- `Failed` 상태면 **Stop** 후 다시 **Run**

### Git Clone 오류 시
```bash
# 이미 폴더가 있으면 삭제 후 재시도
rm -rf deepfake-detection-sagemaker
git clone https://github.com/Hannah-ireum/deepfake-detection-sagemaker.git
```

### 권한 오류 발생 시
- Workshop Studio 세션이 만료되었을 수 있음
- 이벤트 페이지로 돌아가서 다시 **Open AWS Console** 클릭

---

## 다음 단계

환경 설정이 완료되면 [1. 데이터 준비](../labs/01-data-preparation.md)로 이동합니다.

또는 노트북에서 바로 시작:
```
1_data_preparation/prepare_data.ipynb
```
