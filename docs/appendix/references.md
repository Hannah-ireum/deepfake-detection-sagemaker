# 참고 자료

## 데이터셋

### KoDF (Korean DeepFake)
- **설명**: 한국인 얼굴 기반 딥페이크 데이터셋
- **출처**: AI Hub
- **링크**: https://www.aihub.or.kr/

### FaceForensics++
- **설명**: 딥페이크 탐지 벤치마크 데이터셋
- **논문**: "FaceForensics++: Learning to Detect Manipulated Facial Images"
- **GitHub**: https://github.com/ondyari/FaceForensics

### Celeb-DF
- **설명**: Celebrity 기반 딥페이크 데이터셋
- **논문**: "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics"
- **링크**: https://github.com/yuezunli/celeb-deepfakeforensics

## AWS 문서

### Amazon SageMaker
- **공식 문서**: https://docs.aws.amazon.com/sagemaker/
- **Python SDK**: https://sagemaker.readthedocs.io/
- **예제 노트북**: https://github.com/aws/amazon-sagemaker-examples

### 주요 가이드
- [Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
- [PyTorch Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/index.html)
- [Endpoint 배포](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [비용 최적화](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-cost-optimization.html)

## 딥러닝 라이브러리

### PyTorch
- **공식 문서**: https://pytorch.org/docs/
- **튜토리얼**: https://pytorch.org/tutorials/

### timm (PyTorch Image Models)
- **설명**: 사전 학습 이미지 모델 컬렉션
- **GitHub**: https://github.com/huggingface/pytorch-image-models
- **문서**: https://huggingface.co/docs/timm/

### Gradio
- **공식 문서**: https://gradio.app/docs/
- **튜토리얼**: https://gradio.app/guides/

## 논문

### 딥페이크 탐지
1. **FaceForensics++** (2019)
   - "FaceForensics++: Learning to Detect Manipulated Facial Images"
   - https://arxiv.org/abs/1901.08971

2. **EfficientNet** (2019)
   - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
   - https://arxiv.org/abs/1905.11946

3. **Face X-ray** (2020)
   - "Face X-ray for More General Face Forgery Detection"
   - https://arxiv.org/abs/1912.13458

### 전이 학습
1. **Transfer Learning Survey** (2020)
   - "A Survey on Transfer Learning"
   - https://arxiv.org/abs/1808.01974

## 블로그 및 튜토리얼

### AWS 블로그
- [SageMaker로 딥러닝 모델 학습하기](https://aws.amazon.com/blogs/machine-learning/)
- [PyTorch on SageMaker](https://aws.amazon.com/blogs/machine-learning/tag/pytorch/)

### 기술 블로그
- [Towards Data Science - Deepfake Detection](https://towardsdatascience.com/tagged/deepfake)
- [Medium - SageMaker Tutorials](https://medium.com/tag/amazon-sagemaker)

## 유용한 도구

### 얼굴 검출
- **MTCNN**: https://github.com/ipazc/mtcnn
- **RetinaFace**: https://github.com/serengil/retinaface
- **dlib**: http://dlib.net/

### 시각화
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/
- **Plotly**: https://plotly.com/python/

## 커뮤니티

### 포럼
- **AWS re:Post**: https://repost.aws/
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/amazon-sagemaker

### GitHub
- **SageMaker Examples**: https://github.com/aws/amazon-sagemaker-examples
- **Deepfake Detection Challenge**: https://github.com/selimsef/dfdc_deepfake_challenge
