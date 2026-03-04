"""SageMaker Inference Script"""
import os
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import io
import json
import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_fn(model_dir):
    """모델 로드"""
    logger.info(f"Loading model from {model_dir}")
    logger.info(f"Directory contents: {os.listdir(model_dir)}")

    class DeepfakeDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        def forward(self, x):
            return self.backbone(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # model.pth 또는 best_model.pth 찾기
    model_path = None
    for filename in ['best_model.pth', 'model.pth']:
        candidate = os.path.join(model_dir, filename)
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(f"No model file found in {model_dir}. Contents: {os.listdir(model_dir)}")

    logger.info(f"Loading model from: {model_path}")

    model = DeepfakeDetector()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    logger.info("Model loaded successfully!")
    return model

def input_fn(request_body, request_content_type):
    """입력 처리"""
    logger.info(f"Processing input with content type: {request_content_type}")

    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """예측"""
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(input_data.to(device))
        probs = torch.softmax(output, dim=1)
    return probs.cpu().numpy()[0]

def output_fn(prediction, accept):
    """출력 포맷"""
    return json.dumps({
        'fake_probability': float(prediction[0]),
        'real_probability': float(prediction[1]),
        'prediction': 'FAKE' if prediction[0] > prediction[1] else 'REAL',
        'confidence': float(max(prediction))
    })
