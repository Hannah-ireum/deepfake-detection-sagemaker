"""Gradio 데모 UI"""
import gradio as gr
import boto3
import json
import os
from pathlib import Path

# config.json에서 endpoint 이름 로드
PROJECT_ROOT = Path(__file__).parent.parent
config_path = PROJECT_ROOT / 'config.json'

if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    ENDPOINT_NAME = config.get('endpoint_name', 'deepfake-detector')
else:
    print(f"⚠️ config.json not found at {config_path}")
    print("노트북에서 먼저 Endpoint를 배포해주세요.")
    ENDPOINT_NAME = None

runtime = boto3.client('sagemaker-runtime')

def predict_deepfake(image):
    """이미지 딥페이크 판별"""
    if ENDPOINT_NAME is None:
        return "❌ Endpoint가 설정되지 않았습니다. 노트북에서 먼저 배포해주세요."

    import io
    from PIL import Image

    # 이미지를 바이트로 변환
    img_byte_arr = io.BytesIO()
    Image.fromarray(image).save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    try:
        # SageMaker Endpoint 호출
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/x-image',
            Body=img_bytes
        )

        result = json.loads(response['Body'].read().decode())

        prediction = result['prediction']
        confidence = result['confidence'] * 100

        if prediction == 'FAKE':
            return f"🚨 FAKE (가짜) - 신뢰도: {confidence:.1f}%"
        else:
            return f"✅ REAL (진짜) - 신뢰도: {confidence:.1f}%"
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

# Gradio 인터페이스
demo = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Image(label="이미지 업로드"),
    outputs=gr.Textbox(label="판별 결과"),
    title="🎭 딥페이크 탐지 데모 (한국인 특화)",
    description="KoDF 데이터로 Fine-tuning된 모델입니다. 이미지를 업로드하면 딥페이크 여부를 판별합니다.",
    examples=[]
)

if __name__ == "__main__":
    print(f"Endpoint: {ENDPOINT_NAME}")
    demo.launch(share=True)
