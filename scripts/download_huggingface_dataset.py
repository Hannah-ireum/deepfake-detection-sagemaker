#!/usr/bin/env python3
"""
Hugging Face 딥페이크 데이터셋 다운로드 + KoDF 결합
- Hugging Face에서 190K 이미지 다운로드
- 기존 KoDF Face Crop 데이터와 결합
- Train/Val/Test 분리

사용법:
    pip install datasets pillow
    python download_huggingface_dataset.py
"""

import os
from pathlib import Path
from PIL import Image
import random
import shutil

# ============================================
# 설정
# ============================================
OUTPUT_DIR = Path("/tmp/combined_dataset")
KODF_DIR = Path("/tmp/dataset_face_crop")  # 기존 KoDF Face Crop 데이터 (있으면 사용)

# Hugging Face 데이터셋에서 가져올 이미지 수
HF_TRAIN_PER_CLASS = 3000  # Train: Real 3000 + Fake 3000
HF_VAL_PER_CLASS = 500     # Val: Real 500 + Fake 500
HF_TEST_PER_CLASS = 500    # Test: Real 500 + Fake 500

TARGET_SIZE = (224, 224)

# ============================================
# 메인 함수
# ============================================

def download_and_prepare():
    print("=" * 60)
    print("Hugging Face 딥페이크 데이터셋 다운로드")
    print("=" * 60)

    # 1. Hugging Face 데이터셋 로드
    print("\n[1/4] Hugging Face 데이터셋 로드 중...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets 라이브러리 설치 중...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    print("    Hemg/deepfake-and-real-images 다운로드 중... (1.8GB)")
    dataset = load_dataset("Hemg/deepfake-and-real-images")

    print(f"    ✅ 로드 완료: {len(dataset['train'])}장")

    # 2. Real/Fake 분리
    print("\n[2/4] Real/Fake 이미지 분리 중...")
    real_images = []
    fake_images = []

    for item in dataset['train']:
        if item['label'] == 1:  # Real
            real_images.append(item['image'])
        else:  # Fake
            fake_images.append(item['image'])

    print(f"    Real: {len(real_images)}장")
    print(f"    Fake: {len(fake_images)}장")

    # 셔플
    random.seed(42)
    random.shuffle(real_images)
    random.shuffle(fake_images)

    # 3. 출력 디렉토리 생성
    print("\n[3/4] 데이터셋 저장 중...")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    splits = {
        'train': (HF_TRAIN_PER_CLASS, HF_TRAIN_PER_CLASS),
        'val': (HF_VAL_PER_CLASS, HF_VAL_PER_CLASS),
        'test': (HF_TEST_PER_CLASS, HF_TEST_PER_CLASS),
    }

    real_idx = 0
    fake_idx = 0

    for split_name, (real_count, fake_count) in splits.items():
        # Real 저장
        real_dir = OUTPUT_DIR / split_name / "real"
        real_dir.mkdir(parents=True, exist_ok=True)

        for i in range(real_count):
            if real_idx >= len(real_images):
                break
            img = real_images[real_idx]
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            img.save(real_dir / f"{i:05d}.jpg", quality=95)
            real_idx += 1

        # Fake 저장
        fake_dir = OUTPUT_DIR / split_name / "fake"
        fake_dir.mkdir(parents=True, exist_ok=True)

        for i in range(fake_count):
            if fake_idx >= len(fake_images):
                break
            img = fake_images[fake_idx]
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            img.save(fake_dir / f"{i:05d}.jpg", quality=95)
            fake_idx += 1

        real_saved = len(list(real_dir.glob("*.jpg")))
        fake_saved = len(list(fake_dir.glob("*.jpg")))
        print(f"    {split_name}: Real={real_saved}, Fake={fake_saved}")

    # 4. KoDF 데이터 결합 (있으면)
    print("\n[4/4] KoDF 데이터 결합 확인...")

    if KODF_DIR.exists():
        print(f"    KoDF 데이터 발견: {KODF_DIR}")

        for split_name in ['train', 'val', 'test']:
            for label in ['real', 'fake']:
                kodf_label_dir = KODF_DIR / split_name / label
                output_label_dir = OUTPUT_DIR / split_name / label

                if kodf_label_dir.exists():
                    existing_count = len(list(output_label_dir.glob("*.jpg")))
                    kodf_images = list(kodf_label_dir.glob("*.jpg"))

                    for i, img_path in enumerate(kodf_images):
                        new_path = output_label_dir / f"kodf_{i:05d}.jpg"
                        shutil.copy(img_path, new_path)

                    print(f"    {split_name}/{label}: +{len(kodf_images)}장 (KoDF)")
    else:
        print(f"    KoDF 데이터 없음 (Hugging Face만 사용)")

    # 최종 결과
    print("\n" + "=" * 60)
    print("✅ 데이터셋 생성 완료!")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        split_dir = OUTPUT_DIR / split
        real_count = len(list((split_dir / 'real').glob('*.jpg')))
        fake_count = len(list((split_dir / 'fake').glob('*.jpg')))
        print(f"  {split}: Real={real_count}, Fake={fake_count}")

    print(f"\n📁 저장 위치: {OUTPUT_DIR}")
    print(f"\nS3 업로드 명령어:")
    print(f"  aws s3 sync {OUTPUT_DIR} s3://YOUR-BUCKET/sample-data/")

# ============================================
# 실행
# ============================================

if __name__ == "__main__":
    download_and_prepare()
