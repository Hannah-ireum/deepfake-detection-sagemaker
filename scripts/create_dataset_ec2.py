#!/usr/bin/env python3
"""
KoDF 데이터셋 생성 스크립트 (EC2용)
- 여러 zip 파일에서 공통 피험자 찾기
- 영상에서 프레임 추출
- Face Crop + Augmentation 적용
- 피험자 단위 Train/Val/Test 분리

사용법:
    pip install facenet-pytorch opencv-python pillow
    python create_dataset_ec2.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random
import shutil
from collections import defaultdict

# ============================================
# 설정
# ============================================
BASE_DIR = Path("/tmp/kodf_data")
OUTPUT_DIR = Path("/tmp/dataset_face_crop")

# Real/Fake 폴더들
REAL_DIRS = [
    BASE_DIR / "원본1",
    BASE_DIR / "원본2",
    BASE_DIR / "원본3",
]

FAKE_DIRS = [
    BASE_DIR / "dffs1",
    BASE_DIR / "dffs2",
    BASE_DIR / "dffs3",
]

# 데이터셋 크기 설정
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FRAMES_PER_VIDEO = 5  # 영상당 추출할 프레임 수
TARGET_SIZE = (224, 224)
AUGMENT_FACTOR = 6  # Train 데이터 augmentation 배수

# ============================================
# MTCNN 로드
# ============================================
print("=" * 60)
print("KoDF 데이터셋 생성 스크립트 (EC2용)")
print("=" * 60)

try:
    from facenet_pytorch import MTCNN
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=50)
    print(f"✅ MTCNN 로드 완료 (device: {device})")
except ImportError:
    print("facenet-pytorch 설치 중...")
    os.system("pip install facenet-pytorch -q")
    from facenet_pytorch import MTCNN
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=50)
    print(f"✅ MTCNN 로드 완료 (device: {device})")

# ============================================
# 유틸리티 함수들
# ============================================

def find_all_subjects(dirs):
    """여러 디렉토리에서 모든 피험자 ID 찾기"""
    subjects = set()
    for d in dirs:
        if d.exists():
            for item in d.iterdir():
                if item.is_dir():
                    subjects.add(item.name)
    return subjects

def find_common_subjects(real_dirs, fake_dirs):
    """Real과 Fake 모두에 있는 피험자 찾기"""
    real_subjects = find_all_subjects(real_dirs)
    fake_subjects = find_all_subjects(fake_dirs)
    common = real_subjects & fake_subjects
    return sorted(list(common))

def get_videos_for_subject(subject, dirs):
    """특정 피험자의 모든 영상 파일 반환"""
    videos = []
    for d in dirs:
        subject_dir = d / subject
        if subject_dir.exists():
            videos.extend(list(subject_dir.glob("*.mp4")))
    return videos

def extract_frames_from_video(video_path, num_frames=5):
    """영상에서 균등 간격으로 프레임 추출"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = max(1, total_frames)

    if total_frames == 0:
        cap.release()
        return frames

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

def detect_and_crop_face(frame, target_size=(224, 224), margin=40):
    """프레임에서 얼굴 검출 및 크롭"""
    try:
        img = Image.fromarray(frame)

        # MTCNN으로 얼굴 검출
        boxes, probs = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            # 얼굴 검출 실패 시 중앙 크롭
            w, h = img.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            img = img.crop((left, top, left + size, top + size))
            return img.resize(target_size, Image.BILINEAR)

        # 가장 큰 얼굴 선택
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]

        # 마진 추가
        w, h = img.size
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        # 정사각형으로 만들기
        face_w = x2 - x1
        face_h = y2 - y1
        if face_w > face_h:
            diff = face_w - face_h
            y1 = max(0, y1 - diff // 2)
            y2 = min(h, y2 + diff // 2)
        else:
            diff = face_h - face_w
            x1 = max(0, x1 - diff // 2)
            x2 = min(w, x2 + diff // 2)

        face = img.crop((x1, y1, x2, y2))
        return face.resize(target_size, Image.BILINEAR)

    except Exception as e:
        return None

# ============================================
# Augmentation 함수들
# ============================================

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def adjust_brightness(img, factor):
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_contrast(img, factor):
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def adjust_color(img, factor):
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def add_noise(img, intensity=10):
    arr = np.array(img)
    noise = np.random.randint(-intensity, intensity, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_augmentation(img, num_augments=6):
    """여러 augmentation 적용"""
    augmented = [img]  # 원본 포함

    # 좌우 반전
    augmented.append(horizontal_flip(img))

    # 밝기 변화
    augmented.append(adjust_brightness(img, random.uniform(0.7, 1.3)))

    # 대비 변화
    augmented.append(adjust_contrast(img, random.uniform(0.8, 1.2)))

    # 색상 변화
    augmented.append(adjust_color(img, random.uniform(0.8, 1.2)))

    # 노이즈 추가
    augmented.append(add_noise(img, 15))

    return augmented[:num_augments]

# ============================================
# 메인 데이터셋 생성
# ============================================

def create_dataset():
    """데이터셋 생성 메인 함수"""

    # 1. 공통 피험자 찾기
    print("\n[1/5] 공통 피험자 찾기...")
    common_subjects = find_common_subjects(REAL_DIRS, FAKE_DIRS)
    print(f"    Real/Fake 공통 피험자: {len(common_subjects)}명")

    if len(common_subjects) < 10:
        print("⚠️ 피험자가 너무 적습니다. 데이터를 확인하세요.")
        return

    # 2. Train/Val/Test 피험자 분리
    print("\n[2/5] 피험자 분리 (Train/Val/Test)...")
    random.seed(42)
    random.shuffle(common_subjects)

    n_train = int(len(common_subjects) * TRAIN_RATIO)
    n_val = int(len(common_subjects) * VAL_RATIO)

    train_subjects = common_subjects[:n_train]
    val_subjects = common_subjects[n_train:n_train + n_val]
    test_subjects = common_subjects[n_train + n_val:]

    print(f"    Train: {len(train_subjects)}명")
    print(f"    Val: {len(val_subjects)}명")
    print(f"    Test: {len(test_subjects)}명")

    # 3. 출력 디렉토리 생성
    print("\n[3/5] 출력 디렉토리 생성...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    # 4. 데이터셋 생성
    splits = {
        'train': (train_subjects, True),   # (피험자들, augmentation 여부)
        'val': (val_subjects, False),
        'test': (test_subjects, False),
    }

    for split_name, (subjects, do_augment) in splits.items():
        print(f"\n[4/5] {split_name.upper()} 데이터 생성 중...")

        for label, dirs in [('real', REAL_DIRS), ('fake', FAKE_DIRS)]:
            output_label_dir = OUTPUT_DIR / split_name / label
            count = 0

            for subject in subjects:
                videos = get_videos_for_subject(subject, dirs)

                for video in videos:
                    frames = extract_frames_from_video(video, FRAMES_PER_VIDEO)

                    for frame in frames:
                        face = detect_and_crop_face(frame, TARGET_SIZE)
                        if face is None:
                            continue

                        if do_augment:
                            augmented = apply_augmentation(face, AUGMENT_FACTOR)
                        else:
                            augmented = [face]

                        for aug_img in augmented:
                            output_path = output_label_dir / f"{count:05d}.jpg"
                            aug_img.save(output_path, quality=95)
                            count += 1

            print(f"    {label}: {count}장 저장")

    # 5. 결과 출력
    print("\n" + "=" * 60)
    print("✅ 데이터셋 생성 완료!")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        split_dir = OUTPUT_DIR / split
        real_count = len(list((split_dir / 'real').glob('*.jpg')))
        fake_count = len(list((split_dir / 'fake').glob('*.jpg')))
        print(f"  {split}: Real={real_count}, Fake={fake_count}")

    print(f"\n📁 저장 위치: {OUTPUT_DIR}")
    print(f"\n다음 명령어로 S3에 업로드하세요:")
    print(f"  aws s3 sync {OUTPUT_DIR} s3://YOUR-BUCKET/sample-data/")

# ============================================
# 실행
# ============================================

if __name__ == "__main__":
    create_dataset()
