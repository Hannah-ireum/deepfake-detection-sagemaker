"""
샘플 테스트 영상 생성 스크립트
- test 이미지들을 사용해서 REAL/FAKE 샘플 영상 생성
- 데모 테스트용
"""
import cv2
import os
from pathlib import Path
import random

def create_video_from_images(image_dir, output_path, fps=30, duration=5):
    """이미지들을 사용해서 영상 생성"""
    images = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return False

    # 랜덤하게 섞기
    random.shuffle(images)

    # 첫 번째 이미지로 크기 확인
    first_img = cv2.imread(str(images[0]))
    height, width = first_img.shape[:2]

    # 영상 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # duration초 동안의 프레임 수
    total_frames = fps * duration

    for i in range(total_frames):
        # 이미지 순환
        img_idx = i % len(images)
        img = cv2.imread(str(images[img_idx]))

        if img is not None:
            # 크기 맞추기
            img = cv2.resize(img, (width, height))
            out.write(img)

    out.release()
    return True

def main():
    # 프로젝트 루트
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "test"
    output_dir = Path(__file__).parent / "sample_videos"

    # 출력 디렉토리 생성
    output_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("샘플 테스트 영상 생성")
    print("=" * 50)

    # REAL 영상 생성
    real_dir = data_dir / "real"
    if real_dir.exists():
        output_path = str(output_dir / "sample_real.mp4")
        print(f"\n📹 REAL 샘플 영상 생성 중...")
        if create_video_from_images(real_dir, output_path, fps=30, duration=5):
            print(f"   ✅ 저장: {output_path}")
    else:
        print(f"❌ {real_dir} 디렉토리가 없습니다.")

    # FAKE 영상 생성
    fake_dir = data_dir / "fake"
    if fake_dir.exists():
        output_path = str(output_dir / "sample_fake.mp4")
        print(f"\n📹 FAKE 샘플 영상 생성 중...")
        if create_video_from_images(fake_dir, output_path, fps=30, duration=5):
            print(f"   ✅ 저장: {output_path}")
    else:
        print(f"❌ {fake_dir} 디렉토리가 없습니다.")

    print("\n" + "=" * 50)
    print("🎉 샘플 영상 생성 완료!")
    print(f"   위치: {output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
