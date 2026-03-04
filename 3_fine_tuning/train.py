"""
SageMaker Training Script for Deepfake Detection
Supports: Full Fine-tuning, Layer Freezing, LoRA
"""
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--model-name', type=str, default='efficientnet_b0')

    # Fine-tuning 기법 선택
    parser.add_argument('--finetune-method', type=str, default='full',
                        choices=['full', 'freeze', 'lora'],
                        help='Fine-tuning method: full, freeze, or lora')

    # LoRA 하이퍼파라미터
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank (default: 8)')

    # SageMaker 환경변수
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/train'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL', './data/val'))

    return parser.parse_args()


class DeepfakeDetector(nn.Module):
    """EfficientNet 기반 딥페이크 탐지 모델"""

    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) Layer"""

    def __init__(self, original_layer, rank=8):
        super().__init__()
        self.original_layer = original_layer

        # 원본 레이어 동결
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA 행렬 추가 (A: down-projection, B: up-projection)
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # 초기화: A는 정규분포, B는 0으로
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = 1.0

    def forward(self, x):
        # 원본 출력 + LoRA 출력
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output


def apply_lora(model, rank=8):
    """모델의 Linear 레이어에 LoRA 적용 (개선된 버전)

    EfficientNet의 SE (Squeeze-and-Excitation) 모듈과
    Classifier에 LoRA를 적용하여 더 많은 학습 용량 확보
    """
    lora_applied_count = 0

    # 1. SE 모듈의 Linear 레이어에 LoRA 적용
    # EfficientNet의 SE 모듈은 fc1 (reduce), fc2 (expand) 구조
    for name, module in model.backbone.named_modules():
        # SE 모듈의 fc1, fc2 레이어 찾기
        if 'se' in name.lower() and isinstance(module, nn.Linear):
            # 부모 모듈 찾기
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.backbone
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)

            # LoRA 적용
            original_layer = getattr(parent, child_name)
            setattr(parent, child_name, LoRALayer(original_layer, rank=rank))
            lora_applied_count += 1

    # 2. Classifier에 LoRA 적용
    if hasattr(model.backbone, 'classifier'):
        original_classifier = model.backbone.classifier
        model.backbone.classifier = LoRALayer(original_classifier, rank=rank)
        lora_applied_count += 1

    print(f"  LoRA 적용된 레이어 수: {lora_applied_count}")
    return model


def merge_lora_weights(model):
    """LoRA 가중치를 원본 레이어에 병합하여 표준 모델로 변환 (개선된 버전)

    모델 전체의 모든 LoRA 레이어를 찾아서 병합
    """
    merged_count = 0

    # 모든 LoRA 레이어 찾아서 병합
    def merge_lora_layer(lora_layer):
        """단일 LoRA 레이어 병합"""
        # LoRA 가중치 병합: W' = W + B @ A
        merged_weight = lora_layer.original_layer.weight.data + \
                       lora_layer.scaling * (lora_layer.lora_B.weight.data @ lora_layer.lora_A.weight.data)

        # 새 Linear 레이어 생성
        in_features = lora_layer.original_layer.in_features
        out_features = lora_layer.original_layer.out_features
        new_layer = nn.Linear(in_features, out_features)

        # 병합된 가중치 복사
        new_layer.weight.data = merged_weight
        if lora_layer.original_layer.bias is not None:
            new_layer.bias.data = lora_layer.original_layer.bias.data

        return new_layer

    # 모든 모듈 순회하면서 LoRA 레이어 병합
    modules_to_replace = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, LoRALayer):
            modules_to_replace.append(name)

    for name in modules_to_replace:
        # 부모 모듈 찾기
        parts = name.split('.')
        parent = model.backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]
        lora_layer = getattr(parent, child_name)
        new_layer = merge_lora_layer(lora_layer)
        setattr(parent, child_name, new_layer)
        merged_count += 1

    print(f"✓ LoRA 가중치 병합 완료 ({merged_count}개 레이어)")
    return model


def apply_layer_freezing(model, unfreeze_last_blocks=3):
    """Backbone 부분 동결 (개선된 버전)

    EfficientNet의 마지막 N개 블록 + Classifier를 학습
    초기 층은 일반적 특징(엣지, 텍스처)을 추출하므로 동결하고,
    후반 층은 태스크 특화 특징을 학습하도록 해제

    Args:
        model: 모델
        unfreeze_last_blocks: 학습할 마지막 블록 수 (기본값: 3)
    """
    # 먼저 모든 파라미터 동결
    for param in model.parameters():
        param.requires_grad = False

    # EfficientNet 블록 구조 분석
    # timm의 EfficientNet: conv_stem -> bn1 -> blocks (7개 스테이지) -> conv_head -> bn2 -> classifier
    if hasattr(model.backbone, 'blocks'):
        total_blocks = len(model.backbone.blocks)
        unfreeze_from = max(0, total_blocks - unfreeze_last_blocks)

        print(f"  전체 블록: {total_blocks}개, 학습할 블록: {unfreeze_from}~{total_blocks-1}")

        # 마지막 N개 블록 학습 가능하게
        for i in range(unfreeze_from, total_blocks):
            for param in model.backbone.blocks[i].parameters():
                param.requires_grad = True

    # conv_head (마지막 Conv 레이어) 학습 가능하게
    if hasattr(model.backbone, 'conv_head'):
        for param in model.backbone.conv_head.parameters():
            param.requires_grad = True

    # bn2 (마지막 BatchNorm) 학습 가능하게
    if hasattr(model.backbone, 'bn2'):
        for param in model.backbone.bn2.parameters():
            param.requires_grad = True

    # Classifier 학습 가능하게
    if hasattr(model.backbone, 'classifier'):
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True

    return model


def count_parameters(model):
    """학습 가능한 파라미터 수 계산"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_data_loaders(train_dir, val_dir, batch_size):
    """데이터 로더 생성"""

    # 학습용 transform (augmentation 포함)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 검증용 transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), correct / total


def main():
    args = parse_args()

    print("=" * 60)
    print("  딥페이크 탐지 모델 Fine-tuning")
    print("=" * 60)
    print(f"  Method: {args.finetune_method.upper()}")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    if args.finetune_method == 'lora':
        print(f"  LoRA Rank: {args.lora_rank}")
    print("=" * 60)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 모델 저장 디렉토리 생성
    os.makedirs(args.model_dir, exist_ok=True)

    # 데이터 로더
    train_loader, val_loader = get_data_loaders(args.train, args.val, args.batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # 모델 생성
    model = DeepfakeDetector(model_name=args.model_name, pretrained=True)

    # Fine-tuning 기법 적용
    if args.finetune_method == 'freeze':
        print("\n🔒 Layer Freezing 적용: 마지막 3개 블록 + Classifier 학습")
        model = apply_layer_freezing(model, unfreeze_last_blocks=3)
    elif args.finetune_method == 'lora':
        print(f"\n🔧 LoRA 적용: rank={args.lora_rank}")
        # LoRA는 전체 동결 후 LoRA 어댑터만 학습
        model = apply_layer_freezing(model, unfreeze_last_blocks=0)  # 전체 동결
        model = apply_lora(model, rank=args.lora_rank)
    else:
        print("\n🔓 Full Fine-tuning: 전체 파라미터 학습")

    # 파라미터 수 출력
    trainable, total = count_parameters(model)
    print(f"학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model = model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()

    # 학습 가능한 파라미터만 옵티마이저에 전달
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 학습 로그
    history = {
        'method': args.finetune_method,
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'trainable_params': trainable,
        'total_params': total
    }
    best_val_acc = 0.0

    # 학습 루프
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 베스트 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f"✓ Best model saved! (Val Acc: {val_acc*100:.2f}%)")

    # LoRA인 경우 가중치 병합 (표준 모델로 변환)
    if args.finetune_method == 'lora':
        # 최종 모델 병합 및 저장
        model = merge_lora_weights(model)
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

        # best_model도 병합하여 다시 저장
        best_model_path = os.path.join(args.model_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            # LoRA 구조로 저장된 best_model 로드
            best_model = DeepfakeDetector(model_name=args.model_name, pretrained=False)
            best_model = apply_layer_freezing(best_model, unfreeze_last_blocks=0)
            best_model = apply_lora(best_model, rank=args.lora_rank)
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            # 병합 후 저장
            best_model = merge_lora_weights(best_model)
            torch.save(best_model.state_dict(), best_model_path)
            print("✓ Best model의 LoRA 가중치도 병합되었습니다.")
    else:
        # Full/Freeze는 그대로 저장
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

    # 학습 기록 저장
    history['best_val_acc'] = best_val_acc
    with open(os.path.join(args.model_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Fine-tuning 완료! ({args.finetune_method.upper()})")
    print(f"  Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  학습 파라미터: {trainable:,} ({100*trainable/total:.2f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
