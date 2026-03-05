"""
All diagrams for Deepfake Detection Workshop - Larger fonts
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUTPUT_DIR = "./images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Larger font settings
plt.rcParams['font.size'] = 14
plt.rcParams['figure.facecolor'] = 'white'


def create_overfitting_graph():
    """과적합 그래프"""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(1, 11)
    train_loss = 1.0 * np.exp(-0.3 * epochs) + 0.1
    val_loss = np.concatenate([
        1.0 * np.exp(-0.3 * epochs[:5]) + 0.15,
        [0.35, 0.40, 0.50, 0.65, 0.85]
    ])

    ax.plot(epochs, train_loss, 'b-', linewidth=3, label='Train Loss')
    ax.plot(epochs, val_loss, 'r-', linewidth=3, label='Val Loss')
    ax.axvline(x=5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Early Stopping')

    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.set_title('Overfitting Detection', fontsize=18, fontweight='bold')
    ax.legend(fontsize=14)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)

    ax.annotate('Overfitting!', xy=(7, 0.55), fontsize=14, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/overfitting_graph.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ overfitting_graph.png")


def create_lora_structure():
    """LoRA 구조도"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Boxes with larger text
    boxes = [
        (3.5, 8.5, 3, 1, 'Input (x)', 'lightgray'),
        (0.5, 5.5, 3, 1, 'W (frozen)', 'lightcoral'),
        (6.5, 5.5, 3, 1, 'A (train)', 'lightgreen'),
        (6.5, 3.5, 3, 1, 'B (train)', 'lightgreen'),
        (3.5, 1, 3, 1, 'Output (y)', 'lightyellow'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=14, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2, 6.5), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(8, 6.5), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(8, 4.5), xytext=(8, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5, 2), xytext=(2, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5, 2), xytext=(8, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Labels
    ax.text(2.5, 3.8, 'W×x', fontsize=13, fontweight='bold')
    ax.text(6.5, 2.8, 'B×A×x', fontsize=13, fontweight='bold')
    ax.text(5, 0.3, 'W×x + B×A×x', fontsize=14, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lora_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ lora_structure.png")


def create_layer_comparison():
    """Full vs Freeze 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, (ax, title, trainable) in enumerate(zip(
        axes,
        ['Full Fine-tuning', 'Layer Freezing'],
        [[True, True, True, True], [False, False, False, True]]
    )):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        layers = ['Conv Block 1', 'Conv Block 2', 'Conv Block 3', 'Classifier']
        colors = ['#90EE90' if t else '#FFB6C1' for t in trainable]
        labels = ['Trainable' if t else 'Frozen' for t in trainable]

        for i, (layer, color, label) in enumerate(zip(layers, colors, labels)):
            y = 8 - i * 2
            rect = FancyBboxPatch((0.5, y), 9, 1.5, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(5, y + 0.75, f'{layer}', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(5, y + 0.2, f'[{label}]', ha='center', va='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/layer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ layer_comparison.png")


if __name__ == "__main__":
    print("Generating diagrams with larger fonts...")
    create_overfitting_graph()
    create_lora_structure()
    create_layer_comparison()
    print("\n🎉 Done!")
