"""
All diagrams for Deepfake Detection Workshop
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTPUT_DIR = "./images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style settings
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


def create_overfitting_graph():
    """과적합 그래프"""
    fig, ax = plt.subplots(figsize=(6, 4))

    epochs = np.arange(1, 11)
    train_loss = 1.0 * np.exp(-0.3 * epochs) + 0.1
    val_loss = np.concatenate([
        1.0 * np.exp(-0.3 * epochs[:5]) + 0.15,
        [0.35, 0.40, 0.50, 0.65, 0.85]
    ])

    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.7, label='Early Stopping')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Overfitting Detection')
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('Overfitting!', xy=(7, 0.55), fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/overfitting_graph.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ overfitting_graph.png")


def create_lora_math():
    """LoRA 수학적 원리"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')

    text = """
    LoRA (Low-Rank Adaptation)

    Traditional:  W' = W + ΔW
                  ΔW is d×d matrix (many parameters)

    LoRA:         W' = W + B × A
                  B: d×r,  A: r×d  (r << d)

    Example: d=1000, r=8
    • Traditional: 1000×1000 = 1,000,000 params
    • LoRA: 1000×8 + 8×1000 = 16,000 params (1.6%)
    """

    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lora_math.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ lora_math.png")


def create_lora_structure():
    """LoRA 구조도"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Boxes
    boxes = [
        (4, 8.5, 2, 0.8, 'Input (x)', 'lightgray'),
        (1, 6, 2, 0.8, 'W (frozen)', 'lightcoral'),
        (7, 6, 2, 0.8, 'A (train)', 'lightgreen'),
        (7, 4, 2, 0.8, 'B (train)', 'lightgreen'),
        (4, 2, 2, 0.8, 'Output (y)', 'lightyellow'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)

    # Arrows
    ax.annotate('', xy=(2, 6.8), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(8, 6.8), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(8, 4.8), xytext=(8, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(5, 2.8), xytext=(2, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(5, 2.8), xytext=(8, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Labels
    ax.text(3, 3.5, 'W×x', fontsize=9)
    ax.text(6.5, 3.5, 'B×A×x', fontsize=9)
    ax.text(5, 1.3, 'W×x + B×A×x', fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lora_structure.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ lora_structure.png")


def create_layer_comparison():
    """Full vs Freeze 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for idx, (ax, title, trainable) in enumerate(zip(
        axes,
        ['Full Fine-tuning', 'Layer Freezing'],
        [[True, True, True, True], [False, False, False, True]]
    )):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

        layers = ['Conv Block 1', 'Conv Block 2', 'Conv Block 3', 'Classifier']
        colors = ['lightgreen' if t else 'lightcoral' for t in trainable]
        labels = ['[Train]' if t else '[Frozen]' for t in trainable]

        for i, (layer, color, label) in enumerate(zip(layers, colors, labels)):
            y = 8 - i * 2
            rect = FancyBboxPatch((1, y), 8, 1.5, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(5, y + 0.75, f'{layer}  {label}', ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/layer_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ layer_comparison.png")


def create_data_structure():
    """데이터 구조"""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axis('off')

    text = """
    data/
    ├── train/
    │   ├── real/    (1,000 images)
    │   └── fake/    (1,000 images)
    ├── val/
    │   ├── real/    (200 images)
    │   └── fake/    (200 images)
    └── test/
        ├── real/    (200 images)
        └── fake/    (200 images)
    """

    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/data_structure.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ data_structure.png")


def create_demo_ui():
    """데모 UI 화면"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Deepfake Detection Demo', ha='center', fontsize=14, fontweight='bold')

    # Image upload box
    rect1 = FancyBboxPatch((1, 2), 5, 4, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor='gray', linewidth=2)
    ax.add_patch(rect1)
    ax.text(3.5, 4, '[Image]\nUpload', ha='center', va='center', fontsize=10, color='gray')
    ax.text(3.5, 1.5, 'Image Upload', ha='center', fontsize=9)

    # Result box
    rect2 = FancyBboxPatch((8, 2), 5, 4, boxstyle="round,pad=0.1",
                           facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(rect2)
    ax.text(10.5, 4.5, 'REAL', ha='center', va='center', fontsize=14, fontweight='bold', color='green')
    ax.text(10.5, 3, 'Confidence: 95%', ha='center', va='center', fontsize=10)
    ax.text(10.5, 1.5, 'Detection Result', ha='center', fontsize=9)

    # Arrow
    ax.annotate('', xy=(7.8, 4), xytext=(6.2, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/demo_ui.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ demo_ui.png")


def create_conclusion_summary():
    """종합 결론"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    text = """
    Workshop Conclusion

    [1] Domain Shift Problem Solved
        - Before (Western faces): ~70%
        - After (Korean faces):   ~90%
        - Improvement: +20%p with Fine-tuning!

    [2] Fine-tuning Technique Selection
        - Best performance -> Full Fine-tuning
        - Fast experiment  -> Layer Freezing
        - Large models     -> LoRA

    [3] Key Takeaway
        Fine-tuning enables domain adaptation
        with minimal data and compute cost.
    """

    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/conclusion_summary.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ conclusion_summary.png")


if __name__ == "__main__":
    print("Generating all diagrams...")
    create_overfitting_graph()
    create_lora_math()
    create_lora_structure()
    create_layer_comparison()
    create_data_structure()
    create_demo_ui()
    create_conclusion_summary()
    print("\n🎉 All diagrams generated!")
