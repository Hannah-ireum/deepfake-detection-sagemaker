"""Better Demo UI diagram"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title bar
title_box = FancyBboxPatch((0.5, 8.5), 15, 1.2, boxstyle="round,pad=0.05",
                            facecolor='#2196F3', edgecolor='#1976D2', linewidth=2)
ax.add_patch(title_box)
ax.text(8, 9.1, 'Deepfake Detection Demo', ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

# Main container
main_box = FancyBboxPatch((0.5, 0.5), 15, 7.8, boxstyle="round,pad=0.05",
                           facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=2)
ax.add_patch(main_box)

# Left panel - Image Upload
left_panel = FancyBboxPatch((1, 1), 6.5, 7, boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor='#BDBDBD', linewidth=1.5)
ax.add_patch(left_panel)

# Upload area (dashed)
upload_area = FancyBboxPatch((1.5, 2), 5.5, 4.5, boxstyle="round,pad=0.05",
                              facecolor='#F5F5F5', edgecolor='#9E9E9E',
                              linewidth=1.5, linestyle='--')
ax.add_patch(upload_area)

ax.text(4.25, 4.5, 'Drag & Drop', ha='center', va='center', fontsize=12, color='#757575')
ax.text(4.25, 3.8, 'or Click to Upload', ha='center', va='center', fontsize=10, color='#9E9E9E')

ax.text(4.25, 7.5, 'Image Upload', ha='center', va='center', fontsize=12, fontweight='bold')

# Right panel - Result
right_panel = FancyBboxPatch((8.5, 1), 6.5, 7, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='#BDBDBD', linewidth=1.5)
ax.add_patch(right_panel)

ax.text(11.75, 7.5, 'Detection Result', ha='center', va='center', fontsize=12, fontweight='bold')

# Result box - REAL
result_box = FancyBboxPatch((9, 3.5), 5.5, 3, boxstyle="round,pad=0.1",
                             facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
ax.add_patch(result_box)

ax.text(11.75, 5.5, 'REAL', ha='center', va='center', fontsize=20, fontweight='bold', color='#2E7D32')
ax.text(11.75, 4.3, 'Confidence: 95.3%', ha='center', va='center', fontsize=12, color='#388E3C')

# Progress bar
progress_bg = FancyBboxPatch((9, 2), 5.5, 0.4, boxstyle="round,pad=0.02",
                              facecolor='#E0E0E0', edgecolor='none')
ax.add_patch(progress_bg)
progress_fill = FancyBboxPatch((9, 2), 5.2, 0.4, boxstyle="round,pad=0.02",
                                facecolor='#4CAF50', edgecolor='none')
ax.add_patch(progress_fill)

# Buttons
submit_btn = FancyBboxPatch((2, 1.2), 2.5, 0.6, boxstyle="round,pad=0.05",
                             facecolor='#2196F3', edgecolor='#1976D2', linewidth=1)
ax.add_patch(submit_btn)
ax.text(3.25, 1.5, 'Submit', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

clear_btn = FancyBboxPatch((5, 1.2), 2.5, 0.6, boxstyle="round,pad=0.05",
                            facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1)
ax.add_patch(clear_btn)
ax.text(6.25, 1.5, 'Clear', ha='center', va='center', fontsize=10, color='#616161', fontweight='bold')

plt.tight_layout()
plt.savefig('./images/demo_ui.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ demo_ui.png created!")
