import numpy as np
import matplotlib.pyplot as plt
from dataloader import generate_gaussian_heatmap

def analyze_heatmap_variance():
    """Visualize ground truth heatmaps with different variances"""
    h, w = 360, 640
    center_x, center_y = 320, 180
    
    variances = [7, 10, 15]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, var in enumerate(variances):
        # Generate heatmap (will be [0, 1] with current code, [0, 255] after fix)
        heatmap = generate_gaussian_heatmap(h, w, center_x, center_y, visibility=1, variance=var)
        
        # Calculate effective radius at 50% of peak
        threshold = heatmap.max() * 0.5
        radius = np.sqrt(2 * var * np.log(2))
        
        # Top row: full heatmap
        axes[0, idx].imshow(heatmap, cmap='hot', vmin=0, vmax=heatmap.max())
        axes[0, idx].set_title(f'Variance={var}, Peak={heatmap.max():.1f}\nRadius@50%={radius:.2f}px')
        plt.colorbar(axes[0, idx].images[0], ax=axes[0, idx])
        
        # Bottom row: zoomed 40x40 crop
        crop = heatmap[center_y-20:center_y+20, center_x-20:center_x+20]
        im = axes[1, idx].imshow(crop, cmap='hot', vmin=0, vmax=heatmap.max())
        
        # Draw circles for ball size reference
        circle_small = plt.Circle((20, 20), 2, fill=False, color='cyan', linewidth=2, label='2px ball')
        circle_mean = plt.Circle((20, 20), 2.5, fill=False, color='yellow', linewidth=2, label='5px ball')
        circle_large = plt.Circle((20, 20), 6, fill=False, color='red', linewidth=2, label='12px ball')
        circle_heatmap = plt.Circle((20, 20), radius, fill=False, color='white', 
                                   linewidth=2, linestyle='--', label=f'Heatmap radius')
        
        axes[1, idx].add_patch(circle_small)
        axes[1, idx].add_patch(circle_mean)
        axes[1, idx].add_patch(circle_large)
        axes[1, idx].add_patch(circle_heatmap)
        axes[1, idx].set_title(f'Zoomed (40x40px)')
        if idx == 2:
            axes[1, idx].legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=axes[1, idx])
    
    plt.tight_layout()
    plt.savefig('heatmap_variance_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved heatmap_variance_analysis.png")
    
    # Print summary
    print("\n" + "="*60)
    print("HEATMAP VARIANCE ANALYSIS")
    print("="*60)
    print(f"Ball diameter in dataset: 2-12px (mean ~5px)")
    print(f"\nRecommendations:")
    for var in variances:
        radius = np.sqrt(2 * var * np.log(2))
        diameter = 2 * radius
        print(f"  Variance={var:2d} → Heatmap diameter={diameter:.1f}px ", end="")
        if 4.5 <= diameter <= 6.5:
            print("← RECOMMENDED (matches mean ball size)")
        elif diameter < 4.5:
            print("(too small, may miss larger balls)")
        else:
            print("(may blur small balls)")

if __name__ == "__main__":
    analyze_heatmap_variance()