import cv2
import numpy as np

def draw_ball_circle(frame, x, y, radius=5, color=(0, 255, 0), thickness=2):
    """Draw a circle at ball position on frame (in-place)."""
    cv2.circle(frame, (int(x), int(y)), radius, color, thickness)
    return frame

def overlay_heatmap(frame, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on frame with transparency."""
    heatmap_norm = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)