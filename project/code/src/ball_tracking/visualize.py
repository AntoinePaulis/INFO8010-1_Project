import os
import cv2
import numpy as np
import torch
from dataloader import BallDataset
import sys
sys.path.append('..')
from config import OUTPUTS_DIR
from utils.draw import draw_ball_circle
from datetime import datetime

# Claude generated
def extract_ball_position(pred_heatmap):
    """
    Extract ball position using Hough circle detection.
    
    Args:
        pred_heatmap: (H, W) numpy array, values 0-255
    
    Returns:
        (x, y) tuple if exactly one ball detected, else None
    """
    binary_pred = np.where(pred_heatmap >= 128, 255, 0).astype(np.uint8)
    
    circles = cv2.HoughCircles(
        binary_pred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=10,
        minRadius=2,
        maxRadius=12
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        if len(circles) == 1:
            x, y, r = circles[0]
            return (x, y)
    
    return None

# Claude generated
def visualize_from_predictions(predictions_file, dataset, output_dir, clip_idx=0, save_video=False):
    """
    Load saved predictions and visualize specific clip with metrics overlay.
    """
    # Load predictions
    data = torch.load(predictions_file, map_location='cpu')
    predictions = data['predictions']  # (N, H, W)
    ground_truths = data['ground_truths']  # (N, 1, H, W)
    dataset_indices = data['dataset_indices']
    
    print(f"Loaded {len(predictions)} predictions from {predictions_file}")
    print(f"Model: {data['model_file']}, Metrics: F1={data['metrics']['f1']:.3f}")
    
    # Group dataset indices by clip
    clips = {}
    for idx in dataset_indices:
        img_paths, _, _, _ = dataset.dataset[idx]
        clip_path = os.path.dirname(img_paths[0])
        
        if clip_path not in clips:
            clips[clip_path] = []
        clips[clip_path].append(idx)
    
    clip_paths = list(clips.keys())
    if clip_idx >= len(clip_paths):
        print(f"Error: clip_idx={clip_idx} but only {len(clip_paths)} clips available")
        return
    
    selected_clip_path = clip_paths[clip_idx]
    clip_indices = clips[selected_clip_path]
    clip_name = os.path.basename(selected_clip_path)
    
    print(f"Visualizing clip {clip_idx}: {clip_name} ({len(clip_indices)} frames)")
    
    # Create output directory
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")
    clip_output_dir = os.path.join(output_dir, f"{clip_name}_{timestamp}")
    os.makedirs(clip_output_dir, exist_ok=True)
    
    # Track detection stats
    detection_count = 0
    
    for frame_num, idx in enumerate(clip_indices):
        # Get prediction for this index
        pred_idx = dataset_indices.index(idx)
        pred_heatmap = predictions[pred_idx].numpy().astype(np.uint8)  # (H, W)
        gt_heatmap = ground_truths[pred_idx, 0].numpy()  # (H, W)
        
        # Load original frame
        img_paths, _, _, _ = dataset.dataset[idx]
        last_frame = cv2.imread(img_paths[-1])  # last of 3 input frames
        last_frame = cv2.resize(last_frame, (dataset.w, dataset.h))
        
        # Extract ball position from prediction
        ball_pos = extract_ball_position(pred_heatmap)
        
        # Create heatmap visualization
        heatmap_vis = cv2.applyColorMap(pred_heatmap, cv2.COLORMAP_JET)
        heatmap_vis = cv2.resize(heatmap_vis, (dataset.w, dataset.h))
        
        # Annotate frame
        frame_annotated = last_frame.copy()
        detected = ball_pos is not None
        
        if detected:
            x_ball, y_ball = ball_pos
            draw_ball_circle(frame_annotated, x_ball, y_ball, radius=5, color=(0, 255, 0), thickness=2)
            detection_count += 1
            status_text = f"DETECTED at ({x_ball}, {y_ball})"
            status_color = (0, 255, 0)
        else:
            status_text = "NO DETECTION"
            status_color = (0, 0, 255)
        
        # Add text overlay
        cv2.putText(frame_annotated, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame_annotated, f"Frame {frame_num}/{len(clip_indices)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create side-by-side: original + heatmap
        combined = np.hstack([frame_annotated, heatmap_vis])
        
        # Save frame
        frame_path = os.path.join(clip_output_dir, f"frame_{frame_num:04d}.jpg")
        cv2.imwrite(frame_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    print(f"Saved {len(clip_indices)} frames to {clip_output_dir}")
    print(f"Detection rate: {detection_count}/{len(clip_indices)} ({100*detection_count/len(clip_indices):.1f}%)")
    
    # Optionally save video
    if save_video:
        video_path = os.path.join(clip_output_dir, "clip.mp4")
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(clip_output_dir, "frame_0000.jpg"))
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        
        for i in range(len(clip_indices)):
            frame = cv2.imread(os.path.join(clip_output_dir, f"frame_{i:04d}.jpg"))
            out.write(frame)
        
        out.release()
        print(f"Saved video: {video_path}")

if __name__ == "__main__":
    # Load test set
    testSet = BallDataset(type="test", train_coef=0.7, val_coef=0.15, 
                          nb_input_frames=3, variance=10, frame="last")
    
    # Load predictions
    predictions_file = os.path.join(OUTPUTS_DIR, "ball_tracking", "predictions", 
                                    "predictions_06052026_00h06m41s.pt")
    
    data = torch.load(predictions_file, map_location='cpu')
    predictions = data['predictions']
    dataset_indices = data['dataset_indices']
    
    # Count detections per clip using pre-computed metrics
    from collections import defaultdict
    clip_detections = defaultdict(int)
    clip_frames = defaultdict(list)
    
    sample_metrics = data['sample_metrics']
    
    for metric in sample_metrics:
        idx = metric['dataset_idx']
        img_paths, _, _, _ = testSet.dataset[idx]
        clip_name = os.path.basename(os.path.dirname(img_paths[0]))
        
        clip_frames[clip_name].append(idx)
        
        if metric['detected']:
            clip_detections[clip_name] += 1
    
    # Find best clip
    best_clip_name = max(clip_detections.items(), key=lambda x: x[1])[0]
    best_clip_idx = list(clip_frames.keys()).index(best_clip_name)
    
    print(f"Auto-selected best clip: {best_clip_name} (index {best_clip_idx})")
    print(f"Detections: {clip_detections[best_clip_name]}/{len(clip_frames[best_clip_name])}")
    
    output_dir = os.path.join(OUTPUTS_DIR, "ball_tracking", "visualizations")
    
    visualize_from_predictions(predictions_file, testSet, output_dir, 
                               clip_idx=best_clip_idx, save_video=False)