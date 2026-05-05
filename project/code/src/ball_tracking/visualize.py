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
    Load saved predictions and visualize specific clip.
    
    Args:
        predictions_file: path to saved .pt file from inference.py
        dataset: BallDataset instance (same one used in inference)
        output_dir: directory to save frames
        clip_idx: which clip to visualize (0 = first clip in test set)
        save_video: if True, also create mp4
    """
    # Load predictions
    data = torch.load(predictions_file, map_location='cpu')
    predictions = data['predictions']  # (N, H, W)
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
    
    annotated_frames = []
    
    for idx in clip_indices:
        # Get prediction for this index
        pred_idx = dataset_indices.index(idx)
        pred_heatmap = predictions[pred_idx].numpy().astype(np.uint8)  # (H, W)
        
        # Load original frame
        img_paths, _, _, _ = dataset.dataset[idx]
        last_frame = cv2.imread(img_paths[-1])  # last of 3 input frames
        last_frame = cv2.resize(last_frame, (dataset.w, dataset.h))
        
        # Extract and draw ball position
        ball_pos = extract_ball_position(pred_heatmap)
        
        if ball_pos is not None:
            x_ball, y_ball = ball_pos
            draw_ball_circle(last_frame, x_ball, y_ball, radius=5, color=(0, 255, 0), thickness=2)
        
        annotated_frames.append(last_frame)
    
    # Save individual frames
    for i, frame in enumerate(annotated_frames):
        frame_path = os.path.join(clip_output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    print(f"Saved {len(annotated_frames)} frames to {clip_output_dir}")
    
    # Optionally save video
    if save_video and annotated_frames:
        video_path = os.path.join(clip_output_dir, "clip.mp4")
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        
        for frame in annotated_frames:
            out.write(frame)
        
        out.release()
        print(f"Saved video: {video_path}")

if __name__ == "__main__":
    # Load test set (must match inference.py setup)
    testSet = BallDataset(type="test", train_coef=0.7, val_coef=0.15, 
                          nb_input_frames=3, variance=10, frame="last")
    
    # Path to predictions from inference.py
    predictions_file = os.path.join(OUTPUTS_DIR, "ball_tracking", "predictions", 
                                    "predictions_05052026_14h30m00s.pt")  # UPDATE THIS
    
    output_dir = os.path.join(OUTPUTS_DIR, "ball_tracking", "visualizations")
    
    visualize_from_predictions(predictions_file, testSet, output_dir, clip_idx=0, save_video=False)