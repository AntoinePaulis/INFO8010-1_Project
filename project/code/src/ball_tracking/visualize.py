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

def extract_ball_position(pred_heatmap, threshold=2.55):
    if pred_heatmap.max() <= threshold:
        return None
    
    H, W = pred_heatmap.shape
    flat_idx = np.argmax(pred_heatmap)
    y, x = np.unravel_index(flat_idx, (H, W))  # ← Use unravel_index, not divmod
    return (int(x), int(y))

def visualize_from_predictions(predictions_file, dataset, output_dir, clip_idx=0, save_video=False):
    """
    Load saved predictions and visualize specific clip with metrics overlay.
    Shows: original frame | raw heatmap | model output with detection
    """
    data = torch.load(predictions_file, map_location='cpu')
    predictions = data['predictions']  # (N, H, W)
    ground_truths = data['ground_truths']  # (N, 1, H, W)
    dataset_indices = data['dataset_indices']
    
    print(f"Loaded {len(predictions)} predictions from {predictions_file}")
    print(f"Model: {data['model_file']}, Metrics: F1={data['metrics']['f1']:.3f}")
    
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
    
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")
    clip_output_dir = os.path.join(output_dir, f"{clip_name}_{timestamp}")
    os.makedirs(clip_output_dir, exist_ok=True)
    
    detection_count = 0
    
    for frame_num, idx in enumerate(clip_indices):
        pred_idx = dataset_indices.index(idx)
        pred_heatmap = predictions[pred_idx].numpy().astype(np.uint8)  # (H, W)
        
        # Load original frame
        img_paths, _, _, _ = dataset.dataset[idx]
        last_frame = cv2.imread(img_paths[-1])
        last_frame = cv2.resize(last_frame, (dataset.w, dataset.h))
        print(f"Frame shape: {last_frame.shape[:2]}")  # Should be (dataset.h, dataset.w)
        
        # Extract ball position
        ball_pos = extract_ball_position(pred_heatmap, threshold=2.55)
        
        # Create heatmap visualization
        heatmap_vis = cv2.applyColorMap(pred_heatmap, cv2.COLORMAP_JET)
        heatmap_vis = cv2.resize(heatmap_vis, (dataset.w, dataset.h))
        
        # Create model output frame (frame + detection overlay)
        output_frame = last_frame.copy()
        detected = ball_pos is not None

        # Add to visualize_from_predictions, inside the frame loop:
        print(f"Frame {frame_num}: pred max={pred_heatmap.max()}, argmax pos={np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)}")
        print(f"  GT max={ground_truths[pred_idx, 0].numpy().max()}, argmax pos={np.unravel_index(np.argmax(ground_truths[pred_idx, 0].numpy()), ground_truths[pred_idx, 0].numpy().shape)}")
        
        if detected:
            x_ball, y_ball = ball_pos
            draw_ball_circle(output_frame, x_ball, y_ball, radius=5, color=(0, 255, 0), thickness=2)
            detection_count += 1
            status_text = f"DETECTED ({x_ball}, {y_ball})"
            status_color = (0, 255, 0)
        else:
            status_text = "NO DETECTION"
            status_color = (0, 0, 255)
        
        cv2.putText(output_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(output_frame, f"Frame {frame_num}/{len(clip_indices)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Three panels: original | heatmap | output
        combined = np.hstack([last_frame, heatmap_vis, output_frame])
        
        frame_path = os.path.join(clip_output_dir, f"frame_{frame_num:04d}.jpg")
        cv2.imwrite(frame_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    print(f"Saved {len(clip_indices)} frames to {clip_output_dir}")
    print(f"Detection rate: {detection_count}/{len(clip_indices)} ({100*detection_count/len(clip_indices):.1f}%)")
    
    if save_video:
        video_path = os.path.join(clip_output_dir, "clip.mp4")
        first_frame = cv2.imread(os.path.join(clip_output_dir, "frame_0000.jpg"))
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        
        for i in range(len(clip_indices)):
            frame = cv2.imread(os.path.join(clip_output_dir, f"frame_{i:04d}.jpg"))
            out.write(frame)
        
        out.release()
        print(f"Saved video: {video_path}")

def compute_clip_tp(predictions, ground_truths, dataset_indices, dataset, threshold=5):
    """
    Compute true positives per clip by checking pixel distance between pred and GT.
    """
    from collections import defaultdict
    clip_tp = defaultdict(int)
    clip_frames = defaultdict(list)
    
    for i, idx in enumerate(dataset_indices):
        img_paths, _, _, vis = dataset.dataset[idx]
        clip_name = os.path.basename(os.path.dirname(img_paths[0]))
        clip_frames[clip_name].append(idx)
        
        # Skip if ball not visible
        if vis == 0 or vis == 3:
            continue
            
        pred_heatmap = predictions[i].numpy()
        gt_heatmap = ground_truths[i, 0].numpy()
        
        # Check if ball detected
        if pred_heatmap.max() <= 2.55:
            continue
        
        # Get predicted and ground truth positions
        H, W = pred_heatmap.shape
        pred_idx = np.argmax(pred_heatmap)
        gt_idx = np.argmax(gt_heatmap)
        
        pred_y, pred_x = divmod(pred_idx, W)
        gt_y, gt_x = divmod(gt_idx, W)
        
        dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5
        
        if dist < threshold:
            clip_tp[clip_name] += 1
    
    return clip_tp, clip_frames

if __name__ == "__main__":
    testSet = BallDataset(type="test", train_coef=0.7, val_coef=0.15, 
                          nb_input_frames=3, variance=10, frame="last")
    
    print(f"Dataset dimensions: w={testSet.w}, h={testSet.h}")
    
    predictions_file = os.path.join(OUTPUTS_DIR, "ball_tracking", "predictions", 
                                    "predictions_11052026_11h59m45s.pt")
    
    data = torch.load(predictions_file, map_location='cpu')
    predictions = data['predictions']
    ground_truths = data['ground_truths']
    dataset_indices = data['dataset_indices']
    
    # Compute TP per clip
    clip_tp, clip_frames = compute_clip_tp(predictions, ground_truths, dataset_indices, testSet)
    
    # Find clip with most TPs
    best_clip_name = max(clip_tp.items(), key=lambda x: x[1])[0]
    best_clip_idx = list(clip_frames.keys()).index(best_clip_name)
    
    print(f"Auto-selected best clip: {best_clip_name} (index {best_clip_idx})")
    print(f"True Positives: {clip_tp[best_clip_name]}/{len(clip_frames[best_clip_name])}")
    
    output_dir = os.path.join(OUTPUTS_DIR, "ball_tracking", "visualizations")
    
    visualize_from_predictions(predictions_file, testSet, output_dir, 
                               clip_idx=best_clip_idx, save_video=False)