import os
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime

timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

parameters = {
    "device":       0 if torch.cuda.is_available() else "cpu",
    "imgsz":        320,
    "loading_file": "yolov8s.pt_12052026_18h22m44s",
    "dataset_dir":  "/scratch/users/andyjalloh/cointe_dataset/",
    "conf":         0.5,
    "save_video":   True,
    "output_dir":   f"/home/andyjalloh/andy/INFO8010-1_Project/project/outputs/player_tracking/prediction_results_{timestamp}",
}

print(f"Using device: {parameters['device']}")

loading_path = f"/home/andyjalloh/andy/INFO8010-1_Project/project/code/models/player_tracking/{parameters['loading_file']}/weights/best.pt"
model = YOLO(loading_path)

root_dir = parameters["dataset_dir"]

for game in sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))):
    game_path = os.path.join(root_dir, game)
    for clip in sorted(d for d in os.listdir(game_path) if os.path.isdir(os.path.join(game_path, d))):
        clip_path = os.path.join(game_path, clip)
        clip_out = os.path.join(parameters["output_dir"], game, clip)
        frames_dir = os.path.join(clip_out, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        if not any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(clip_path)):
            print(f"{game}/{clip}: no images, skipping")
            continue

        rendered = []
        for result in model.predict(
            source=clip_path,
            imgsz=parameters["imgsz"],
            conf=parameters["conf"],
            device=parameters["device"],
            stream=True,
            verbose=False,
        ):
            frame = result.plot()  # BGR numpy array with boxes drawn
            fname = os.path.basename(result.path)
            cv2.imwrite(os.path.join(frames_dir, fname), frame)
            rendered.append(frame)

        print(f"{game}/{clip}: {len(rendered)} frames saved")

        if parameters["save_video"] and rendered:
            H, W = rendered[0].shape[:2]
            out = cv2.VideoWriter(
                os.path.join(clip_out, "annotated.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H)
            )
            for frame in rendered:
                out.write(frame)
            out.release()
            print(f"  Saved video: {clip_out}/annotated.mp4")
