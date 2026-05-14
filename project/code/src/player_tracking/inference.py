from ultralytics import YOLO
import torch
import wandb
from datetime import datetime

parameters = {
    "device":       0 if torch.cuda.is_available() else "cpu",
    "imgsz":        320,
    "loading_file": "yolov8s.pt_12052026_18h22m44s",
}

timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

wandb.init(
    entity="uliege-tennis-tracking",
    project="player-tracking",
    name=f"yolov8s_inference_{timestamp}",
    config=parameters
)

print(f'Using device: {parameters["device"]}')

loading_path = f"/home/andyjalloh/andy/INFO8010-1_Project/project/code/models/player_tracking/{parameters['loading_file']}/weights/best.pt"
model = YOLO(loading_path)

metrics = model.val(
    data="/scratch/users/andyjalloh/player_tracking_yolov8_roboflow_dataset/data.yaml",
    split="test",
    imgsz=parameters["imgsz"],
    device=parameters["device"],
    save_dir=f"/home/andyjalloh/andy/INFO8010-1_Project/project/code/models/player_tracking/inference_results_{timestamp}",
    plots=True
)

print(f"Test precision:{metrics.box.mp}")
print(f"Test recall: {metrics.box.mr}")

wandb.log({
    "test/precision": metrics.box.mp,
    "test/recall" : metrics.box.mr
})

wandb.finish()