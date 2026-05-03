from ultralytics import YOLO
import torch
import wandb
from datetime import datetime

parameters = {
    "model" : "yolov8s.pt",
    "epochs" : 10,
    "imgsz" : 320,
    "batch" : 2,
    "device" : 0 if torch.cuda.is_available() else "cpu"
}

timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

wandb.init(
    entity="uliege-tennis-tracking",
    project="player-tracking",
    name=f"yolov8n_train_{timestamp}",
    config=parameters
)

print(f'Using device: {parameters['device']}')

model = YOLO(parameters["model"])

training = model.train(
    data="/scratch/users/andyjalloh/player_tracking_yolov8_roboflow_dataset/data.yaml",
    epochs=parameters["epochs"],
    imgsz=parameters["imgsz"],
    batch=parameters["batch"],
    project="/home/andyjalloh/antoine/INFO8010-1_Project/project/code/models/player_tracking",
    name=f"{parameters['model']}_{timestamp}",
    device=parameters["device"],
    plots=True
)

wandb.log({
    "val/precision":  training.results_dict.get("metrics/precision(B)"),
    "val/recall":     training.results_dict.get("metrics/recall(B)"),  
})

wandb.finish()