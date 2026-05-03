# Not tested yet
from ultralytics import YOLO
import torch

parameters = {
    "device" : 0 if torch.cuda.is_available() else "cpu",
    "imgsz" : 320,
    "loading_file" : "yolov8n.pt_03052026_11h36m14s",
    "folder_clip" : "scratch/users/andyjalloh/cointe_dataset/clip_test",
    "conf" : 0.5,
    "save" : True
}

loading_path = f"/home/andyjalloh/antoine/INFO8010-1_Project/project/code/models/player_tracking/{parameters['loading_file']}/weights/best.pt"

print(f'Using device: {parameters['device']}')

model = YOLO(loading_path)

results = model.predict(
    source=parameters["folder_clip"],
    imgsz=parameters["imgsz"],
    conf=parameters["conf"],
    device=parameters["device"],
    save_dir="/home/andyjalloh/antoine/INFO8010-1_Project/project/code/models/player_tracking/prediction_results",
    save=True,
    stream=True
)

results = model.predict(
    source=parameters["folder_clip"],
    imgsz=parameters["imgsz"],
    conf=parameters["conf"],
    device=parameters["device"],
    save_dir="/home/andyjalloh/antoine/INFO8010-1_Project/project/code/models/player_tracking/prediction_results",
    save=True,
    stream=False
)

print(f"Prediction of {parameters['folder_clip']} done")