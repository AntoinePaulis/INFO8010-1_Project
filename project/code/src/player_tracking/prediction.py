# Not tested yet
from ultralytics import YOLO
import torch
import os

parameters = {
    "device" : 0 if torch.cuda.is_available() else "cpu",
    "imgsz" : 320,
    "loading_file" : "yolov8n.pt_03052026_11h36m14s",
    "dataset_dir" : "scratch/users/andyjalloh/cointe_dataset/",
    "save_dir"   : f"/home/andyjalloh/antoine/INFO8010-1_Project/project/code/models/player_tracking/prediction_results_{timestamp}",
    "conf" : 0.5,
    "save" : True
}

loading_path = f"/home/andyjalloh/antoine/INFO8010-1_Project/project/code/models/player_tracking/{parameters['loading_file']}/weights/best.pt"

root_dir = parameters["dataset_dir"]

print(f'Using device: {parameters['device']}')

model = YOLO(loading_path)

game_dir_list = sorted([
    g for g in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, g))
]) # Besoin de sorted ?

for game in game_dir_list:            
    game_path = os.path.join(root_dir, game)
    clip_dir_list = sorted([
        c for c in os.listdir(game_path)
        if os.path.isdir(os.path.join(game_path, c))
    ]) # Besoin de sorted ?
    for clip in clip_dir_list:
        clip_path = os.path.join(game_path, clip)

        results = model.predict(
            source=clip_path,
            imgsz=parameters["imgsz"],
            conf=parameters["conf"],
            device=parameters["device"],
            save_dir=os.path.join(parameters["output_dir"], game, clip),
            save=True,
            plots=True,
            stream=True # Not sur we need it
        )

        print(f"Prediction of {clip} of {game} is done")