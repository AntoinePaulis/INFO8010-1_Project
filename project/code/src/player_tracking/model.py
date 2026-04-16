from ultralytics import YOLO
import wandb

parameters = {
    "model" : "yolov8n.pt",
    "eporchs" : 50,
    "imgsz" : 640,
    "batch" : 16,
}

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="tennis-tracking",
    name="yolov8n_test"
    config=parameters
)

# Charger un modèle YOLOv8 préentraîné
model = YOLO("yolov8n.pt")  # n=nano, s=small, m=medium, l=large, x=xlarge

# Entraîner
model.train(
    data="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/data.yaml",
    epochs=parameters["eporchs"],
    imgsz=parameters["imgsz"],
    batch=parameters["batch"],
    project="runs/tennis_player",
    name="yolov8_exp1"
)

# Évaluation sur le test set
metrics = model.val(split="test")

wandb.finish()

# print the metrics

# add wandb stuf