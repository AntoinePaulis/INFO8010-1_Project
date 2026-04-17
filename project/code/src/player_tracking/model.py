from ultralytics import YOLO
import wandb

# Now test different YOLO and different config

parameters = {
    "model" : "yolov8n.pt",
    "eporchs" : 1,
    "imgsz" : 320,
    "batch" : 2,
    "device" : "cpu"
}

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="player-tracking",
    name="yolov8n_test",
    config=parameters
)

model = YOLO("yolov8n.pt")

model.train(
    data="/scratch/users/andyjalloh/player_tracking/Tennis_Player_Detection.yolov8/data.yaml",
    epochs=parameters["eporchs"],
    imgsz=parameters["imgsz"],
    batch=parameters["batch"],
    project="runs/tennis_player",
    name="yolov8_test",
    device=parameters["device"]
)

# Évaluation sur le test set
metrics = model.val(split="test")

table = wandb.Table(columns=["Metric", "Value"])
table.add_data("mAP50-95",metrics.box.map)
table.add_data("mAP50",metrics.box.map50)
table.add_data("mAP75",metrics.box.map75)
table.add_data("Precision",metrics.box.mp)
table.add_data("Recall",metrics.box.mr)
table.add_data("F1",metrics.box.f1.mean())

# Further add loss if necessary

wandb.log({"Test Metrics": table})
wandb.finish()