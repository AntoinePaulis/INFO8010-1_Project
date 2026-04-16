from ultralytics import YOLO

# Path to your dataset YAML on Alan
data_path = "/scratch/users/andyjalloh/Dataset/data.yaml"

# Load model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data=data_path,
    epochs=50,
    imgsz=640,
    batch=16
)