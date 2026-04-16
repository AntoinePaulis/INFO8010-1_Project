from ultralytics import YOLO

# Charger un modèle YOLOv8 préentraîné
model = YOLO("yolov8n.pt")  # n=nano, s=small, m=medium, l=large, x=xlarge

# Entraîner
model.train(
    data="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/tennis_player",
    name="yolov8_exp1"
)

# Évaluation sur le test set
metrics = model.val()


"""
# Inférence sur une image
results = model.predict("chemin/vers/image.jpg", conf=0.5)
results[0].show()
"""