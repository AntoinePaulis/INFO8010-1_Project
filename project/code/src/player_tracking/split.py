# From https://roboflow.com/split-datasets/yolo
# To check : LS=70%, TS=20% and VS=10%
import supervision as sv

ds = sv.DetectionDataset.from_yolo(
    images_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/train/images",
    annotations_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/train/labels",
    data_yaml_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/data.yaml"
)

train_ds, temp_ds = ds.split(split_ratio=0.7,random_state=42, shuffle=True)

val_ds, test_ds = temp_ds.split(split_ratio=0.67, random_state=42, shuffle=True)

print(len(train_ds), len(val_ds), len(test_ds))

train_ds.as_yolo(
    images_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/train/images",
    annotations_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/train/labels",
)

val_ds.as_yolo(
    images_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/val/images",
    annotations_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/val/labels",
)

test_ds.as_yolo(
    images_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/test/images",
    annotations_directory_path="/scratch/users/andyjalloh/Tennis_Player_Detection.yolov8/test/labels",
)