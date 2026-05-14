import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import TrackNet
from datetime import datetime


class BallDatasetPrediction(Dataset):
    def __init__(self, root_dir, nb_input_frames=3, img_size=(640, 360)):
        self.w, self.h = img_size
        self.nb_input_frames = nb_input_frames

        self.transform = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dataset = []

        for game in sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))):
            game_path = os.path.join(root_dir, game)
            for clip in sorted(d for d in os.listdir(game_path) if os.path.isdir(os.path.join(game_path, d))):
                clip_path = os.path.join(game_path, clip)
                imgs = sorted(
                    f for f in os.listdir(clip_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                )
                for i in range(nb_input_frames - 1, len(imgs)):
                    paths = [os.path.join(clip_path, imgs[i - (nb_input_frames - 1) + j]) for j in range(nb_input_frames)]
                    self.dataset.append((paths, game, clip))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        paths, game, clip = self.dataset[index]
        imgs = [self.transform(Image.open(p).convert("RGB")) for p in paths]
        return torch.cat(imgs, dim=0), os.path.basename(paths[-1]), game, clip


def extract_ball_position(pred_heatmap, threshold=2.55):
    """Returns (x, y) if ball detected above threshold, else None."""
    if pred_heatmap.max() <= threshold:
        return None
    H, W = pred_heatmap.shape
    y, x = divmod(int(np.argmax(pred_heatmap)), W)
    return x, y


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

    parameters = {
        "dataset_dir":    "/scratch/users/andyjalloh/cointe_dataset/",
        "loading_file":   "tracknet_ball_epoch30_11052026_03h27m06s.pth",
        "nb_input_frames": 3,
        "weight_init":    "he",
        "dropout":        True,
        "dropout_p":      0.3,
        "batch_size":     4,
        "num_workers":    2,
        "save_video":     True,
        "output_dir":     f"/home/andyjalloh/andy/INFO8010-1_Project/project/outputs/ball_tracking/prediction_results_{timestamp}",
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    network = TrackNet(
        weight_init=parameters["weight_init"],
        nb_input_frames=parameters["nb_input_frames"],
        dropout=parameters["dropout"],
        dropout_p=parameters["dropout_p"],
    )
    network.load_state_dict(torch.load(
        f"../../models/ball_tracking/{parameters['loading_file']}", map_location=device
    ))
    network.to(device)
    network.eval()

    dataset = BallDatasetPrediction(root_dir=parameters["dataset_dir"])
    loader = DataLoader(dataset, batch_size=parameters["batch_size"],
                        shuffle=False, num_workers=parameters["num_workers"])
    print(f"Dataset: {len(dataset)} samples")

    results = {}  # (game, clip) -> [{img_name, x, y}]

    with torch.no_grad():
        for batch_idx, (x, img_names, games, clips) in enumerate(loader):
            x = x.to(device)
            pred_class = torch.argmax(network(x), dim=1).cpu().numpy()  # (B, H, W)

            for b in range(x.shape[0]):
                pos = extract_ball_position(pred_class[b])
                key = (games[b], clips[b])
                if key not in results:
                    results[key] = []
                results[key].append({
                    "img_name": img_names[b],
                    "x": pos[0] if pos else None,
                    "y": pos[1] if pos else None,
                })

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(loader)}")

    W, H = 640, 360
    for (game, clip), frames in results.items():
        clip_out = os.path.join(parameters["output_dir"], game, clip)
        os.makedirs(clip_out, exist_ok=True)

        pd.DataFrame(frames).to_csv(os.path.join(clip_out, "predictions.csv"), index=False)
        detections = sum(1 for f in frames if f["x"] is not None)
        print(f"{game}/{clip}: {detections}/{len(frames)} detections")

        clip_src = os.path.join(parameters["dataset_dir"], game, clip)
        frames_dir = os.path.join(clip_out, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        rendered = []
        for entry in frames:
            frame = cv2.resize(cv2.imread(os.path.join(clip_src, entry["img_name"])), (W, H))
            if entry["x"] is not None:
                cv2.circle(frame, (entry["x"], entry["y"]), 5, (0, 255, 0), 2)
                cv2.putText(frame, f"({entry['x']},{entry['y']})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO DET", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(frames_dir, entry["img_name"]), frame)
            rendered.append(frame)

        print(f"  Saved frames: {frames_dir}")

        if parameters["save_video"]:
            out = cv2.VideoWriter(
                os.path.join(clip_out, "annotated.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H)
            )
            for frame in rendered:
                out.write(frame)
            out.release()
            print(f"  Saved video: {clip_out}/annotated.mp4")
