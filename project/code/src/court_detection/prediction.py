import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import TrackNetCourt
from datetime import datetime


class CourtDatasetPrediction(Dataset):
    def __init__(self, root_dir, img_size=(320, 176)):
        self.w, self.h = img_size

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
                for fname in sorted(
                    f for f in os.listdir(clip_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ):
                    self.dataset.append((os.path.join(clip_path, fname), fname, game, clip))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, fname, game, clip = self.dataset[index]
        return self.transform(Image.open(path).convert("RGB")), fname, game, clip


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

    parameters = {
        "dataset_dir":  "/scratch/users/andyjalloh/cointe_dataset/",
        "loading_file": "tracknet_court_epoch100_13052026_14h35m50s.pth",
        "weight_init":  "he",
        "dropout":      True,
        "dropout_p":    0.3,
        "img_size":     (320, 176),
        "batch_size":   8,
        "num_workers":  2,
        "save_video":   True,
        "output_dir":   f"/home/andyjalloh/andy/INFO8010-1_Project/project/outputs/court_detection/prediction_results_{timestamp}",
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    network = TrackNetCourt(
        weight_init=parameters["weight_init"],
        dropout=parameters["dropout"],
        dropout_p=parameters["dropout_p"],
    )
    network.load_state_dict(torch.load(
        f"../../models/court_detection/{parameters['loading_file']}", map_location=device
    ))
    network.to(device)
    network.eval()

    dataset = CourtDatasetPrediction(root_dir=parameters["dataset_dir"], img_size=parameters["img_size"])
    loader = DataLoader(dataset, batch_size=parameters["batch_size"],
                        shuffle=False, num_workers=parameters["num_workers"])
    print(f"Dataset: {len(dataset)} frames")

    results = {}  # (game, clip) -> [{img_name, kp0_x, kp0_y, ..., kp14_x, kp14_y}]

    with torch.no_grad():
        for batch_idx, (x, img_names, games, clips) in enumerate(loader):
            x = x.to(device)
            pred = network(x).cpu()  # (B, 15, H, W)
            B, nb_kps, _, W = pred.shape

            for b in range(B):
                row = {"img_name": img_names[b]}
                for k in range(nb_kps):
                    y_pred, x_pred = divmod(int(torch.argmax(pred[b, k])), W)
                    row[f"kp{k}_x"] = x_pred
                    row[f"kp{k}_y"] = y_pred

                key = (games[b], clips[b])
                if key not in results:
                    results[key] = []
                results[key].append(row)

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(loader)}")

    W, H = parameters["img_size"]
    for (game, clip), frames in results.items():
        clip_out = os.path.join(parameters["output_dir"], game, clip)
        os.makedirs(clip_out, exist_ok=True)

        pd.DataFrame(frames).to_csv(os.path.join(clip_out, "predictions.csv"), index=False)
        print(f"{game}/{clip}: {len(frames)} frames saved")

        clip_src = os.path.join(parameters["dataset_dir"], game, clip)
        frames_dir = os.path.join(clip_out, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        kp_cols = [(f"kp{k}_x", f"kp{k}_y") for k in range(15)]
        rendered = []
        for entry in frames:
            frame = cv2.resize(cv2.imread(os.path.join(clip_src, entry["img_name"])), (W, H))
            for kx_col, ky_col in kp_cols:
                cv2.circle(frame, (entry[kx_col], entry[ky_col]), 4, (0, 255, 0), -1)
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
