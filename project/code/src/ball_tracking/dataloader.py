import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

def generate_gaussian_heatmap(h, w, ball_x, ball_y, visibility, variance): 
    """Generate a 2D Gaussian heatmap at position (cx, cy)."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if visibility == 0 or visibility == 3:
        return heatmap
    x = np.arange(0, w, 1, np.float32)
    y = np.arange(0, h, 1, np.float32)
    y, x = np.meshgrid(y, x, indexing='ij')  # shape: (H, W)
    heatmap = np.exp(-((x - ball_x)**2 + (y - ball_y)**2) / (2 * variance))
    return heatmap


class BallDataset(Dataset):
    def __init__(self, type="train", frame="last", train_coef=0.7, val_coef=0.15, root_dir="/scratch/users/andyjalloh/ball_tracking_kaggle_dataset/",
                 img_size=(640, 360), variance=10, nb_input_frames=3):
        """
        Args:
            gameList: list of the games we want in the form of [game1, game, ...]
            root_dir: path to Dataset/
            train:
            img_size: the ouput size
            variance: variance of the Gaussian heatmap. variance=10 cfr. Tracknet paper
        """
        super().__init__()
        self.root_dir = root_dir
        self.w, self.h = img_size
        self.variance = variance
        self.nb_input_frames = nb_input_frames
        
        self.transform = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Autre valeurs ?
        ])

        self.dataset = []

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
            
            n = len(clip_dir_list)
            train_end = int(train_coef * n)
            val_end   = int((train_coef + val_coef) * n)
            if type == "train":
                clip_dir_list = clip_dir_list[:train_end]
            elif type == "val":
                clip_dir_list = clip_dir_list[train_end:val_end]
            elif type == "test":
                clip_dir_list = clip_dir_list[val_end:]
            
            for clip in clip_dir_list:
                clip_path = os.path.join(game_path, clip)
                df = pd.read_csv(os.path.join(clip_path, "Label.csv"))
                
                if frame == "first": 
                    for i in range(len(df)-self.nb_input_frames+1):
                        list_img_path = []
                        for j in range(nb_input_frames):
                            img_path = os.path.join(clip_path,df.iloc[i + j]["file name"])
                            list_img_path.append(img_path)
                        x = float(df.iloc[i]["x-coordinate"]) * self.w / 1280.0
                        y = float(df.iloc[i]["y-coordinate"]) * self.h / 720.0
                        self.dataset.append((list_img_path, x, y, df.iloc[i]["visibility"]))
                        self.dataset.append((list_img_path, x, y, df.iloc[i]["visibility"]))
                        
                elif frame == "last":
                    for i in range(self.nb_input_frames - 1, len(df)):
                        list_img_path = []
                        for j in range(self.nb_input_frames):
                            idx = i - (self.nb_input_frames - 1) + j
                            img_path = os.path.join(clip_path, df.iloc[idx]["file name"])
                            list_img_path.append(img_path)
                        x = float(df.iloc[i]["x-coordinate"]) * self.w / 1280.0
                        y = float(df.iloc[i]["y-coordinate"]) * self.h / 720.0
                        self.dataset.append((list_img_path, x, y, df.iloc[i]["visibility"]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        list_img_path, x, y, visibility = self.dataset[index]

        list_img = []
        
        for img_path in list_img_path:
            list_img.append(self.transform(Image.open(img_path).convert("RGB")))
        
        heatmap = generate_gaussian_heatmap(self.h, self.w, x, y, visibility, self.variance)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0) # (H, W) -> (1, H, W)

        return torch.cat(list_img, dim=0), heatmap