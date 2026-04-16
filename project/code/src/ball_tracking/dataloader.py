import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

def generate_gaussian_heatmap(h, w, ball_x, ball_y, visibility, variance): # To check because Claude-generated
    """Generate a 2D Gaussian heatmap at position (cx, cy)."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if visibility == False:
        return heatmap
    x = np.arange(0, w, 1, np.float32)
    y = np.arange(0, h, 1, np.float32)
    y, x = np.meshgrid(y, x, indexing='ij')  # shape: (H, W)
    heatmap = np.exp(-((x - ball_x)**2 + (y - ball_y)**2) / (2 * variance))
    return heatmap


class BallDataset(Dataset):
    def __init__(self, train=True, split=0.7, root_dir="/scratch/users/andyjalloh/Dataset",
                 img_size=(640, 360), variance=10):
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
        
        self.transform = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Autre valeurs ?
        ])

        self.dataset = []

        game_dir_list = sorted(os.listdir(root_dir)) # Besoin de sorted ?
        
        for game in game_dir_list:
            game_path = os.path.join(root_dir, game)
            clip_dir_list = sorted(os.listdir(game_path)) # Besoin de sorted ?
            
            split_idx = int(split * len(clip_dir_list))
            if train:
                clip_dir_list = clip_dir_list[:split_idx]
            else:
                clip_dir_list = clip_dir_list[split_idx:]
            
            for clip in clip_dir_list:
                clip_path = os.path.join(game_path, clip)
                df = pd.read_csv(os.path.join(clip_path, "Label.csv"))
                
                for _, row in df.iterrows():
                    img_path = os.path.join(clip_path, row["file name"])
                    x = float(row["x-coordinate"]) * self.w / 1280.0
                    y = float(row["y-coordinate"]) * self.h / 720.0
                    self.dataset.append((img_path, x, y, row["visibility"], game, clip))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, x, y, visibility, game, clip = self.dataset[index]

        img = self.transform(Image.open(img_path).convert("RGB"))

        heatmap = generate_gaussian_heatmap(self.h, self.w, x, y, visibility, self.variance)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0) # (H, W) -> (1, H, W)

        # Return game and clip because when we will need ithem when we will train
        # to check the consecutive frames
        return img, heatmap, game, clip