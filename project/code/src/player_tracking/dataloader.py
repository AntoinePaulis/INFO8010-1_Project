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
    def __init__(selfroot_dir="/scratch/users/andyjalloh/Dataset"):
        """
        Args:
            gameList: list of the games we want in the form of [game1, game, ...]
            root_dir: path to Dataset/
            train: if True use games 1-8, else games 9-10 (adjust as needed)
            img_size: the ouput size
            variance: variance of the Gaussian heatmap. variance=10 cfr. Tracknet paper
        """
        super().__init__()

    def __len__(self):
        return

    def __getitem__(self, index):
        return