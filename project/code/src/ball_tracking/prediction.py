# Not tested yet
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd
from PIL import Image
from model import TrackNet
from torch.utils.data import DataLoader
from datetime import datetime

class BallDatasetPrediction(Dataset):
    def __init__(self, root_dir, nb_input_frames=3, img_size=(640, 360), frame="last"):
        self.root_dir = root_dir
        self.nb_input_frames = nb_input_frames
        self.w, self.h = img_size
        self.frame = frame

        self.transform = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataset = []
        
        game_dir_list = sorted([
            g for g in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, g))
        ]) # Besoin de sorted ?
        self.nb_games = len(game_dir_list)
        self.game_dict = {}
        for game in game_dir_list:            
            game_path = os.path.join(root_dir, game)
            clip_dir_list = sorted([
                c for c in os.listdir(game_path)
                if os.path.isdir(os.path.join(game_path, c))
            ]) # Besoin de sorted ?
            self.game_dict[game] = len(clip_dir_list)
            for clip in clip_dir_list:
                clip_path = os.path.join(game_path, clip)
                self.imgs = sorted([f for f in os.listdir(clip_path)])
                if self.frame == "first":
                    for i in range(len(self.imgs)-self.nb_input_frames+1):
                        list_img_path = []
                        for j in range(nb_input_frames):
                            img_path = os.path.join(clip_path, self.imgs[i + j])
                            list_img_path.append(img_path)
                        self.dataset.append((list_img_path, game, clip))
                elif self.frame == "last":
                    for i in range(self.nb_input_frames - 1, len(self.imgs)):
                        list_img_path = []
                        for j in range(nb_input_frames):
                            img_path = os.path.join(clip_path, self.imgs[i - (self.nb_input_frames - 1) + j])
                            list_img_path.append(img_path)
                        self.dataset.append((list_img_path, game, clip))

    def get_nb_clips(self, game):
        return self.game_dict[game]
        
    def get_nb_games(self):
        return self.nb_games
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        list_img_path, game, clip = self.dataset[index]
        list_img = []
        
        for img_path in list_img_path:
            list_img.append(self.transform(Image.open(img_path).convert("RGB")))
        
        if self.frame == "first":
            img_path = list_img_path[0]
        elif self.frame == "last":
            img_path = list_img_path[-1]
        
        # os.path.basename("/.../game1/Clip1/0001.jpg") -> "0001.jpg"
        return torch.cat(list_img, dim=0), os.path.basename(img_path), game, clip

if __name__ == "__main__":
    
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")
    
    parameters ={
        "root_dir" : "scratch/users/andyjalloh/cointe_dataset/",
        "weight_init" : "uniform",
        "nb_input_frames" : 3,
        "dropout" : False,
        "dropout_p" : 0.2 ,
        "shuffle" : False,
        "num_workers" : 0,
        "batch_size" : 2,
        "loading_file" : "tracknet_ball_epoch30_30042026_03h28m14s.pth",
        "output_dir" : f"../../models/ball_tracking/prediction_results_{timestamp}"
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f'Using device: {device}')

    network = TrackNet(weight_init=parameters["weight_init"], nb_input_frames=parameters["nb_input_frames"], 
                   dropout=parameters["dropout"], dropout_p=parameters["dropout_p"])
    
    network.to(device)

    loading_path = f"../../models/ball_tracking/{parameters['loading_file']}"
    network.load_state_dict(torch.load(loading_path, map_location=device))

    network.eval()
    
    prediSet = BallDatasetPrediction(root_dir=parameters["root_dir"])
    
    prediloader = DataLoader(prediSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"], 
                        num_workers=parameters["num_workers"])
    
    print(f"\nTest size: {len(prediloader)}")
    
    output_dir = parameters["output_dir"]
    
    # key: (game, clip) and value: list of dict with 3 keys (img_name, x, y)
    dict = {}

    with torch.no_grad():
        for x, img_names, games, clips in prediloader:
            x = x.to(device)
            pred = network(x)
            pred_class = torch.argmax(pred, dim=1)  # (B, H, W)
            B, _, W = pred_class.shape
            
            for b in range(B):
                pred_idx = torch.argmax(pred_class[b])
                
                # y = int(idx /W) and x = idx % W
                y_pred, x_pred = divmod(pred_idx.item(), W)

                key = (games[b], clips[b])
                if key not in dict:
                    dict[key] = []

                dict[key].append({
                    "img_name" : img_names[b],
                    "x" : x_pred,
                    "y" : y_pred,
                })
                
    for (game, clip), records in dict.items():
        save_dir = os.path.join(output_dir, game, clip)
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
        print(f"Saved {len(records)} predictions → {save_dir}/predictions.csv")