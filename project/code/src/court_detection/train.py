import wandb
import torch
from model import TrackNetCourt

import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import CourtDataset
import os
from datetime import datetime
from accelerate import Accelerator
import argparse

# weighted loss was a claude suggestion
def weighted_mse_loss(pred, target, pos_weight=1000):
    weights = 1 + pos_weight * target
    return (weights * (pred - target) ** 2).mean()

def compute_court_metrics(pred, y, threshold=7):
    B, nb_kps, _, W = pred.shape
    TP, TN, FP, FN = 0, 0, 0, 0

    for b in range(B):
        for i in range(nb_kps):
            gt_heatmap = y[b, i]
            pred_heatmap = pred[b, i]

            gt_idx = torch.argmax(gt_heatmap)
            pred_idx = torch.argmax(pred_heatmap)

            gt_y_coord, gt_x_coord = divmod(gt_idx.item(), W)
            pred_y_coord, pred_x_coord = divmod(pred_idx.item(), W)

            dist = math.dist([gt_x_coord, gt_y_coord], [pred_x_coord, pred_y_coord])

            if dist < threshold:
                TP += 1
            else:
                FP += 1
                FN += 1
    # we never increase true negatives since theres never a frame where the court is absent
    return TP, TN, FP, FN

def train(num_epochs, accelerator=None):
    train_avg_loss = []
    val_avg_loss = []

    for i in range(num_epochs):
        train_losses = []
        val_losses = []
        val_mae = []

        network.train()
        TP, TN, FP, FN = 0, 0, 0, 0

        for x, y in trainloader:
            if accelerator is None:
                x = x.to(device)
                y = y.to(device)

            pred = network(x)
            loss = weighted_mse_loss(pred, y)
            train_losses.append(loss.detach())

            optimizer.zero_grad()

            if accelerator is not None:
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(network.parameters(), max_norm=1.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

            optimizer.step()
            torch.cuda.empty_cache()

            batch_idx = len(train_losses)
            if batch_idx % 100 == 0:
                print(f"Epoch {i} - Train batch {batch_idx}/{len(trainloader)} ({100*batch_idx/len(trainloader):.1f}%)")

        network.eval()
        with torch.no_grad():

            for x, y in valloader:
                if accelerator is None:
                    x = x.to(device)
                    y = y.to(device)

                pred = network(x)

                loss = weighted_mse_loss(pred, y)
                mae = F.l1_loss(pred, y)
                val_losses.append(loss.detach())
                val_mae.append(mae.item())

                TP_i, TN_i, FP_i, FN_i = compute_court_metrics(pred, y)
                TP += TP_i
                TN += TN_i
                FP += FP_i
                FN += FN_i

                batch_idx = len(val_losses)
                if batch_idx % 50 == 0:
                    print(f"Epoch {i} - Val batch {batch_idx}/{len(valloader)} ({100*batch_idx/len(valloader):.1f}%)")

        if parameters["scheduler"] == True:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        epoch_train_loss = torch.mean(torch.tensor(train_losses))
        epoch_val_loss = torch.mean(torch.tensor(val_losses))
        epoch_val_mae = torch.mean(torch.tensor(val_mae))

        train_avg_loss.append(epoch_train_loss)
        val_avg_loss.append(epoch_val_loss)

        # For court detection all keypoints are always present, so TN=0 structurally.
        # FP=FN always (a wrong localization increments both), so precision=recall=f1.
        # accuracy = TP / total_keypoints = TP / (TP+FP)
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0.0
        accuracy = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

        wandb.log({
            "epoch": i + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_mae": epoch_val_mae,
            "learning_rate": lr,
            "val/accuracy": accuracy,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/TP": TP,
            "val/FP": FP,
            "val/FN": FN,
        })

        print("Epoch "+str(i)+" : train_loss = "+str(epoch_train_loss)+" and val_loss = "+str(epoch_val_loss))

        if (i+1) % parameters["save_every"] == 0:
            os.makedirs('../../models/court_detection', exist_ok=True)
            timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")
            filename = f'tracknet_court_epoch{i+1}_{timestamp}.pth'
            model_to_save = accelerator.unwrap_model(network) if accelerator is not None else network
            torch.save(model_to_save.state_dict(), f'../../models/court_detection/{filename}')
            print(f"Saved checkpoint: {filename}")

    return train_avg_loss, val_avg_loss

if __name__ == "__main__":

    parameters = {
        "optimizer": "Adam",
        "model": "TrackNetCourt",
        "num_workers": 2,
        "batch_size": 4,
        "split": 0.7,
        "criterion": "weighted_MSE",
        "pos_weight": 1000,
        "learning_rate": 1e-5,
        "num_epochs": 100,
        "variance": 10,
        "scheduler": False,
        "weight_init": "he",
        "dropout": True,
        "save_every": 5,
        "shuffle": True,
        "loading": False,
        "accelerate": True,
        "normalization": "imagenet",
        "img_size": (320, 176),
    }

    if parameters["optimizer"] == "AdamW":
        parameters["weight_decay"] = 1e-4

    if parameters["scheduler"]:
        parameters["gamma_scheduler"] = 0.1
        parameters["step_size_scheduler"] = 15

    if parameters["dropout"]:
        parameters["dropout_p"] = 0.3

    if parameters["loading"]:
        parameters["loading_path"] = "../../models/court_detection/XXX.pth"

    timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

    run = wandb.init(
        entity="uliege-tennis-tracking",
        project="court-tracking",
        name=f"TrackNet_{timestamp}",
        config=parameters
    )

    if parameters["accelerate"]:
        accelerator = Accelerator(mixed_precision="fp16")
        device = accelerator.device
        print(f"Using device: {device} (accelerate fp16 enabled)")
    else:
        accelerator = None
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")

    network = TrackNetCourt(weight_init=parameters["weight_init"], dropout=parameters["dropout"],
                            dropout_p=parameters.get("dropout_p", 0.3))
    network.to(device)

    if parameters["loading"]:
        network.load_state_dict(torch.load(parameters["loading_path"], map_location=device))
        print(f"Loaded weights from: {parameters['loading_path']}")

    if parameters["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=parameters["learning_rate"])
    elif parameters["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(network.parameters(), lr=parameters["learning_rate"],
                                    weight_decay=parameters["weight_decay"])

    if parameters["scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters["step_size_scheduler"],
            gamma=parameters["gamma_scheduler"])

    trainSet = CourtDataset(type="train", split=parameters["split"], variance=parameters["variance"], normalization=parameters["normalization"], img_size=parameters["img_size"])
    valSet = CourtDataset(type="val", split=parameters["split"], variance=parameters["variance"], normalization=parameters["normalization"], img_size=parameters["img_size"])

    trainloader = DataLoader(trainSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"],
                            num_workers=parameters["num_workers"])
    valloader = DataLoader(valSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"],
                            num_workers=parameters["num_workers"])

    if accelerator is not None:
        network, optimizer, trainloader, valloader = accelerator.prepare(
            network, optimizer, trainloader, valloader
        )

    print(f"\nTrain size: {len(trainloader)}, Val size: {len(valloader)}")

    train_avg_loss, val_avg_loss = train(parameters["num_epochs"], accelerator=accelerator)
