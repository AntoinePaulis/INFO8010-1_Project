import wandb
import torch
from model import TrackNet
from torch.utils.data import DataLoader
from dataloader import BallDataset
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
import cv2
import numpy as np
from accelerate import Accelerator
import argparse

def compute_ball_metrics(pred, y, visibility_batch, threshold=5):
    """
    pred : (B, 256, H, W) - raw logits from TrackNet (256 classes)
    y    : (B, 1, H, W)   - normalized heatmap [0,1]
    tp   : the ball centroid was correctly predicted withing [threshold] pixels of error
    fp   : the ball centroid was predicted with > [threshold] pixels of error
    fn   : no ball/more than one ball was predicted where there was exactly one ball
    tn   : no ball predicted, where there was no ball 
    """
    B, _, H, W = pred.shape
    tp, fp, tn, fn = 0, 0, 0, 0

    pred_class = torch.argmax(pred, dim=1)  # (B, H, W)

    for b in range(B):
        visibility = visibility_batch[b].item()
        true_heatmap = y[b, 0]
        pred_heatmap = pred_class[b].float() 
        ball_detected = pred_heatmap.max() > 2.55

        if visibility == 0 or visibility == 3:
            ball_visible = False
        else:
            ball_visible = True

        if not ball_visible:
            if not ball_detected:
                tn += 1
            else:
                fp += 1
        else: # ball is visible
            if not ball_detected:
                fn += 1
            else:
                true_idx = torch.argmax(true_heatmap)
                pred_idx = torch.argmax(pred_heatmap)

                true_y_coord, true_x_coord = divmod(true_idx.item(), W)
                pred_y_coord, pred_x_coord = divmod(pred_idx.item(), W)
                
                dist = ((pred_x_coord - true_x_coord)**2 + 
                        (pred_y_coord - true_y_coord)**2) ** 0.5
                
                if dist < threshold:
                    tp += 1
                else:
                    fp += 1

    return tp, fp, tn, fn

def criterionCrossEntropy(pred, y):
    y = y.squeeze(1).long()
    loss = nn.CrossEntropyLoss()(pred, y)
    return loss

def criterionFocalLoss(pred, y, gamma=2):
    y = y.squeeze(1).long()
    ce = F.cross_entropy(pred, y, reduction="none")
    loss = (1 - torch.exp(-ce)) ** gamma * ce
    return loss.mean()

def train(num_epochs, accelerator=None):
    train_avg_loss = []
    val_avg_loss = []

    for i in range(num_epochs):
        train_losses = []
        val_losses = []
        network.train()
        TP, TN, FP, FN = 0, 0, 0, 0
        
        # DEBUG: Check one sample
        sample_x, sample_y, sample_vis = next(iter(trainloader))
        print(f"DEBUG - Input range: [{sample_x.min():.3f}, {sample_x.max():.3f}]")
        print(f"DEBUG - Heatmap range: [{sample_y.min():.3f}, {sample_y.max():.3f}]")
        print(f"DEBUG - Heatmap dtype: {sample_y.dtype}")
        print(f"DEBUG - Ball visible samples: {(sample_y.max(dim=-1)[0].max(dim=-1)[0] > 2.55).sum()}/{sample_y.shape[0]}")
        for x, y, vis in trainloader:
            # accelerator.prepare() already moved data to device,
            # manual .to(device) is only needed without accelerate
            if accelerator is None:
                x = x.to(device)
                y = y.to(device)
                vis = vis.to(device)
            
            pred = network(x)
            
            if parameters["criterion"] == "Cross-entropy loss":
                loss = criterionCrossEntropy(pred, y)
            elif parameters["criterion"] == "Focal loss":
                loss = criterionFocalLoss(pred, y, parameters["gamma_loss"])
            
            train_losses.append(loss.detach())

            optimizer.zero_grad()

            # KEY difference: accelerator handles fp16 gradient scaling
            if accelerator is not None:
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(network.parameters(), max_norm=1.0) # claude suggestion
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0) # claude suggestion

            optimizer.step()
            
            batch_idx = len(train_losses)
            if batch_idx % 100 == 0:
                print(f"Epoch {i} - Train batch {batch_idx}/{len(trainloader)} ({100*batch_idx/len(trainloader):.1f}%)")

        network.eval()
        
        with torch.no_grad():
            for x, y, vis in valloader:
                if accelerator is None:
                    x = x.to(device)
                    y = y.to(device)
                    vis = vis.to(device)

                pred = network(x)
                
                if parameters["criterion"] == "Cross-entropy loss":
                    loss = criterionCrossEntropy(pred, y)
                elif parameters["criterion"] == "Focal loss":
                    loss = criterionFocalLoss(pred, y, parameters["gamma_loss"])

                TP_i, FP_i, TN_i, FN_i = compute_ball_metrics(pred, y, vis)
                TP += TP_i
                FP += FP_i
                TN += TN_i
                FN += FN_i
                
                val_losses.append(loss)
                batch_idx = len(val_losses)
                if batch_idx % 50 == 0:
                    print(f"Epoch {i} - Val batch {batch_idx}/{len(valloader)} ({100*batch_idx/len(valloader):.1f}%)")

        if parameters["scheduler"] == True:
            scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]
        
        epoch_train_loss = torch.mean(torch.tensor(train_losses))
        epoch_val_loss = torch.mean(torch.tensor(val_losses))

        train_avg_loss.append(epoch_train_loss)
        val_avg_loss.append(epoch_val_loss)
        
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0.0
        accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        
        wandb.log({
            "epoch" : i + 1,
            "train_loss" : epoch_train_loss,
            "val_loss" : epoch_val_loss,
            "learning_rate" : lr,
            "val/accuracy" : accuracy,
            "val/precision" : precision,
            "val/recall" : recall,
            "val/f1" : f1,
            "val/TP" : TP,
            "val/FP" : FP,
            "val/TN" : TN,
            "val/FN" : FN,
        })
        
        print("Epoch "+str(i)+" : train_loss = "+str(epoch_train_loss)+" and val_loss = "+str(epoch_val_loss))

        if (i+1) % parameters["save_every"] == 0:
            os.makedirs('../../models/ball_tracking', exist_ok=True)
            timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")
            filename = f'tracknet_ball_epoch{i+1}_{timestamp}.pth'
            # unwrap_model needed with accelerate to get the raw nn.Module
            model_to_save = accelerator.unwrap_model(network) if accelerator is not None else network
            torch.save(model_to_save.state_dict(), f'../../models/ball_tracking/{filename}')
            print(f"Saved checkpoint: {filename}")
    
    return train_avg_loss, val_avg_loss

if __name__ == "__main__":
    parameters = {
        "optimizer" : "Adam",
        "model" : "TrackNet",
        "num_workers" : 2,
        "batch_size" : 4,
        "frame" : "last",
        "train_coef" : 0.7, 
        "val_coef" : 0.15,
        "criterion" : "Focal loss",
        "learning_rate" : 0.0005,
        "num_epochs" : 10, 
        "nb_input_frame" : 3,
        "variance" : 10, # chosen after running test_heatmap 
        "scheduler" : False,
        "weight_init" : "he",
        "dropout" : True,
        "save_every": 5,
        "shuffle" : True, #IMPORTANT EDIT
        "loading" : False,
        "accelerate" : True,   # ← toggle here
        "normalization" : "imagenet" # matching the Tracknet paper
    }

    if parameters["criterion"] == "Focal loss":
        parameters["gamma_loss"] = 2

    if parameters["optimizer"] == "AdamW":
        parameters["weight_decay"] = 1e-4

    if parameters["scheduler"]:
        parameters["gamma_scheduler"] = 0.1
        parameters["step_size_scheduler"] = 15  # Changed from 5

    if parameters["dropout"]:
        parameters["dropout_p"] = 0.3

    if parameters["loading"]:
        parameters["loading_path"] = "../../models/ball_tracking/tracknet_ball_epoch30_30042026_03h28m14s.pth"
    
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")
    
    run = wandb.init(
        entity="uliege-tennis-tracking",
        project="ball-tracking",
        name=f"TrackNet_{timestamp}",
        config=parameters
    )

    # ── Accelerate setup ───────────────────────────────────────────────────
    if parameters["accelerate"]:
        accelerator = Accelerator(mixed_precision="fp16")
        device = accelerator.device
        print(f"Using device: {device} (accelerate fp16 enabled)")
    else:
        accelerator = None
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")

    network = TrackNet(weight_init=parameters["weight_init"], nb_input_frames=parameters["nb_input_frame"])
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
        
    trainSet = BallDataset(type="train", train_coef=parameters["train_coef"], val_coef=parameters["val_coef"], 
                           nb_input_frames=parameters["nb_input_frame"], variance=parameters["variance"], frame=parameters["frame"], normalization=parameters["normalization"])
    valSet = BallDataset(type="val", train_coef=parameters["train_coef"], val_coef=parameters["val_coef"], 
                         nb_input_frames=parameters["nb_input_frame"], variance=parameters["variance"], frame=parameters["frame"], normalization=parameters["normalization"])

    trainloader = DataLoader(trainSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"], 
                             num_workers=parameters["num_workers"])
    valloader = DataLoader(valSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"], 
                           num_workers=parameters["num_workers"])

    # accelerator.prepare() wraps model, optimizer and dataloaders
    # it handles device placement and fp16 casting automatically
    if accelerator is not None:
        network, optimizer, trainloader, valloader = accelerator.prepare(
            network, optimizer, trainloader, valloader
        )

    print(f"\nTrain size: {len(trainloader)}, Test size: {len(valloader)}")

    train_avg_loss, val_avg_loss = train(parameters["num_epochs"], accelerator=accelerator)