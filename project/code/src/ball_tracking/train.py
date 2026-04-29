import wandb
import torch
from model import TrackNet
from torch.utils.data import DataLoader
from dataloader import BallDataset
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime

# Claude generated
def compute_ball_metrics(pred, y, threshold=7):
    """
    pred : (B, 256, H, W) - raw logits from TrackNet (256 classes)
    y    : (B, 1, H, W)   - normalized heatmap [0,1]
    """
    B, _, H, W = pred.shape
    tp, fp, fn, tn = 0, 0, 0, 0

    # Convert pred logits to class indices (0-255), then normalize back
    pred_class = torch.argmax(pred, dim=1)  # (B, H, W)

    for b in range(B):
        true_heatmap = y[b, 0]          # (H, W)
        pred_heatmap = pred_class[b].float() / 255.0  # (H, W)

        ball_visible = true_heatmap.max() > 0.01

        if not ball_visible:
            # Ball is invisible — good prediction = near-zero heatmap
            if pred_heatmap.max() < 0.01:
                tn += 1   # correctly predicted no ball
            else:
                fp += 1   # predicted a ball that doesn't exist
        else:
            # Ball is visible — check if predicted position is close enough
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
                fn += 1

    return tp, fp, fn, tn

def criterionCrossEntropy(pred, y):
    # Page 8 - TrackNet paper
    y = (y * 255).squeeze(1).long()   # (B, H, W)

    loss = nn.CrossEntropyLoss()(pred, y)

    return loss

# Default value of gamma=2 based on the focal loss paper - page 5
def criterionFocalLoss(pred, y, gamma=2):
    y = (y * 255).squeeze(1).long()
    
    ce = F.cross_entropy(pred, y, reduction="none")  # (B, H, W)
    
    loss = (1 - torch.exp(-ce)) ** gamma * ce
    
    return loss.mean()

parameters = {
    "optimizer" : "Adam",
    "model" : "TrackNet",
    "num_workers" : 0,
    "batch_size" : 2, # was at 4, got out of memory warning
    "train_coef" : 0.7, 
    "val_coef" : 0.15,
    "criterion" : "Focal Loss",
    "learning_rate" : 0.001,
    "num_eprochs" : 10, # for testing purposes, will be set to 10 later on
    "nb_input_frame" : 3,
    "variance" : 10,
    "scheduler" : False,
    "weight_init" : "uniform", # uniform on the paper but probably updated
    "dropout" : False,
    "save_every": 2 # every x epochs checkpoint for saving weights
}

if parameters["criterion"] == "Focal loss":
    parameters["gamma_loss"] = 2

if parameters["optimizer"] == "AdamW":
    parameters["weight_decay"] = 1e-4

if parameters["scheduler"]:
    parameters["gamma_scheduler"] = 0.1
    parameters["step_size_scheduler"] = 5

if parameters["dropout"]:
    parameters["dropout_p"] = 0.2

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="ball-tracking",
    name="TrackNet_test",
    config=parameters
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

# hardcoding dropout to false and dropout_p = 0.2 for now, might change that later
network = TrackNet(weight_init=parameters["weight_init"], nb_input_frames=parameters["nb_input_frame"],
                   dropout=False, dropout_p=0.2) 
network.to(device)

if parameters["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(network.parameters(), lr=parameters["learning_rate"])
elif parameters["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(network.parameters(), lr=parameters["learning_rate"], 
                                  weight_decay=parameters["weight_decay"])

# Improvement is to trigger the scheduler only when the validation loss stops improving
if parameters["scheduler"] == True:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=parameters["step_size_scheduler"],
        gamma=parameters["gamma_scheduler"])
    
    
trainSet = BallDataset(type="train", train_coef=parameters["train_coef"], val_coef=parameters["val_coef"], 
                       nb_input_frames=parameters["nb_input_frame"],  variance=parameters["variance"])
valSet = BallDataset(type="val", train_coef=parameters["train_coef"], val_coef=parameters["val_coef"], 
                     nb_input_frames=parameters["nb_input_frame"], variance=parameters["variance"])

trainloader = DataLoader(trainSet, batch_size=parameters["batch_size"], shuffle=False, 
                         num_workers=parameters["num_workers"])
valloader = DataLoader(valSet, batch_size=parameters["batch_size"], shuffle=False, 
                        num_workers=parameters["num_workers"])

print(f"\nTrain size: {len(trainloader)}, Test size: {len(valloader)}")

# From homework 2

def train(num_epochs):
    train_avg_loss = []
    val_avg_loss = []

    for i in range(num_epochs):
        train_losses = []
        val_losses = []
        network.train()
        TP, TN, FP, FN = 0, 0, 0, 0
        
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            
            pred = network(x)
            
            if parameters["criterion"] == "Cross-entropy loss":
                loss = criterionCrossEntropy(pred, y)
            elif parameters["criterion"] == "Focal loss":
                loss = criterionFocalLoss(pred, y, parameters["gamma_loss"])
            
            train_losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print progress every 100 batches
            batch_idx = len(train_losses)
            if batch_idx % 100 == 0:
                print(f"Epoch {i} - Train batch {batch_idx}/{len(trainloader)} ({100*batch_idx/len(trainloader):.1f}%)")

        network.eval()
        with torch.no_grad():

            for x, y in valloader:
                x = x.to(device)
                y = y.to(device)

                pred = network(x)
                
                if parameters["criterion"] == "Cross-entropy loss":
                    loss = criterionCrossEntropy(pred, y)
                elif parameters["criterion"] == "Focal loss":
                    loss = criterionFocalLoss(pred, y, parameters["gamma_loss"])

                TP_i, TN_i, FP_i, FN_i = compute_ball_metrics(pred, y)
                TP += TP_i
                TN += TN_i
                FP += FP_i
                FN += FN_i
                
                val_losses.append(loss)
                batch_idx = len(val_losses)
                if batch_idx % 50 == 0:
                    print(f"Epoch {i} - Val batch {batch_idx}/{len(valloader)} ({100*batch_idx/len(valloader):.1f}%)")

        if parameters["scheduler"] == True:
            scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"] # Because can change with the scheduler
        
        epoch_train_loss = torch.mean(torch.tensor(train_losses))
        epoch_val_loss = torch.mean(torch.tensor(val_losses))

        train_avg_loss.append(epoch_train_loss)
        val_avg_loss.append(epoch_val_loss)
        
        # number of good predictions over the total number
        accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0.0
        # proportion of good predictions among all the positive predictions
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        # proportion of positives that are detected
        recall = TP/(TP+FN) if (TP+FN) >0 else 0.0
        
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        
        wandb.log({
            "epoch" : i + 1,
            "train_loss" : epoch_train_loss,
            "val_loss" : epoch_val_loss,
            "learning_rate" : lr,
            "val/precision" : precision,
            "val/recall" : recall,
            "val/f1" : f1,
            "val/TP" : TP,
            "val/FP" : FP,
            "val/FN" : FN,
            "val/TN" : TN,
        })
        
        print("Epoch "+str(i)+" : train_loss = "+str(epoch_train_loss)+" and val_loss = "+str(epoch_val_loss))

        if i+1 % parameters["save_every"] == 0:
            os.makedirs('../../models/ball_tracking', exist_ok=True)
            timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")
            filename = f'tracknet_ball_epoch{i+1}_{timestamp}.pth'
            torch.save(network.state_dict(), f'../../models/ball_tracking/{filename}')
            print(f"Saved checkpoint: {filename}")
    
    return train_avg_loss, val_avg_loss

train_avg_loss, val_avg_loss = train(parameters["num_eprochs"])