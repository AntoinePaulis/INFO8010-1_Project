import wandb
import torch
from model import TrackNetCourt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import CourtDataset

parameters = {
    "optimizer" : "Adam",
    "model" : "TrackNet",
    "num_workers" : 0, # 0 for the simple local test
    "batch_size" : 4,
    "split" : 0.7,
    "criterion" : "MSE",
    "learning_rate" : 0.01,
    "num_eprochs" : 1, # 1 for the simple local test
    "variance" : 10,
    "scheduler" : False,
    "weight_init" : "uniform", # uniform on the paper but probably updated
    "dropout" : False
}

if parameters["optimizer"] == "AdamW":
    parameters["weight_decay"] = 1e-4

if parameters["scheduler"]:
    parameters["gamma_scheduler"] = 0.1
    parameters["step_size_scheduler"] = 5

if parameters["dropout"]:
    parameters["dropout_p"] = 0.2

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="court-tracking",
    name="TrackNet_test",
    config=parameters
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

network = TrackNetCourt(weight_init=parameters["weight_init"], dropout=parameters["dropout"],
                        dropout_p=parameters["dropout_p"])
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

criterion = nn.MSELoss()

trainSet = CourtDataset(type="train", split=parameters["split"], variance=parameters["variance"])
valSet = CourtDataset(type="val", split=parameters["split"], variance=parameters["variance"])

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
        val_mae = []
        
        network.train()

        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            
            pred = network(x)
            loss = criterion(pred, y)
            train_losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():

            for x, y in valloader:
                x = x.to(device)
                y = y.to(device)

                pred = network(x)
                
                loss = criterion(pred, y)
                val_losses.append(loss)
                
                mae = F.l1_loss(pred, y)
                val_mae.append(mae)
        
        if parameters["scheduler"] == True:
            scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"] # Because can change with the scheduler

        epoch_train_loss = torch.mean(torch.tensor(train_losses))
        epoch_val_loss = torch.mean(torch.tensor(val_losses))
        epoch_val_mae = torch.mean(torch.tensor(val_mae))
        
        train_avg_loss.append(epoch_train_loss)
        val_avg_loss.append(epoch_val_loss)
        
        wandb.log({
            "epoch": i + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_mae": epoch_val_mae,
            "learning_rate": lr
        })
        
        print("Epoch "+str(i)+" : train_loss = "+str(epoch_train_loss)+" and val_loss = "+str(epoch_val_loss))
        
    return train_avg_loss, val_avg_loss

train_avg_loss, val_avg_loss = train(parameters["num_eprochs"])