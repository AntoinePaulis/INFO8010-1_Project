import wandb
import torch
from model import TrackNet
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import BallDataset

def criterionCrossEntropy(pred, y):
    # Page 8 - TrackeNet paper
    y = (y * 255).squeeze(1).long()   # (B, H, W)

    loss = nn.CrossEntropyLoss()(pred, y)

    return loss

# Default value of gamme=2 based on the focal loss paper - page 5
def criterionFocalLoss(pred, y, gamma=2):
    y = (y * 255).squeeze(1).long()
    
    ce = F.cross_entropy(pred, y, reduction="none")  # (B, H, W)
    
    loss = (1 - torch.exp(-ce)) ** gamma * ce
    
    return loss.mean()

criterionList = [
    "Cross-entropy loss",
    "Focal loss"
]

optimizerList = [
    "Adam",
    "AdamW"
]

parameters = {
    "optimizer" : optimizerList[0],
    "model" : "TrackNet",
    "criterion" : criterionList[0],
    "learning_rate" : 0.01,
    "num_eprochs" : 10,
    "nb_input_frame" : 3,
    "variance" : 10,
    "split" : 0.7,
    "scheduler" : False,
    "weight_init" : False,
    "drop out" : False
}

if parameters["criterion"] == "Focal loss":
    parameters["gamma_loss"] = 2

if parameters["optimizer"] == "AdamW":
    parameters["weight_decay"] = 1e-4

if parameters["scheduler"]:
    parameters["gamma_scheduler"] = 0.1
    parameters["step_size_scheduler"] = 5

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="ball-tracking",
    name="TrackNet_test",
    config=parameters
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

network = TrackNet(parameters["nb_input_frame"])
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
    
    
trainSet = BallDataset(train=True, nb_input_frames=parameters["nb_input_frame"], 
                       variance=parameters["variance"], split=parameters["split"])
testSet = BallDataset(train=False, nb_input_frames=parameters["nb_input_frame"], 
                      variance=parameters["variance"], split=parameters["split"])

trainloader = DataLoader(trainSet, batch_size=4, shuffle=False, num_workers=2)
testloader = DataLoader(testSet, batch_size=4, shuffle=False, num_workers=2)

print(f"\nTrain size: {len(trainloader)}, Test size: {len(testloader)}")

# From homework 2

def train(num_epochs):
    train_avg_loss = []
    test_avg_loss = []

    for i in range(num_epochs):
        train_losses = []
        test_losses = []
        network.train()

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

        network.eval()
        with torch.no_grad():

            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)

                pred = network(x)
                
                if parameters["criterion"] == "Cross-entropy loss":
                    loss = criterionCrossEntropy(pred, y)
                elif parameters["criterion"] == "Focal loss":
                    loss = criterionFocalLoss(pred, y, parameters["gamma_loss"])
            
                test_losses.append(loss)

        if parameters["scheduler"] == True:
            scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"] # Because can change with the scheduler
        
        epoch_train_loss = torch.mean(torch.tensor(train_losses))
        epoch_test_loss = torch.mean(torch.tensor(test_losses))

        train_avg_loss.append(epoch_train_loss)
        test_avg_loss.append(epoch_test_loss)
        
        wandb.log({
            "epoch": i + 1,
            "train_loss": epoch_train_loss,
            "test_loss": epoch_test_loss,
            "learning_rate": lr
        })
        
        print("Epoch "+str(i)+" : train_loss = "+str(epoch_train_loss)+" and test_loss = "+str(epoch_test_loss))
        
    return train_avg_loss, test_avg_loss

train_avg_loss, test_avg_loss = train(parameters["num_eprochs"])