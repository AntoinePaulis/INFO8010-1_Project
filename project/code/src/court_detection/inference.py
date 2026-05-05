import torch
import wandb
from model import TrackNetCourt
from dataloader import CourtDataset
from torch.utils.data import DataLoader
from train import compute_court_metrics
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime

parameters = {
    "weight_init" : "uniform",
    "dropout" : False,
    "dropout_p" : 0.2 ,
    "shuffle" : False,
    "num_workers" : 0,
    "batch_size" : 1,
    "loading_file" : "tracknet_court_epoch2_03052026_09h09m35s.pth"
}

timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="court-tracking",
    name=f"TrackNet_inference_{timestamp}",
    config=parameters
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

network = TrackNetCourt(weight_init=parameters["weight_init"], dropout=parameters["dropout"],
                        dropout_p=parameters["dropout_p"])
    
network.to(device)

loading_path = f"../../models/court_detection/{parameters['loading_file']}"
network.load_state_dict(torch.load(loading_path, map_location=device))

network.eval()

testSet = CourtDataset(type="test")
testloader = DataLoader(testSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"], 
                        num_workers=parameters["num_workers"])

print(f"\nTest size: {len(testloader)}")

criterion = nn.MSELoss()

TP, TN, FP, FN = 0, 0, 0, 0
test_losses = []
test_maes = []

with torch.no_grad():
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        pred = network(x)
        
        test_losses.append(criterion(pred, y).item())
        test_maes.append(F.l1_loss(pred, y).item())
        
        TP_i, FP_i, FN_i, TN_i = compute_court_metrics(pred, y)
        TP += TP_i
        FP += FP_i
        FN += FN_i
        TN += TN_i
        
        # Print progress every 100 batches
        batch_idx = len(test_losses)
        if batch_idx % 100 == 0:
            print(f"Test batch {batch_idx}/{len(testloader)} ({100*batch_idx/len(testloader):.1f}%)")

test_loss = torch.mean(torch.tensor(test_losses))
test_mae = torch.mean(torch.tensor(test_maes))

# number of good predictions over the total number
accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0.0
# proportion of good predictions among all the positive predictions
precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
# proportion of positives that are detected
recall = TP/(TP+FN) if (TP+FN) >0 else 0.0

f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

print(f"test_loss = {test_loss} ancd test_mae = {test_mae}")
print(f"accuracy = {accuracy} , precision = {precision} , recall = {recall} and f1 = {f1}")
print(f"TP = {TP} , FP = {FP} , FN = {FN} and TN = {TN}")

wandb.log({
    "test_loss" : test_loss,
    "test_mae" : test_mae, 
    "test/accuracy" : accuracy,
    "test/precision" : precision,
    "test/recall" : recall,
    "test/f1" : f1,
    "test/TP" : TP,
    "test/FP" : FP,
    "test/FN" : FN,
    "test/TN" : TN,
})

wandb.finish()