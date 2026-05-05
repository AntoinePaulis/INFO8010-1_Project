import torch
import wandb
from model import TrackNet
from dataloader import BallDataset
from torch.utils.data import DataLoader
from train import compute_ball_metrics, criterionFocalLoss
from datetime import datetime
import sys
sys.path.append('..')
from config import OUTPUTS_DIR
import os

parameters = {
    "weight_init" : "uniform",
    "nb_input_frames" : 3,
    "dropout" : False,
    "dropout_p" : 0.2 ,
    "shuffle" : False,
    "num_workers" : 0,
    "batch_size" : 2,
    "loading_file" : "tracknet_ball_epoch30_30042026_03h28m14s.pth",
    "gamma_loss" : 2
}

timestamp = datetime.now().strftime("%d%m%Y_%Hh%Mm%Ss")

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="ball-tracking",
    name=f"TrackNet_inference_{timestamp}",
    config=parameters
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

network = TrackNet(weight_init=parameters["weight_init"], nb_input_frames=parameters["nb_input_frames"], 
                   dropout=parameters["dropout"], dropout_p=parameters["dropout_p"])
network.to(device)

loading_path = f"../../models/ball_tracking/{parameters['loading_file']}"
network.load_state_dict(torch.load(loading_path, map_location=device))

network.eval()

testSet = BallDataset(type="test", train_coef=0.7, val_coef=0.15,
                      nb_input_frames=3, variance=10, frame="last")
testloader = DataLoader(testSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"], 
                        num_workers=parameters["num_workers"])

print(f"\nTest size: {len(testloader)}")

TP, TN, FP, FN = 0, 0, 0, 0

test_losses = []

# Storage for predictions
all_predictions = []
all_ground_truths = []
all_dataset_indices = []

with torch.no_grad():
    batch_start_idx = 0
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        pred = network(x)
        
        loss = criterionFocalLoss(pred, y, parameters["gamma_loss"])
        
        test_losses.append(loss.detach())
        
        TP_i, FP_i, FN_i, TN_i, multiple_balls = compute_ball_metrics(pred, y)
        TP += TP_i
        FP += FP_i
        FN += FN_i
        TN += TN_i
        
        # Save predictions (claude)
        pred_class = torch.argmax(pred, dim=1)  # (B, H, W)
        all_predictions.append(pred_class.cpu())
        all_ground_truths.append(y.cpu())
        
        # Track which dataset indices these correspond to (claude)
        batch_size = x.shape[0]
        all_dataset_indices.extend(range(batch_start_idx, batch_start_idx + batch_size))
        batch_start_idx += batch_size

        # Print progress every 100 batches 
        batch_idx = len(test_losses)
        if batch_idx % 100 == 0:
            print(f"Test batch {batch_idx}/{len(testloader)} ({100*batch_idx/len(testloader):.1f}%)")

test_loss = torch.mean(torch.tensor(test_losses))

# number of good predictions over the total number
accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0.0
# proportion of good predictions among all the positive predictions
precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
# proportion of positives that are detected
recall = TP/(TP+FN) if (TP+FN) >0 else 0.0

f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

print(f"test_loss = {test_loss}")
print(f"accuracy = {accuracy} , precision = {precision} , recall = {recall} and f1 = {f1}")
print(f"TP = {TP} , FP = {FP} , FN = {FN} and TN = {TN}")



wandb.log({
    "test_loss" : test_loss,
    "test/accuracy" : accuracy,
    "test/precision" : precision,
    "test/recall" : recall,
    "test/f1" : f1,
    "test/TP" : TP,
    "test/FP" : FP,
    "test/FN" : FN,
    "test/TN" : TN,
})

# Save predictions to disk
predictions_dir = os.path.join(OUTPUTS_DIR, "ball_tracking", "predictions")
os.makedirs(predictions_dir, exist_ok=True)

predictions_file = os.path.join(predictions_dir, f"predictions_{timestamp}.pt")
torch.save({
    'predictions': torch.cat(all_predictions, dim=0),  # (N, H, W)
    'ground_truths': torch.cat(all_ground_truths, dim=0),  # (N, 1, H, W)
    'dataset_indices': all_dataset_indices,
    'model_file': parameters['loading_file'],
    'timestamp': timestamp,
    'metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN
    }
}, predictions_file)

print(f"\nSaved predictions to: {predictions_file}")

wandb.finish()