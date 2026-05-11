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
    "nb_input_frames": 3,
    "batch_size": 4,
    "shuffle": False,
    "num_workers": 2,
    "loading_file": "tracknet_ball_epoch30_11052026_03h27m06s.pth",
    "gamma_loss": 2
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

# Initialize model with same architecture as training (dropout=True to match saved weights)
network = TrackNet(weight_init="he", nb_input_frames=parameters["nb_input_frames"], 
                   dropout=True, dropout_p=0.3)
network.load_state_dict(torch.load(f"../../models/ball_tracking/{parameters['loading_file']}", 
                                   map_location=device))
network.to(device)
network.eval()

testSet = BallDataset(type="test", train_coef=0.7, val_coef=0.15,
                      nb_input_frames=3, variance=10, frame="last")
testloader = DataLoader(testSet, batch_size=parameters["batch_size"], 
                        shuffle=parameters["shuffle"], 
                        num_workers=parameters["num_workers"])

print(f"\nTest size: {len(testloader)}")

TP, FP, TN, FN = 0, 0, 0, 0
test_losses = []

all_predictions = []
all_ground_truths = []
all_dataset_indices = []
sample_metrics = []

with torch.no_grad():
    batch_start_idx = 0
    for x, y, vis in testloader:
        x = x.to(device)
        y = y.to(device)
        vis = vis.to(device)
        
        pred = network(x)
        loss = criterionFocalLoss(pred, y, parameters["gamma_loss"])
        test_losses.append(loss.detach())
        
        pred_class = torch.argmax(pred, dim=1)  # (B, H, W)

        # Compute metrics per sample
        for b in range(x.shape[0]):
            idx = batch_start_idx + b
            pred_heatmap = pred_class[b].float()
            ball_detected = pred_heatmap.max() > 2.55
            sample_metrics.append({
                'dataset_idx': idx,
                'detected': ball_detected,
                'max_pred_value': pred_heatmap.max().item()
            })
        
        TP_i, FP_i, TN_i, FN_i = compute_ball_metrics(pred, y, vis)
        TP += TP_i
        FP += FP_i
        TN += TN_i
        FN += FN_i

        all_predictions.append(pred_class.cpu())
        all_ground_truths.append(y.cpu())
        
        batch_size = x.shape[0]
        all_dataset_indices.extend(range(batch_start_idx, batch_start_idx + batch_size))
        batch_start_idx += batch_size

        batch_idx = len(test_losses)
        if batch_idx % 100 == 0:
            print(f"Test batch {batch_idx}/{len(testloader)} ({100*batch_idx/len(testloader):.1f}%)")

test_loss = torch.mean(torch.tensor(test_losses))

accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0.0
precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
recall = TP/(TP+FN) if (TP+FN) > 0 else 0.0
specificity = TN/(TN+FP) if (TN+FP) > 0 else 0.0
f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

print(f"test_loss = {test_loss}")
print(f"accuracy = {accuracy} , precision = {precision} , recall = {recall} and f1 = {f1}")
print(f"TP = {TP} , FP = {FP} , FN = {FN} and TN = {TN}")

wandb.log({
    "test_loss": test_loss,
    "test/accuracy": accuracy,
    "test/precision": precision,
    "test/recall": recall,
    "test/specificity": specificity,
    "test/f1": f1,
    "test/TP": TP,
    "test/FP": FP,
    "test/FN": FN,
    "test/TN": TN,
})

predictions_dir = os.path.join(OUTPUTS_DIR, "ball_tracking", "predictions")
os.makedirs(predictions_dir, exist_ok=True)

predictions_file = os.path.join(predictions_dir, f"predictions_{timestamp}.pt")
torch.save({
    'predictions': torch.cat(all_predictions, dim=0),
    'ground_truths': torch.cat(all_ground_truths, dim=0),
    'dataset_indices': all_dataset_indices,
    'sample_metrics': sample_metrics,
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