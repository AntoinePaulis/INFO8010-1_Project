import torch
import wandb
from model import TrackNetCourt
from dataloader import CourtDataset
from torch.utils.data import DataLoader
from train import compute_court_metrics
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from datetime import datetime
import sys
sys.path.append('..')
from config import OUTPUTS_DIR

parameters = {
    "weight_init":  "he",
    "dropout":      True,
    "dropout_p":    0.3,
    "normalization": "imagenet",
    "img_size":     (320, 176),
    "variance":     10,
    "shuffle":      False,
    "num_workers":  2,
    "batch_size":   4,
    "loading_file": "tracknet_court_epoch100_13052026_14h35m50s.pth",
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

testSet = CourtDataset(type="test", img_size=parameters["img_size"],
                       normalization=parameters["normalization"], variance=parameters["variance"])
testloader = DataLoader(testSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"],
                        num_workers=parameters["num_workers"])

print(f"\nTest size: {len(testloader)}")

criterion = nn.MSELoss()

TP, TN, FP, FN = 0, 0, 0, 0
test_losses = []
test_maes = []

all_pred_keypoints = []  # list of (nb_kps, 2) tensors — (x, y) in image pixel space
all_gt_keypoints = []
all_dataset_indices = []
sample_metrics = []

with torch.no_grad():
    batch_start_idx = 0
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        pred = network(x)

        B, nb_kps, H, W = pred.shape

        test_losses.append(criterion(pred, y).item())
        test_maes.append(F.l1_loss(pred, y).item())

        TP_i, TN_i, FP_i, FN_i = compute_court_metrics(pred, y)
        TP += TP_i
        TN += TN_i
        FP += FP_i
        FN += FN_i

        for b in range(B):
            pred_kps = []
            gt_kps = []
            sample_tp, sample_fp = 0, 0

            for k in range(nb_kps):
                pred_idx = torch.argmax(pred[b, k]).item()
                gt_idx = torch.argmax(y[b, k]).item()

                pred_row, pred_col = divmod(pred_idx, W)
                gt_row, gt_col = divmod(gt_idx, W)

                pred_kps.append([pred_col, pred_row])  # (x, y)
                gt_kps.append([gt_col, gt_row])

                dist = math.dist([pred_col, pred_row], [gt_col, gt_row])
                if dist < 7:
                    sample_tp += 1
                else:
                    sample_fp += 1

            all_pred_keypoints.append(torch.tensor(pred_kps, dtype=torch.float32))
            all_gt_keypoints.append(torch.tensor(gt_kps, dtype=torch.float32))
            sample_metrics.append({
                'dataset_idx': batch_start_idx + b,
                'TP': sample_tp,
                'FP': sample_fp,
            })

        batch_size = x.shape[0]
        all_dataset_indices.extend(range(batch_start_idx, batch_start_idx + batch_size))
        batch_start_idx += batch_size

        batch_idx = len(test_losses)
        if batch_idx % 100 == 0:
            print(f"Test batch {batch_idx}/{len(testloader)} ({100*batch_idx/len(testloader):.1f}%)")

test_loss = torch.mean(torch.tensor(test_losses))
test_mae = torch.mean(torch.tensor(test_maes))

accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0.0
precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
recall = TP/(TP+FN) if (TP+FN) > 0 else 0.0
f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

print(f"test_loss = {test_loss} and test_mae = {test_mae}")
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

predictions_dir = os.path.join(OUTPUTS_DIR, "court_detection", "predictions")
os.makedirs(predictions_dir, exist_ok=True)

predictions_file = os.path.join(predictions_dir, f"predictions_{timestamp}.pt")
torch.save({
    'pred_keypoints': torch.stack(all_pred_keypoints),  # (N, nb_kps, 2) — (x, y) in image pixel space
    'gt_keypoints': torch.stack(all_gt_keypoints),      # (N, nb_kps, 2)
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
