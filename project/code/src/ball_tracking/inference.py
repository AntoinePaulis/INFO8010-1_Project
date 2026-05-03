import torch
import wandb
from model import TrackNet
from dataloader import BallDataset
from torch.utils.data import DataLoader
from train import compute_ball_metrics

parameters = {
    "weight_init" : "uniform",
    "nb_input_frames" : 3,
    "dropout" : False,
    "dropout_p" : 0.2 ,
    "shuffle" : False,
    "num_workers" : 0,
    "batch_size" : 2,
}

run = wandb.init(
    entity="uliege-tennis-tracking",
    project="ball-tracking",
    name="TrackNet_inference",
    config=parameters
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

network = TrackNet(weight_init=parameters["weight_init"], nb_input_frames=parameters["nb_input_frames"], 
                   dropout=parameters["dropout"], dropout_p=parameters["dropout_p"])
network.to(device)

loading_path = "../../models/ball_tracking/tracknet_ball_epoch30_30042026_03h28m14s.pth"
network.load_state_dict(torch.load(loading_path, map_location=device))

network.eval()

testSet = BallDataset(type="test", train_coef=0.7, val_coef=0.15,
                      nb_input_frames=3, variance=10, frame="last")
testloader = DataLoader(testSet, batch_size=parameters["batch_size"], shuffle=parameters["shuffle"], 
                        num_workers=parameters["num_workers"])

TP, TN, FP, FN = 0, 0, 0, 0

with torch.no_grad():
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        pred = network(x)
        TP_i, FP_i, FN_i, TN_i = compute_ball_metrics(pred, y)
        TP += TP_i
        FP += FP_i
        FN += FN_i
        TN += TN_i

# number of good predictions over the total number
accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0.0
# proportion of good predictions among all the positive predictions
precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
# proportion of positives that are detected
recall = TP/(TP+FN) if (TP+FN) >0 else 0.0

f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

print(f"accuracy = {accuracy} , precision = {precision} , recall = {recall} and f1 = {f1}")
print(f"TP = {TP} , FP = {FP} , FN = {FN} and TN = {TN}")

wandb.log({
    "test/accuracy"  : accuracy,
    "test/precision" : precision,
    "test/recall"    : recall,
    "test/f1"        : f1,
    "test/TP"        : TP,
    "test/FP"        : FP,
    "test/FN"        : FN,
    "test/TN"        : TN,
})

wandb.finish()