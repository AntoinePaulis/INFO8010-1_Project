import wandb
import torch
from model import TrackNet

parameters = {
    "model" : "TrackNet",
    "learning_rate" : 0.01,
    "num_eprochs" : 10,
    "nb_input_frame" : 3
}

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

network = TrackNet(parameters["nb_input_frame"])

optimizer = torch.optim.Adam(network.parameters(), lr=parameters["learning_rate"])

trainset = ""
testset = ""

print(f"\nTrain size: {len(trainset)}, Test size: {len(testset)}")

