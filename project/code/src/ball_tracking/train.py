import wandb
import torch
from model import TrackNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import BallDataset

parameters = {
    "model" : "TrackNet",
    "learning_rate" : 0.01,
    "num_eprochs" : 10,
    "nb_input_frame" : 3
}

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

network = TrackNet(parameters["nb_input_frame"])
network.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=parameters["learning_rate"])

criterion = ""

trainSet = BallDataset(train=True)
testSet = BallDataset(train=False)

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

        for img, heatmap, game, clip in trainloader:
            """
            x = x.to(device)
            y = y.to(device)
            """
            pred = network(x)
            loss = criterion(pred, y)
            train_losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            correct = 0

            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)

                pred = network(x)
                loss = criterion(pred, y)
                test_losses.append(loss)

                y_pred = pred.argmax(dim=-1)
                correct = correct + (y_pred == y).sum()

        train_avg_loss.append(torch.mean(torch.tensor(train_losses)))
        test_avg_loss.append(torch.mean(torch.tensor(test_losses)))
        print("Epoch "+str(i))
    return train_avg_loss, test_avg_loss