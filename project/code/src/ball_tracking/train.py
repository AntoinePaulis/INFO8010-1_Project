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

optimizer = torch.optim.Adam(network.parameters(), lr=parameters["learning_rate"])

# Initialize the criterion
criterion = ""

trainSet = BallDataset(train=True, nb_input_frames=parameters["nb_input_frame"])
testSet = BallDataset(train=False, nb_input_frames=parameters["nb_input_frame"])

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
            loss = criterion(pred, y)
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
                loss = criterion(pred, y)
                test_losses.append(loss)

        epoch_train_loss = torch.mean(torch.tensor(train_losses))
        epoch_test_loss = torch.mean(torch.tensor(test_losses))

        train_avg_loss.append(epoch_train_loss)
        test_avg_loss.append(epoch_test_loss)
        
        wandb.log({"epoch": i + 1,"train_loss": epoch_train_loss,"test_loss": epoch_test_loss})
        print("Epoch "+str(i)+" : train_loss = "+str(epoch_train_loss)+" and test_loss = "+str(epoch_test_loss))
        
    return train_avg_loss, test_avg_loss

train_avg_loss, test_avg_loss, test_accuracy = train(parameters["num_eprochs"])