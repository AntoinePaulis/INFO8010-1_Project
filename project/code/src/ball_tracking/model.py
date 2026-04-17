import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            # ReLU after the bacth normalization because however, batch normalization will create some negative values
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class TrackNet(nn.Module):
    def __init__(self, nb_input_frames=3):
        # The number of input frames is a parameter cfr. Tracknet paper
        super().__init__()
        self.net = nn.Sequential (
            Block(in_channels=3*nb_input_frames, out_channels=64),
            Block(in_channels=64, out_channels=64),
            nn.MaxPool2d(2),
            Block(in_channels=64, out_channels=128),
            Block(in_channels=128, out_channels=128),
            nn.MaxPool2d(2),
            Block(in_channels=128, out_channels=256),
            Block(in_channels=256, out_channels=256),
            Block(in_channels=256, out_channels=256),
            nn.MaxPool2d(2),
            Block(in_channels=256, out_channels=512),
            Block(in_channels=512, out_channels=512),
            Block(in_channels=512, out_channels=512),
            nn.Upsample(2),
            Block(in_channels=512, out_channels=256),
            Block(in_channels=256, out_channels=256),
            Block(in_channels=256, out_channels=256),
            nn.Upsample(2),
            Block(in_channels=256, out_channels=128),
            Block(in_channels=128, out_channels=128),
            nn.Upsample(scale_factor=2),
            Block(in_channels=128, out_channels=64),
            Block(in_channels=64, out_channels=64),
            Block(in_channels=64, out_channels=1),
            # 256 ???
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x