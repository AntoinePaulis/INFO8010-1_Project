import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=False, dropout_p=0.2):
        super().__init__()
        
        self.dropout = dropout
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            # ReLU after the bacth normalization because however, batch normalization will create some negative values
            nn.ReLU()
        )
        
        if self.dropout:
            self.dropout = nn.Dropout2d(p=dropout_p)
    
    def forward(self, x):
        x = self.net(x)
        if self.dropout:
            x = self.dropout(x)
        return

class TrackNetCourt(nn.Module):
    def __init__(self, weight_init,dropout=False, dropout_p=0.2):
        # The number of input frames is a parameter cfr. Tracknet paper
        super().__init__()
        self.net = nn.Sequential (
            Block(in_channels=3, out_channels=64, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=64, out_channels=64, dropout=dropout, dropout_p=dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2), # if MaxPool2d(), stride would default at kernel size = 3. we enforce it to 2
            Block(in_channels=64, out_channels=128, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=128, out_channels=128, dropout=dropout, dropout_p=dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=128, out_channels=256, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=256, out_channels=256, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=256, out_channels=256, dropout=dropout, dropout_p=dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=256, out_channels=512, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=512, out_channels=512, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=512, out_channels=512, dropout=dropout, dropout_p=dropout_p),
            nn.Upsample(scale_factor=2),
            Block(in_channels=512, out_channels=256, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=256, out_channels=256, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=256, out_channels=256, dropout=dropout, dropout_p=dropout_p),
            nn.Upsample(scale_factor=2),
            Block(in_channels=256, out_channels=128, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=128, out_channels=128, dropout=dropout, dropout_p=dropout_p),
            nn.Upsample(scale_factor=2),
            Block(in_channels=128, out_channels=64, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=64, out_channels=64, dropout=dropout, dropout_p=dropout_p),
            Block(in_channels=64, out_channels=15, dropout=dropout, dropout_p=dropout_p)
        )

        self._init_weights(weight_init = weight_init)
        
    def forward(self, x):
        x = self.net(x)
        return x
    
    def _init_weights(self, weight_init):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if weight_init == "uniform":
                    nn.init.uniform_(m.weight, -0.05, 0.05)
                elif weight_init == "he":
                     nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif weight_init == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  