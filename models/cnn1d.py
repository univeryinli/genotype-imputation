class Gene(nn.Module):
    def __init__(self):
        super(Gene, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 8, 5, stride=3, padding=0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(8, 16, 5, stride=3, padding=0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(16, 32, 5, stride=3, padding=0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(32, 64, 3, stride=2, padding=0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(64, 128, 3, stride=2, padding=0))
        self.pool = nn.MaxPool1d(8)
        self.linear = nn.Linear(128, 1000)

    def forward(self, x):
        x=self.conv(x)
        x=self.pool(x)
        x=x.squeeze(2)
        x=self.linear(x)
        return x