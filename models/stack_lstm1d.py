class Gene(nn.Module):
    def __init__(self):
        super(Gene,self).__init__()
        self.emed = nn.Embedding(3, 3, padding_idx=0)
        self.rnn1 =nn.LSTM(3,16,num_layers=3,bidirectional=True)
        self.rnn2 = nn.LSTM(16, 64)
        self.rnn3 = nn.LSTM(64, 64)
        self.linear = nn.Linear(3, 1)

    def forward(self, x,mask):
        x = self.emed(x.long())
        x = x.transpose(1, 0)
        x = self.rnn1(x, x)
        x= self.rnn2(x)
        x= self.rnn3(x)
        x = self.linear(x)
        x = x.transpose(1, 0)
        x = x.squeeze(2)
        return x