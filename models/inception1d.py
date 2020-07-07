class convbn(nn.Module):
    def __init__(self,channel_in, channel_out,kernel_size,stride=1,padding=0):
        super(convbn,self).__init__()
        self.convbn = nn.Sequential(nn.Conv1d(channel_in, channel_out, kernel_size, stride=stride,padding=padding),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(channel_out))

    def forward(self, x):
        return self.convbn(x)


class InceptionV2ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA,self).__init__()
        self.branch1 =convbn(channel_in=in_channels,channel_out=out_channels1,kernel_size=1)
        self.branch2=nn.Sequential(convbn(channel_in=in_channels,channel_out=out_channels2reduce,kernel_size=1),
                                   convbn(channel_in=out_channels2reduce,channel_out=out_channels2,kernel_size=5,padding=2))
        self.branch3=nn.Sequential(convbn(channel_in=in_channels,channel_out=out_channels3reduce,kernel_size=1),
                                   convbn(channel_in=out_channels3reduce,channel_out=out_channels3,kernel_size=3,padding=1),
                                   convbn(channel_in=out_channels3,channel_out=out_channels3,kernel_size=3,padding=1))
        self.branch4=nn.Sequential(nn.MaxPool1d(3,stride=1,padding=1),
                                   convbn(channel_in=in_channels,channel_out=out_channels4,kernel_size=1))

    def forward(self, x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out4=self.branch4(x)
        out=torch.cat([out1,out2,out3,out4],dim=1)
        return out


class InceptionV2ModuleD(nn.Module):
    def __init__(self,in_channels , out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV2ModuleD,self).__init__()
        self.branch1=convbn(channel_in=in_channels,channel_out=out_channels1,kernel_size=3,stride=2)
        self.branch2=nn.Sequential(convbn(channel_in=in_channels,channel_out=out_channels2reduce,kernel_size=1),
                                   convbn(channel_in=out_channels2reduce,channel_out=out_channels2,kernel_size=3,stride=1,padding=1),
                                   convbn(channel_in=out_channels2,channel_out=out_channels2,kernel_size=3,stride=2))
        self.branch3=nn.MaxPool1d(kernel_size=3,stride=2)

    def forward(self, x1):
        out1=self.branch1(x1)
        out2=self.branch2(x1)
        out3=self.branch3(x1)
        out=torch.cat([out1,out2,out3],dim=1)
        return out


class InceptionV2ModuleE(nn.Module):
    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV2ModuleE, self).__init__()
        self.branch1=nn.Sequential(convbn(channel_in=in_channels,channel_out=out_channels1reduce,kernel_size=1),
                                   convbn(channel_in=out_channels1reduce,channel_out=out_channels1,kernel_size=3,stride=2))
        self.branch2=nn.Sequential(convbn(channel_in=in_channels,channel_out=out_channels2reduce,kernel_size=1),
                                   convbn(channel_in=out_channels2reduce,channel_out=out_channels2reduce,kernel_size=7,padding=3),
                                   convbn(channel_in=out_channels2reduce,channel_out=out_channels2,kernel_size=3,stride=2))
        self.branch3=nn.MaxPool1d(kernel_size=3,stride=2)

    def forward(self, x1):
        out1=self.branch1(x1)
        out2=self.branch2(x1)
        out3=self.branch3(x1)
        out=torch.cat([out1,out2,out3],dim=1)
        return out


class Gene(nn.Module):
    def __init__(self):
        super(Gene,self).__init__()
        self.conv1_1=convbn(1,8,3,stride=2)
        self.conv1_2=convbn(8,8,3,stride=1)
        self.conv1_3=convbn(8,16,3,stride=1)
        self.pool1=nn.MaxPool1d(3,stride=2)

        self.conv2_1=convbn(16,32,3,stride=2)
        self.conv2_2 =convbn(32,32,3,stride=1)
        self.conv2_3 = convbn(32,64,3,stride=1)
        self.pool2 = nn.MaxPool1d(3, stride=2)

        self.block1=nn.Sequential(InceptionV2ModuleA(64,64,16,20,16,32,64))
        self.block2=nn.Sequential(InceptionV2ModuleD(180,150,30,50),
                                  InceptionV2ModuleA(380,200,64,100,64,100,200))
        self.block3=nn.Sequential(InceptionV2ModuleE(600,80,100,80,100))
        self.block4=nn.Sequential(InceptionV2ModuleE(800,80,100,80,100))
        self.pool3=nn.MaxPool1d(6)
    def forward(self, x1):
        x1=self.conv1_1(x1)
        x1=self.conv1_2(x1)
        x1=self.conv1_3(x1)
        x1=self.pool1(x1)
        x1=self.conv2_1(x1)
        x1=self.conv2_2(x1)
        x1=self.conv2_3(x1)
        x1=self.pool2(x1)
        x1=self.block1(x1)
        x1=self.block2(x1)
        x1=self.block3(x1)
        x1=self.block4(x1)
        x1=self.pool3(x1)
        x1 = x1.squeeze(2)
        return x1