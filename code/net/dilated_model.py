import torch.nn as nn
import torch



class conv_2(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, dilation, dropout=0.1):
        super().__init__()
        self.cn2 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                 nn.Sigmoid(),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride ,padding=0, dilation=dilation),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.cn2(x)
        return self.sigm(x)


class conv_3_2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.cn3_2 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Sigmoid(),
                                   nn.Conv2d(in_channels, out_channels, (3, 2), 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Dropout(dropout, inplace=True))
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.cn3_2(x)
        return self.sigm(x)


class conv_3_1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.cn3_1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Sigmoid(),
                                   nn.Conv2d(in_channels, out_channels, (3, 1), 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Dropout(dropout, inplace=True))
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.cn3_1(x)
        return self.sigm(x)


class conv_5_2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.cn5_2 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Sigmoid(),
                                   nn.Conv2d(in_channels, out_channels, (5, 2), 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Dropout(dropout, inplace=True))
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.cn5_2(x)
        return self.sigm(x)


class conv_5_1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.cn5_1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Sigmoid(),
                                   nn.Conv2d(in_channels, out_channels, (5, 1), 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Dropout(dropout, inplace=True))
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.cn5_1(x)
        return self.sigm(x)


class node2_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #layer 1
        self.conv2_1_1 = conv_2(in_channels, 2, (1, 2), 1, 1)
        self.conv2_1_2 = conv_2(2, 2, 2, 1, 1)
        self.conv2_1_3 = conv_2(2, out_channels, (1, 2), 1, 1)
        self.linear2_1 = nn.Linear(7, 10, bias=False)
        
        #layer 2
        self.conv2_2_1 = conv_2(in_channels, 2, (1, 2), 1, 2)
        self.conv2_2_2 = conv_2(2, out_channels, 2, 1, 1)
        self.linear2_2 = nn.Linear(7, 10, bias=False)
        
        #layer 3
        self.conv2_3_1 = conv_2(in_channels, 2, (1, 2), 1, 3)
        self.conv2_3_2 = conv_2(2, out_channels, (2, 1), 1, 1)
        self.linear2_3 = nn.Linear(7, 10, bias=False)
        
    def forward(self, x):
        x2_1 = self.conv2_1_1(x)
        x2_1 = self.conv2_1_2(x2_1)
        x2_1 = self.conv2_1_3(x2_1)
        x2_1 = self.linear2_1(x2_1)

        x2_2 = self.conv2_2_1(x)
        x2_2 = self.conv2_2_2(x2_2)
        x2_2 = self.linear2_2(x2_2)
        
        x2_3 = self.conv2_3_1(x)
        x2_3 = self.conv2_3_2(x2_3)
        x2_3 = self.linear2_3(x2_3)

        x = x[:, :, 0, :].reshape(x.shape[0], x.shape[1], 1, x.shape[3]) + x2_1 + x2_2 + x2_3

        return x


class node3_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #layer 1
        self.conv3_1_1 = conv_2(in_channels, 2, (1, 2), 1, 1)
        self.conv3_1_2 = conv_3_2(2, 2)
        self.conv3_1_3 = conv_2(in_channels, 2, (1, 2), 1, 1)
        self.linear3_1 = nn.Linear(7, 10, bias=False)
        
        #layer 2
        self.conv3_2_1 = conv_2(in_channels, 2, (1, 2), 1, 2)
        self.conv3_2_2 = conv_3_2(2, out_channels)
        self.linear3_2 = nn.Linear(7, 10, bias=False)
        
        #layer 3
        self.conv3_3_1 = conv_2(in_channels, 2, (1, 2), 1, 3)
        self.conv3_3_2 = conv_3_1(2, out_channels)
        self.linear3_3 = nn.Linear(7, 10, bias=False)

    def forward(self, x):
        x3_1 = self.conv3_1_1(x)
        x3_1 = self.conv3_1_2(x3_1)
        x3_1 = self.conv3_1_3(x3_1)
        x3_1 = self.linear3_1(x3_1)
        
        x3_2 = self.conv3_2_1(x)
        x3_2 = self.conv3_2_2(x3_2)
        x3_2 = self.linear3_2(x3_2)
        
        x3_3 = self.conv3_3_1(x)
        x3_3 = self.conv3_3_2(x3_3)
        x3_3 = self.linear3_3(x3_3)

        x = x[:, :, 0, :].reshape(x.shape[0], x.shape[1], 1, x.shape[3]) + x3_1 + x3_2 + x3_3
        return x


class node5_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #layer 1
        self.conv5_1_1 = conv_2(in_channels, 2, (1, 2), 1, 1)
        self.conv5_1_2 = conv_5_2(2, 2)
        self.conv5_1_3 = conv_2(in_channels, 2, (1, 2), 1, 1)
        self.linear5_1 = nn.Linear(7, 10, bias=False)
        
        #layer 2
        self.conv5_2_1 = conv_2(in_channels, 2, (1, 2), 1, 2)
        self.conv5_2_2 = conv_5_2(2, out_channels)
        self.linear5_2 = nn.Linear(7, 10, bias=False)
        
        #layer 3
        self.conv5_3_1 = conv_2(in_channels, 2, (1, 2), 1, 3)
        self.conv5_3_2 = conv_5_1(2, out_channels)
        self.linear5_3 = nn.Linear(7, 10, bias=False)
        

    def forward(self, x):
        x5_1 = self.conv5_1_1(x)
        x5_1 = self.conv5_1_2(x5_1)
        x5_1 = self.conv5_1_3(x5_1)
        x5_1 = self.linear5_1(x5_1)
        
        x5_2 = self.conv5_2_1(x)
        x5_2 = self.conv5_2_2(x5_2)
        x5_2 = self.linear5_2(x5_2)
        
        x5_3 = self.conv5_3_1(x)
        x5_3 = self.conv5_3_2(x5_3)
        x5_3 = self.linear5_3(x5_3)

        x = x[:, :, 0, :].reshape(x.shape[0], x.shape[1], 1, x.shape[3]) + x5_1 + x5_2 + x5_3
        return x


class refine_Module(nn.Module):

    def __init__(self, in_dim):
        super(refine_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        batchsize, C, node_num, frame = x.size()
        dep_1 = self.conv_1(x).reshape(batchsize, -1, node_num * frame).permute(0, 2, 1)
        dep_2 = self.conv_2(x).reshape(batchsize, -1, node_num * frame)
        dep = torch.bmm(dep_1, dep_2)
        impli_dep = self.softmax(dep)
        proj_data = x.reshape(batchsize, -1, node_num * frame)

        out = torch.bmm(proj_data, impli_dep.permute(0, 2, 1))
        out = out.reshape(batchsize, 1, node_num, frame)
        out = out + x
        return out


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(Generator, self).__init__()

        self.L_Hip_conv = node2_conv(1, 1)
        self.L_Knee_conv = node3_conv(1, 1)
        self.L_Foot_conv = node3_conv(1, 1)
        self.R_Hip_conv = node2_conv(1, 1)
        self.R_Knee_conv = node3_conv(1, 1)
        self.R_Foot_conv = node3_conv(1, 1)
        self.Thorax_conv = node5_conv(1, 1)
        self.Nose_conv = node3_conv(1, 1)
        self.Head_conv = node2_conv(1, 1)
        self.L_Shoulder_conv = node3_conv(1, 1)
        self.L_Elbow_conv = node3_conv(1, 1)
        self.L_Wrist_conv = node3_conv(1, 1)
        self.R_Shoulder_conv = node3_conv(1, 1)
        self.R_Elbow_conv = node3_conv(1, 1)
        self.R_Wrist_conv = node3_conv(1, 1)

        self.hidden_prev = nn.Parameter(torch.zeros(1, batch_size, hidden_size))
        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, dropout=0.05, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.refine_model = refine_Module(1)

    def forward(self, x, hidden_size):
        y = torch.zeros(x.shape[0], x.shape[1], x.shape[2])

        y = y.cuda()
        x = x.cuda()

        L_Hip = x[:, (6, 7), :]
        L_Knee = x[:, (7, 6, 8), :]
        L_Foot = x[:, (8, 7, 3), :]
        R_Hip = x[:, (1, 2), :]
        R_Knee = x[:, (2, 1, 3), :]
        R_Foot = x[:, (3, 2, 8), :]
        Thorax = x[:, (12, 13, 11, 15, 20), :]
        Nose = x[:, (13, 14, 12), :]
        Head = x[:, (14, 13), :]
        L_Shoulder = x[:, (15, 16, 12), :]
        L_Elbow = x[:, (16, 15, 17), :]
        L_Wrist = x[:, (17, 16, 22), :]
        R_Shoulder = x[:, (20, 21, 12), :]
        R_Elbow = x[:, (21, 20, 22), :]
        R_Wrist = x[:, (22, 21, 17), :]

        L_Hip = L_Hip.reshape(L_Hip.shape[0], 1, L_Hip.shape[1], L_Hip.shape[2])
        L_Knee = L_Knee.reshape(L_Knee.shape[0], 1, L_Knee.shape[1], L_Knee.shape[2])
        L_Foot = L_Foot.reshape(L_Foot.shape[0], 1, L_Foot.shape[1], L_Foot.shape[2])
        R_Hip = R_Hip.reshape(R_Hip.shape[0], 1, R_Hip.shape[1], R_Hip.shape[2])
        R_Knee = R_Knee.reshape(R_Knee.shape[0], 1, R_Knee.shape[1], R_Knee.shape[2])
        R_Foot = R_Foot.reshape(R_Foot.shape[0], 1, R_Foot.shape[1], R_Foot.shape[2])
        Thorax = Thorax.reshape(Thorax.shape[0], 1, Thorax.shape[1], Thorax.shape[2])
        Nose = Nose.reshape(Nose.shape[0], 1, Nose.shape[1], Nose.shape[2])
        Head = Head.reshape(Head.shape[0], 1, Head.shape[1], Head.shape[2])
        L_Shoulder = L_Shoulder.reshape(L_Shoulder.shape[0], 1, L_Shoulder.shape[1], L_Shoulder.shape[2])
        L_Elbow = L_Elbow.reshape(L_Elbow.shape[0], 1, L_Elbow.shape[1], L_Elbow.shape[2])
        L_Wrist = L_Wrist.reshape(L_Wrist.shape[0], 1, L_Wrist.shape[1], L_Wrist.shape[2])
        R_Shoulder = R_Shoulder.reshape(R_Shoulder.shape[0], 1, R_Shoulder.shape[1], R_Shoulder.shape[2])
        R_Elbow = R_Elbow.reshape(R_Elbow.shape[0], 1, R_Elbow.shape[1], R_Elbow.shape[2])
        R_Wrist = R_Wrist.reshape(R_Wrist.shape[0], 1, R_Wrist.shape[1], R_Wrist.shape[2])

        pr_L_Hip = self.L_Hip_conv(L_Hip)  # 6
        pr_L_Knee = self.L_Knee_conv(L_Knee)  # 7
        pr_L_Foot = self.L_Foot_conv(L_Foot)  # 8
        pr_R_Hip = self.R_Hip_conv(R_Hip)  # 1
        pr_R_Knee = self.R_Knee_conv(R_Knee)  # 2
        pr_R_Foot = self.R_Foot_conv(R_Foot)  # 3
        pr_Thorax = self.Thorax_conv(Thorax)  # 12
        pr_Nose = self.Nose_conv(Nose)  # 13
        pr_Head = self.Head_conv(Head)  # 14
        pr_L_Shoulder = self.L_Shoulder_conv(L_Shoulder)  # 15
        pr_L_Elbow = self.L_Elbow_conv(L_Elbow)  # 16
        pr_L_Wrist = self.L_Wrist_conv(L_Wrist)  # 17
        pr_R_Shoulder = self.R_Shoulder_conv(R_Shoulder)  # 20
        pr_R_Elbow = self.R_Elbow_conv(R_Elbow)  # 21
        pr_R_Wrist = self.R_Wrist_conv(R_Wrist)  # 22

        pr_L_Hip = pr_L_Hip.reshape(pr_L_Hip.shape[0], pr_L_Hip.shape[3])
        pr_L_Knee = pr_L_Knee.reshape(pr_L_Knee.shape[0], pr_L_Knee.shape[3])
        pr_L_Foot = pr_L_Foot.reshape(pr_L_Foot.shape[0], pr_L_Foot.shape[3])
        pr_R_Hip = pr_R_Hip.reshape(pr_R_Hip.shape[0], pr_R_Hip.shape[3])
        pr_R_Knee = pr_R_Knee.reshape(pr_R_Knee.shape[0], pr_R_Knee.shape[3])
        pr_R_Foot = pr_R_Foot.reshape(pr_R_Foot.shape[0], pr_R_Foot.shape[3])
        pr_Thorax = pr_Thorax.reshape(pr_Thorax.shape[0], pr_Thorax.shape[3])
        pr_Nose = pr_Nose.reshape(pr_Nose.shape[0], pr_Nose.shape[3])
        pr_Head = pr_Head.reshape(pr_Head.shape[0], pr_Head.shape[3])
        pr_L_Shoulder = pr_L_Shoulder.reshape(pr_L_Shoulder.shape[0], pr_L_Shoulder.shape[3])
        pr_L_Elbow = pr_L_Elbow.reshape(pr_L_Elbow.shape[0], pr_L_Elbow.shape[3])
        pr_L_Wrist = pr_L_Wrist.reshape(pr_L_Wrist.shape[0], pr_L_Wrist.shape[3])
        pr_R_Shoulder = pr_R_Shoulder.reshape(pr_R_Shoulder.shape[0], pr_R_Shoulder.shape[3])
        pr_R_Elbow = pr_R_Elbow.reshape(pr_R_Elbow.shape[0], pr_R_Elbow.shape[3])
        pr_R_Wrist = pr_R_Wrist.reshape(pr_R_Wrist.shape[0], pr_R_Wrist.shape[3])

        y[:, 0] = x[:, 0]
        y[:, 1] = pr_R_Hip
        y[:, 2] = pr_R_Knee
        y[:, 3] = pr_R_Foot
        y[:, 4] = x[:, 4]
        y[:, 5] = x[:, 5]
        y[:, 6] = pr_L_Hip
        y[:, 7] = pr_L_Knee
        y[:, 8] = pr_L_Foot
        y[:, 9] = x[:, 9]
        y[:, 10] = x[:, 10]
        y[:, 11] = x[:, 11]
        y[:, 12] = pr_Thorax
        y[:, 13] = pr_Nose
        y[:, 14] = pr_Head
        y[:, 15] = pr_L_Shoulder
        y[:, 16] = pr_L_Elbow
        y[:, 17] = pr_L_Wrist
        y[:, 18] = x[:, 18]
        y[:, 19] = x[:, 19]
        y[:, 20] = pr_R_Shoulder
        y[:, 21] = pr_R_Elbow
        y[:, 22] = pr_R_Wrist
        y[:, 23] = x[:, 23]
        y[:, 24] = x[:, 24]
        x = x + y

        out, h = self.GRU(x, self.hidden_prev)
        out = out.reshape(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        out = out.reshape(x.shape[0], 25, 25)

        ref_out = self.refine_model(out)
        out = ref_out.reshape(1, x.shape[0] * 25, 25)
        return out, h
