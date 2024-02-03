import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from baselines.HiVT.hivt import hivt
from baselines.AutoBots.autobots import autobots 


class Transformer(nn.Module):#256,4
    def __init__(self,C, input_feats=27,latent_dim=256,num_heads=4,ff_size=1024, in_dim = 4,out_dim = 8, dropout=0.1, num_layers=8, activation="gelu"):
        super().__init__()
        C.IMU19_dim = C.IMU19_dim + 200 if C.args.dataset == "h3d" else  C.IMU19_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.output_channels = C.BBX5_dim +1 if C.args.dataset == "h3d" else C.BBX5_dim
        if C.args.is_pos:
            C.IMU19_dim = C.IMU19_dim +5
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.input_process = nn.Linear(2, self.latent_dim)
        self.input_process = nn.Linear(in_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim ,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(self.seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        self.output_process = nn.Linear(self.latent_dim, self.out_dim)
        # self.output_process = nn.Linear(self.latent_dim*100, self.out_dim*100)
    
    def forward(self, bbx_x):
        # bs, nframes, n_channels = x.shape  # should # frame, bs, njoints*nfeats
        # x = torch.cat([bbx_x,  imu_x], dim=2)
        # x = torch.cat([bbx_x, ftm_x, imu_x], dim=2)
        x = bbx_x
        
        # x = x.permute((1,0,2))

        x = self.input_process(x)

        # for i in range(3):
            # t = torch.ones(x.shape[0:2]).unsqueeze(-1).cuda()
            # t[:]=i
        
        x = self.sequence_pos_encoder(x)  # [seqlen+1, bs, d]

        # x = torch.cat([t,x],-1)
        x = self.seqTransEncoder(x)#[1:] 

        output = self.output_process(x)
        # output = self.output_process(x.flatten(1)).view(-1,100,4)
        return output
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class LSTM(nn.Module):
    def __init__(self, C, in_dim = 2, out_dim=4):
        super().__init__()
        if C.args.is_pos:
            C.IMU19_dim = C.IMU19_dim +5
        # ---------------
        #  BBX5 Encoder
        # ---------------
        # self.bbx_conv = nn.ReLU(nn.Conv1d(input_size=C.BBX5_dim, hidden_size=C.n_filters, num_layers=3, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.bbx_lstm1 = nn.LSTM(in_dim, hidden_size= C.h_BBX5_dim, num_layers=3, bidirectional=True, batch_first=True)
        self.bbx_lstm = nn.ReLU()
        # --------------------
        #  FTM2 Encoder
        # --------------------
        # self.ftm_conv = nn.ReLU(nn.Conv1d(C.FTM2_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        # self.ftm_lstm = nn.ReLU(nn.LSTM(C.FTM2_dim, hidden_size= C.h_FTM2_dim, num_layers=3,bidirectional=True, batch_first=True))
        # ---------------
        #  IMU19 Encoder
        # ---------------
        # self.imu_conv = nn.ReLU(nn.Conv1d(C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        # self.imu_lstm = nn.ReLU(nn.LSTM(C.IMU19_dim, hidden_size= C.h_IMU19_dim, num_layers=3,bidirectional=True, batch_first=True))

        # --------------
        #  BBX5 Decoder
        # --------------
        # self.decoder_conv1 = nn.Conv1d(C.BBX5_dim + C.FTM2_dim + C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate')
        # self.decoder_conv1 = nn.Conv1d(128, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate')
        # self.decoder_conv = nn.ReLU()
        # self.decoder_lstm11 = nn.LSTM(C.n_filters, hidden_size= C.h_fused_dim, num_layers=3, bidirectional=True, batch_first=True)
        # self.decoder_lstm1 = nn.ReLU()
        # self.decoder_lstm21 = nn.LSTM(C.h_fused_dim, hidden_size= C.BBX5_dim, num_layers=2, bidirectional=True, batch_first=True)
        # self.decoder_lstm2 = nn.ReLU()
        
        if C.args.dataset == "h3d":
            self.dense = nn.Linear(2 , C.BBX5_dim+1)
        else:
            # self.dense = nn.Linear( 6 + C.FTM2_dim + C.IMU19_dim , C.BBX5_dim*2)
            self.dense = nn.Linear( 64 , out_dim)

    def forward(self, bbx_x): # , ftm_x, imu_x): # interl_gps, nearst_gps, imu
        # bbx_x = self.bbx_conv(bbx_x)
        bbx_x1, _ = self.bbx_lstm1(bbx_x)
        bbx_x2= self.bbx_lstm(bbx_x1)

        # ftm_x = self.ftm_conv(ftm_x)
        # ftm_x = self.ftm_lstm(ftm_x)

        # imu_x = self.imu_conv(imu_x)
        # imu_x = self.imu_lstm(imu_x)

        # x =  bbx_x + ftm_x + imu_x
        # x = torch.cat([bbx_x , ftm_x , imu_x],dim=2)
        # x = bbx_x1
        # x1 = self.decoder_conv1(bbx_x2)
        # x2 = self.decoder_conv(x1)
        # _,x3 = self.decoder_lstm11(x2)
        # x4 = self.decoder_lstm1(x3[0])
        # _,x5  = self.decoder_lstm21(x4)
        # x6 = self.decoder_lstm2(x5[0])
        x7 = self.dense(bbx_x2)
        
        return x7 #.view(-1, 200, 2, 4)

class GRU(nn.Module):
    def __init__(self, C, in_dim = 2, out_dim=4):
        super().__init__()
        if C.args.is_pos:
            C.IMU19_dim = C.IMU19_dim +5
        # ---------------
        #  BBX5 Encoder
        # ---------------
        # self.bbx_conv = nn.ReLU(nn.Conv1d(input_size=C.BBX5_dim, hidden_size=C.n_filters, num_layers=3, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.bbx_lstm1 = nn.GRU(in_dim, hidden_size= C.h_BBX5_dim, num_layers=3, bidirectional=True, batch_first=True)
        self.bbx_lstm = nn.ReLU()
        # --------------------
        #  FTM2 Encoder
        # --------------------
        # self.ftm_conv = nn.ReLU(nn.Conv1d(C.FTM2_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        # self.ftm_lstm = nn.ReLU(nn.LSTM(C.FTM2_dim, hidden_size= C.h_FTM2_dim, num_layers=3,bidirectional=True, batch_first=True))
        # ---------------
        #  IMU19 Encoder
        # ---------------
        # self.imu_conv = nn.ReLU(nn.Conv1d(C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        # self.imu_lstm = nn.ReLU(nn.LSTM(C.IMU19_dim, hidden_size= C.h_IMU19_dim, num_layers=3,bidirectional=True, batch_first=True))

        # --------------
        #  BBX5 Decoder
        # --------------
        # self.decoder_conv1 = nn.Conv1d(C.BBX5_dim + C.FTM2_dim + C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate')
        # self.decoder_conv1 = nn.Conv1d(128, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate')
        # self.decoder_conv = nn.ReLU()
        # self.decoder_lstm11 = nn.LSTM(C.n_filters, hidden_size= C.h_fused_dim, num_layers=3, bidirectional=True, batch_first=True)
        # self.decoder_lstm1 = nn.ReLU()
        # self.decoder_lstm21 = nn.LSTM(C.h_fused_dim, hidden_size= C.BBX5_dim, num_layers=2, bidirectional=True, batch_first=True)
        # self.decoder_lstm2 = nn.ReLU()
        
        if C.args.dataset == "h3d":
            self.dense = nn.Linear(2 , C.BBX5_dim+1)
        else:
            # self.dense = nn.Linear( 6 + C.FTM2_dim + C.IMU19_dim , C.BBX5_dim*2)
            self.dense = nn.Linear( 64 , out_dim)

    def forward(self, bbx_x): # , ftm_x, imu_x): # interl_gps, nearst_gps, imu
        # bbx_x = self.bbx_conv(bbx_x)
        bbx_x1, _ = self.bbx_lstm1(bbx_x)
        bbx_x2= self.bbx_lstm(bbx_x1)

        # ftm_x = self.ftm_conv(ftm_x)
        # ftm_x = self.ftm_lstm(ftm_x)

        # imu_x = self.imu_conv(imu_x)
        # imu_x = self.imu_lstm(imu_x)

        # x =  bbx_x + ftm_x + imu_x
        # x = torch.cat([bbx_x , ftm_x , imu_x],dim=2)
        # x = bbx_x1
        # x1 = self.decoder_conv1(bbx_x2)
        # x2 = self.decoder_conv(x1)
        # _,x3 = self.decoder_lstm11(x2)
        # x4 = self.decoder_lstm1(x3[0])
        # _,x5  = self.decoder_lstm21(x4)
        # x6 = self.decoder_lstm2(x5[0])
        x7 = self.dense(bbx_x2)
        
        return x7 #.view(-1, 200, 2, 4)

class RNN(nn.Module):
    def __init__(self, C, in_dim = 2, out_dim=4):
        super().__init__()
        if C.args.is_pos:
            C.IMU19_dim = C.IMU19_dim +5
        # ---------------
        #  BBX5 Encoder
        # ---------------
        # self.bbx_conv = nn.ReLU(nn.Conv1d(input_size=C.BBX5_dim, hidden_size=C.n_filters, num_layers=3, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.bbx_lstm1 = nn.RNN(in_dim, hidden_size= C.h_BBX5_dim, num_layers=3, bidirectional=True, batch_first=True)
        self.bbx_lstm = nn.ReLU()
        # --------------------
        #  FTM2 Encoder
        # --------------------
        # self.ftm_conv = nn.ReLU(nn.Conv1d(C.FTM2_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        # self.ftm_lstm = nn.ReLU(nn.LSTM(C.FTM2_dim, hidden_size= C.h_FTM2_dim, num_layers=3,bidirectional=True, batch_first=True))
        # ---------------
        #  IMU19 Encoder
        # ---------------
        # self.imu_conv = nn.ReLU(nn.Conv1d(C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        # self.imu_lstm = nn.ReLU(nn.LSTM(C.IMU19_dim, hidden_size= C.h_IMU19_dim, num_layers=3,bidirectional=True, batch_first=True))

        # --------------
        #  BBX5 Decoder
        # --------------
        # self.decoder_conv1 = nn.Conv1d(C.BBX5_dim + C.FTM2_dim + C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate')
        # self.decoder_conv1 = nn.Conv1d(128, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate')
        # self.decoder_conv = nn.ReLU()
        # self.decoder_lstm11 = nn.LSTM(C.n_filters, hidden_size= C.h_fused_dim, num_layers=3, bidirectional=True, batch_first=True)
        # self.decoder_lstm1 = nn.ReLU()
        # self.decoder_lstm21 = nn.LSTM(C.h_fused_dim, hidden_size= C.BBX5_dim, num_layers=2, bidirectional=True, batch_first=True)
        # self.decoder_lstm2 = nn.ReLU()
        
        if C.args.dataset == "h3d":
            self.dense = nn.Linear(2 , C.BBX5_dim+1)
        else:
            # self.dense = nn.Linear( 6 + C.FTM2_dim + C.IMU19_dim , C.BBX5_dim*2)
            self.dense = nn.Linear( 64 , out_dim)

    def forward(self, bbx_x): # , ftm_x, imu_x): # interl_gps, nearst_gps, imu
        # bbx_x = self.bbx_conv(bbx_x)
        bbx_x1, _ = self.bbx_lstm1(bbx_x)
        bbx_x2= self.bbx_lstm(bbx_x1)

        # ftm_x = self.ftm_conv(ftm_x)
        # ftm_x = self.ftm_lstm(ftm_x)

        # imu_x = self.imu_conv(imu_x)
        # imu_x = self.imu_lstm(imu_x)

        # x =  bbx_x + ftm_x + imu_x
        # x = torch.cat([bbx_x , ftm_x , imu_x],dim=2)
        # x = bbx_x1
        # x1 = self.decoder_conv1(bbx_x2)
        # x2 = self.decoder_conv(x1)
        # _,x3 = self.decoder_lstm11(x2)
        # x4 = self.decoder_lstm1(x3[0])
        # _,x5  = self.decoder_lstm21(x4)
        # x6 = self.decoder_lstm2(x5[0])
        x7 = self.dense(bbx_x2)
        
        return x7 #.view(-1, 200, 2, 4)


class NMT(nn.Module):
    def __init__(self, C):
        super().__init__()
        if C.args.is_pos:
            C.IMU19_dim = C.IMU19_dim +5
        # ---------------
        #  BBX5 Encoder
        # ---------------
        kernel_size=2
        self.bbx_conv = nn.ReLU(nn.Conv1d(100,100, kernel_size=kernel_size, stride=1, padding_mode='replicate'))
        self.bbx_lstm = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_BBX5_dim, num_layers=3, bidirectional=True, batch_first=True))
        # --------------------
        #  FTM2 Encoder
        # --------------------
        self.ftm_conv = nn.ReLU(nn.Conv1d(100,100, kernel_size=kernel_size, stride=1, padding_mode='replicate'))
        self.ftm_lstm = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_FTM2_dim, num_layers=3, bidirectional=True, batch_first=True))
        # ---------------
        #  IMU19 Encoder
        # ---------------
        self.imu_conv = nn.ReLU(nn.Conv1d(100,100, kernel_size=kernel_size, stride=1, padding_mode='replicate'))
        self.imu_lstm = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_IMU19_dim, num_layers=3, bidirectional=True, batch_first=True))

        # --------------
        #  BBX5 Decoder
        # --------------
        self.decoder_conv = nn.ReLU(nn.Conv1d(C.BBX5_dim + C.FTM2_dim + C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.decoder_lstm1 = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_fused_dim, num_layers=2, bidirectional=True, batch_first=True))
        self.decoder_lstm2 = nn.ReLU(nn.LSTM(C.h_fused_dim, hidden_size= C.BBX5_dim, num_layers=2, bidirectional=True, batch_first=True))
        self.dense = nn.Linear(C.BBX5_dim + C.FTM2_dim + C.IMU19_dim, C.BBX5_dim)

    def forward(self, bbx_x, ftm_x, imu_x):
        bbx_x = self.bbx_conv(bbx_x)
        bbx_x = self.bbx_lstm(bbx_x)

        ftm_x = self.ftm_conv(ftm_x)
        ftm_x = self.ftm_lstm(ftm_x)

        imu_x = self.imu_conv(imu_x)
        imu_x = self.imu_lstm(imu_x)

        # x =  bbx_x + ftm_x + imu_x
        x = torch.cat([bbx_x , ftm_x , imu_x],dim=2)
        x = self.decoder_conv(x)
        x = self.decoder_lstm1(x)
        x = self.decoder_lstm2(x)
        x = self.dense(x)
        
        return x


class ViTag(nn.Module):
    def __init__(self, C, in_dim = 4, out_dim=4):
        self.C = C
        if C.args.is_pos:
            C.IMU19_dim = C.IMU19_dim +5
        n_filters=100
        kernel_size=2
        dense_in =   (C.BBX5_dim + C.FTM2_dim + C.IMU19_dim) +2 #if C.args.is_pos else (C.BBX5_dim + C.FTM2_dim + C.IMU19_dim + 5)
        super().__init__()
        # ---------------
        #  BBX5 Encoder
        # ---------------
        self.bbx_conv =  nn.Sequential( nn.Conv1d(100, n_filters, kernel_size=kernel_size, stride=1, padding_mode='replicate'))
        self.bbx_lstm = nn.Sequential( nn.LSTM(3, hidden_size= 4, bidirectional=True, batch_first=True))
        # --------------------
        #  FTM2 Encoder
        # --------------------
        self.ftm_conv = nn.Sequential( nn.Conv1d(100, n_filters, kernel_size=kernel_size, stride=1, padding_mode='replicate'))
        self.ftm_lstm = nn.Sequential( nn.LSTM(1, hidden_size= 2, bidirectional=True, batch_first=True))
        # ---------------
        #  IMU19 Encoder
        # ---------------
        self.imu_conv = nn.Sequential( nn.Conv1d(100, n_filters, kernel_size=kernel_size, stride=1, padding_mode='replicate'))
        self.imu_lstm = nn.Sequential( nn.LSTM(1, hidden_size= 2, bidirectional=True, batch_first=True))

        # --------------
        #  BBX5 Decoder
        # --------------
        self.decoder_conv = nn.Sequential( nn.Conv1d(100, C.n_filters, kernel_size=kernel_size, stride=1, padding_mode='replicate'))
        # self.decoder_lstm1 = nn.Sequential( nn.LSTM(15, hidden_size= C.h_fused_dim, bidirectional=True, batch_first=True))
        # self.decoder_lstm2 = nn.Sequential( nn.LSTM(64, hidden_size= 50, bidirectional=True, batch_first=True))
        
        self.decoder_lstm2 = nn.Sequential( nn.LSTM(15, hidden_size= 50, bidirectional=True, batch_first=True))
        if C.args.dataset == "h3d":
            self.dense = nn.Linear(dense_in+200, C.BBX5_dim+1)
        else:
            self.dense = nn.Linear(32, out_dim)

    def forward(self, x):
        bbx_x, ftm_x, imu_x = x[..., :4], x[..., -4:-2], x[..., -2:]
        bbx_x = self.bbx_conv(bbx_x)
        bbx_x = F.relu(self.bbx_lstm(bbx_x)[0])

        ftm_x = self.ftm_conv(ftm_x)
        ftm_x = F.relu(self.ftm_lstm(ftm_x)[0])

        imu_x = self.imu_conv(imu_x)
        imu_x = F.relu(self.imu_lstm(imu_x)[0])

        # x =  bbx_x + ftm_x + imu_x
        x = torch.cat([bbx_x , ftm_x , imu_x],dim=2)
        x = F.relu(self.decoder_conv(x))
        # x = self.decoder_lstm1(x)[0]
        x = self.decoder_lstm2(x)[0].permute([0,2,1])
        x = self.dense(x).view(-1, 100, 4)
        
        return x #.view(-1, 200, 2, 4)
    

# 1222/2000        0.0355          0.0355          0.0367          0.0367
class SingleLayerViTag(nn.Module):
    def __init__(self, C):
        super().__init__()
        if C.args.is_pos:
            C.IMU19_dim = C.IMU19_dim +5
        # ---------------
        #  BBX5 Encoder
        # ---------------
        self.bbx_conv = nn.ReLU(nn.Conv1d(C.BBX5_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.bbx_lstm = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_BBX5_dim, num_layers=3, bidirectional=True, batch_first=True))
        # --------------------
        #  FTM2 Encoder
        # --------------------
        self.ftm_conv = nn.ReLU(nn.Conv1d(C.FTM2_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.ftm_lstm = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_FTM2_dim, num_layers=3, bidirectional=True, batch_first=True))
        # ---------------
        #  IMU19 Encoder
        # ---------------
        self.imu_conv = nn.ReLU(nn.Conv1d(C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.imu_lstm = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_IMU19_dim, num_layers=3, bidirectional=True, batch_first=True))

        # --------------
        #  BBX5 Decoder
        # --------------
        self.decoder_conv = nn.ReLU(nn.Conv1d(C.BBX5_dim + C.FTM2_dim + C.IMU19_dim, C.n_filters, kernel_size=C.kernel_size, stride=1, padding_mode='replicate'))
        self.decoder_lstm1 = nn.ReLU(nn.LSTM(C.n_filters, hidden_size= C.h_fused_dim, num_layers=2, bidirectional=True, batch_first=True))
        self.decoder_lstm2 = nn.ReLU(nn.LSTM(C.h_fused_dim, hidden_size= C.BBX5_dim, num_layers=2, bidirectional=True, batch_first=True))
        self.dense = nn.Linear(C.BBX5_dim + C.FTM2_dim + C.IMU19_dim, C.BBX5_dim)

    def forward(self, bbx_x, ftm_x, imu_x):
        bbx_x = self.bbx_conv(bbx_x)
        bbx_x = self.bbx_lstm(bbx_x)

        ftm_x = self.ftm_conv(ftm_x)
        ftm_x = self.ftm_lstm(ftm_x)

        imu_x = self.imu_conv(imu_x)
        imu_x = self.imu_lstm(imu_x)

        # x =  bbx_x + ftm_x + imu_x
        x = torch.cat([bbx_x , ftm_x , imu_x],dim=2)
        x = self.decoder_conv(x)
        x = self.decoder_lstm1(x)
        x = self.decoder_lstm2(x)
        x = self.dense(x)
        
        return x


from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, C, in_dim = 4, out_dim=4):
        super(UNet, self).__init__()
        is_nosiy_traj=C.args.is_pos
        bilinear=False
        n_channels = 1
        n_classes = 2
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, mid_channels=32))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.dataset = C.args.dataset
        if is_nosiy_traj:
            # self.dense= nn.Linear(4160, 200*4)
            self.dense= nn.Linear(14560, 200*4)
        elif C.args.dataset == "h3d":
            self.dense= nn.Linear(960, 40*6)
            
        else:
            if C.args.dataset == "jrdb":
                in_c = 3888 if in_dim == 10 else  3024
            else:
                in_c = 3456 if in_dim == 8 else 2592
            self.dense= nn.Linear(in_c, 100*out_dim)

    def forward(self, bbx_x):
        # x = torch.cat([bbx_x, ftm_x, imu_x], dim=2)
        x= bbx_x
        x = x.unsqueeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.dense(x)
        if self.dataset == "h3d":
            logits = logits.view(-1,40,6)
        else:
            logits = logits.view(-1,100,4)
        
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    # def old(self,):
    #     # ---------------
    #     #  BBX5 Encoder
    #     # ---------------
    #     in_BBX5 = Input(shape=(C.recent_K, C.BBX5_dim,))
    #     conv1_BBX5 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
    #                     activation='relu', padding='same')(in_BBX5)
    #     en_BBX5 = Bidirectional(LSTM(C.h_BBX5_dim, activation='relu', \
    #             batch_input_shape=(C.n_batch, ), return_sequences=True))(conv1_BBX5) # in_BBX5)
    #     # --------------------
    #     #  FTM2 Encoder
    #     # --------------------
    #     in_FTM2 = Input(shape=(C.recent_K, C.FTM2_dim,))
    #     conv1_FTM2 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
    #                     activation='relu', padding='same')(in_FTM2)
    #     en_FTM2 = Bidirectional(LSTM(C.h_FTM2_dim, activation='relu', \
    #             batch_input_shape=(C.n_batch, ), return_sequences=True))(conv1_FTM2)
    #     # ---------------
    #     #  IMU19 Encoder
    #     # ---------------
    #     in_IMU19 = Input(shape=(C.recent_K, C.IMU19_dim,))
    #     conv1_IMU19 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
    #                     activation='relu', padding='same')(in_IMU19)
    #     en_IMU19 = Bidirectional(LSTM(C.h_IMU19_dim, activation='relu', \
    #             batch_input_shape=(C.n_batch, ), return_sequences=True))(conv1_IMU19) # in_IMU19)
    #     # ----------------------
    #     #  Joint Representation
    #     # ----------------------
    #     # in_fused_add = Add()([en_BBX5, en_IMU19])
    #     in_fused_add = Add()([en_BBX5, en_IMU19, en_FTM2]) ####################################### TODO: why don't en_FTM2, too large? * 0.001?
    #     # in_fused_concat = Concatenate()([en_BBX5, en_IMU19])
    #     # in_fused = RepeatVector(C.recent_K)(in_fused_concat)
    #     # --------------
    #     #  BBX5 Decoder
    #     # --------------
    #     # de_BBX5 = RepeatVector(C.recent_K)(in_fused_add)
    #     de_BBX5 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
    #                     activation='relu', padding='same')(in_fused_add)
    #     de_BBX5 = Bidirectional(LSTM(C.h_fused_dim, activation='relu', \
    #             batch_input_shape=(C.n_batch, ), return_sequences=True))(de_BBX5) # in_fused_add) # in_fused)
    #     de_BBX5 = Bidirectional(LSTM(C.BBX5_dim, activation='relu', \
    #             batch_input_shape=(C.n_batch, ), return_sequences=True))(de_BBX5) # in_fused_add) # de_BBX5)
    #     de_BBX5 = TimeDistributed(Dense(C.BBX5_dim))(de_BBX5)

    #     # ------------
    #     #  Base Model
    #     # ------------
    #     # BaseNet = Model([in_BBX5, in_FTM2, in_IMU19], [de_BBX5, de_FTM2, de_IMU19])
    #     BaseNet = Model([in_BBX5, in_FTM2, in_IMU19], [de_BBX5]) ###########################################TODO: Model Input & Output
    #     print('BaseNet.summary(): ', BaseNet.summary())


    #     sl_rec_BBX5_0 = BaseNet([in_BBX5, in_FTM2, in_IMU19])

    #     def Bhatt_loss(y_true, y_pred):
    #         small_num = 0.000001
    #         # print('np.shape(y_true): ', np.shape(y_true)) # debug # e.g. (None, 10, 2)
    #         y_true, y_pred = tf.cast(y_true, dtype='float64'), tf.cast(y_pred, dtype='float64')
    #         mu_true, sig_true = tf.cast(y_true[:,:,0], dtype='float64'), tf.cast(y_true[:,:,1], dtype='float64')
    #         mu_pred, sig_pred = tf.cast(y_pred[:,:,0], dtype='float64'), tf.cast(y_pred[:,:,1], dtype='float64')
    #         # print('np.shape(mu_true): ', np.shape(mu_true), ', np.shape(sig_true): ', np.shape(sig_true))
    #         # print('mu_true: ', mu_true, ', sig_true: ', sig_true)
    #         term0 = tf.math.truediv(tf.math.log(tf.math.add(tf.math.add(tf.math.truediv(\
    #                         tf.math.truediv(tf.math.pow(sig_true, 2), tf.math.pow(sig_pred, 2) + small_num), 4.), \
    #                         tf.math.truediv(tf.math.pow(sig_pred, 2), tf.math.pow(sig_true, 2) + small_num)), 2.) + small_num), 4.)
    #         term1 = tf.math.truediv(tf.math.truediv(tf.math.pow((mu_true - mu_pred), 2), tf.math.add(tf.math.pow(sig_true, 2), tf.math.pow(sig_pred, 2) + small_num)), 4.)
    #         return tf.reduce_mean((term0 + term1)) #, axis=-1)  # Note the `axis=-1`

    #     if C.args.loss == 'mse':
    #         C.model = Model([in_BBX5, in_FTM2, in_IMU19], [sl_rec_BBX5_0])

    #         # C.opt = keras.optimizers.Adam(learning_rate=C.learning_rate)
    #         # C.model.compile(loss=['mse'],
    #         #     loss_weights=[1, 1, 1], optimizer=C.opt,
    #         #     metrics= losses.mean_squared_error) # C.opt) # 'adam')
    #         C.model.compile(loss=['mse'],
    #             loss_weights=[1], optimizer='adam',
    #             metrics='mse') # C.opt) # 'adam')
    #         # C.model = Model([in_BBX5, in_FTM2, in_IMU19], \
    #         #     [sl_rec_BBX5_0, sl_rec_FTM2_1, sl_rec_IMU19_2, \
    #         #     cr_rec_FTM2_3, cr_rec_IMU19_4, cr_rec_BBX5_5, \
    #         #     cr_rec_IMU19_6, cr_rec_BBX5_7, cr_rec_FTM2_8, \
    #         #     fs_rec_BBX5_9, fs_rec_FTM2_10, fs_rec_IMU19_11, \
    #         #     ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU19_14, \
    #         #     ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU19_17, \
    #         #     ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU19_20, \
    #         #     crd_rec_FTM2_21, crd_rec_IMU19_22, crd_rec_BBX5_23, \
    #         #     mu_rec_BBX5_24, mu_rec_FTM2_25, mu_rec_IMU19_26, \
    #         #     mu_rec_BBX5_27, mu_rec_FTM2_28, mu_rec_IMU19_29, \
    #         #     mu_rec_BBX5_30, mu_rec_FTM2_31, mu_rec_IMU19_32])

    #         # # C.opt = keras.optimizers.Adam(learning_rate=C.learning_rate)
    #         # C.model.compile(loss=['mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
    #         #                         'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
    #         #                         'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
    #         #                         'mse', 'mse', 'mse',  'mse', 'mse', 'mse', ],
    #         #     loss_weights=[1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1, \
    #         #                     1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,
    #         #                     1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,
    #         #                     1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1], optimizer='adam') # C.opt) # 'adam')
    #     elif C.args.loss == 'b':
    #         C.model = Model([in_BBX5, in_FTM2, in_IMU19], \
    #             [sl_rec_BBX5_0, sl_rec_FTM2_1, sl_rec_IMU19_2, \
    #             cr_rec_FTM2_3, cr_rec_IMU19_4, cr_rec_BBX5_5, \
    #             cr_rec_IMU19_6, cr_rec_BBX5_7, cr_rec_FTM2_8, \
    #             fs_rec_BBX5_9, fs_rec_FTM2_10, fs_rec_IMU19_11, \
    #             ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU19_14, \
    #             ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU19_17, \
    #             ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU19_20, \
    #             crd_rec_FTM2_21, crd_rec_IMU19_22, crd_rec_BBX5_23, \
    #             mu_rec_BBX5_24, mu_rec_FTM2_25, mu_rec_IMU19_26, \
    #             mu_rec_BBX5_27, mu_rec_FTM2_28, mu_rec_IMU19_29, \
    #             mu_rec_BBX5_30, mu_rec_FTM2_31, mu_rec_IMU19_32, \
    #             sl_rec_FTM2_1, cr_rec_FTM2_3, cr_rec_FTM2_8, \
    #             fs_rec_FTM2_10, ota_rec_FTM2_13, ota_rec_FTM2_16, \
    #             ota_rec_FTM2_19, crd_rec_FTM2_21, mu_rec_FTM2_25, \
    #             mu_rec_FTM2_28, mu_rec_FTM2_31, fl_rec_FTM2_27])

    #         # C.opt = keras.optimizers.Adam(learning_rate=C.learning_rate)
    #         C.model.compile(loss=['mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
    #                                 'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
    #                                 'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
    #                                 'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
    #                                 Bhatt_loss, Bhatt_loss, Bhatt_loss,  Bhatt_loss, Bhatt_loss, Bhatt_loss, \
    #                                 Bhatt_loss, Bhatt_loss, Bhatt_loss,  Bhatt_loss, Bhatt_loss, Bhatt_loss],
    #             loss_weights=[1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1, \
    #                         1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1, \
    #                         1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1], optimizer='adam') # C.opt) # 'adam')
    #     # plot_model(BaseNet, show_shapes=True, to_file=str(C.model_id + '_base_net.png'))
    #     # plot_model(C.model, show_shapes=True, to_file=str(C.model_id + '.png'))
    #     return C.model