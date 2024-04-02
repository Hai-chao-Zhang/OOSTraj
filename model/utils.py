import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np


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
