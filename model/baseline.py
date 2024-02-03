import torch, torchvision
import torch.nn as nn
from torch.nn import Module
import torchvision.transforms as transforms
import numpy as np
import model.vitag as model_module
from kalman import KalmanFilter


class Baseline(Module):
    def __init__(self, C):
        super().__init__()
        if C.args.dataset == "jrdb":
            self.denoise_enc =  getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=4 ) 

            # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
            self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        else:
            self.denoise_enc =  getattr(model_module, C.args.dec_model)(C, in_dim = 4, out_dim=4 ) 

            # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
            self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 ) 

        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
        B,L,P,_ = passersby_bbx4.shape

        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        denoised_bbx_observation = self.denoise_enc(torch.cat([interl_gps, nearst_gps], dim=-1)).view(-1, 100, 4)      # has no gt of gps, because gt is noisy
        # denoised_bbx_observation = denoised_bbx_observation.view(B,L//2,4)
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss
        in_pred_ob = torch.cat([denoised_bbx_observation,  interl_gps, nearst_gps],dim=-1)
        pred_bbx = self.predictor(in_pred_ob) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss
        return pred_bbx.view(-1, 200, 4), denoised_bbx_observation

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

class Vanilla_Baseline(Module):
    def __init__(self, C):
        super().__init__()
        if C.args.dataset == "jrdb":
            self.denoise_enc_pred =  getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 
        else:
            self.denoise_enc_pred =  getattr(model_module, C.args.dec_model)(C, in_dim = 4, out_dim=8 ) 

        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 ) 

        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
        B,L,P,_ = passersby_bbx4.shape

        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        denoised_bbx_observation = self.denoise_enc_pred(torch.cat([interl_gps, nearst_gps], dim=-1)).view(-1, 200, 4)      # has no gt of gps, because gt is noisy
        # denoised_bbx_observation = denoised_bbx_observation.view(B,L//2,4)
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss
        # in_pred_ob = torch.cat([denoised_bbx_observation,  interl_gps, nearst_gps],dim=-1)
        # pred_bbx = self.predictor(in_pred_ob) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss
        return denoised_bbx_observation, denoised_bbx_observation[:,:100,:]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


# class Kalman_Baseline(Module):
#     def __init__(self, C):
#         super().__init__()
#         # if C.args.dataset == "jrdb":
#         #     self.denoise_enc_pred =  getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 
#         # else:
#         #     self.denoise_enc_pred =  np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])  # Example measurements
#             # getattr(model_module, C.args.dec_model)(C, in_dim = 4, out_dim=8 )     
#         # self.denoise_enc_pred = KalmanFilter


#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 ) 

#         # self.init_weights(self)
    
#     def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
#         B,L,P,_ = passersby_bbx4.shape

#         # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
#         denoised_bbx_observation=[]
#         future_bbx_observation=[]
#         interl_gps = interl_gps.numpy()
#         for i in range(B):
#             denoised_trajectory, future_predictions = torch.Tensor(KalmanFilter(interl_gps[i]))
#             denoised_bbx_observation.append (denoised_trajectory)
#             future_bbx_observation.append(future_predictions)
#         denoised_bbx_observation = torch.stack(denoised_bbx_observation, dim=0)
#         future_bbx_observation = torch.stack(future_bbx_observation, dim=0)
#         future_bbx_observation = torch.cat([denoised_bbx_observation, future_bbx_observation], dim=1)
#         # denoised_bbx_observation = KalmanFilter(interl_gps) .view(-1, 200, 4)      # has no gt of gps, because gt is noisy
#         # denoised_bbx_observation = denoised_bbx_observation.view(B,L//2,4)
#         # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss
#         # in_pred_ob = torch.cat([denoised_bbx_observation,  interl_gps, nearst_gps],dim=-1)
#         # pred_bbx = self.predictor(in_pred_ob) # final output of bbx, should apply loss
#         # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

#         fx=528.365
#         fy=527.925
#         cx=638.925
#         cy=359.2805
#         k1=-0.0394311
#         k2=0.00886432
#         k3=-0.00481956
#         p1=-0.000129881
#         p2=-4.88565e-05


#         u= cx + (x*fx) / z
#         v= cy + (y*fy) / z



#         return future_bbx_observation, denoised_bbx_observation

class Kalman_Baseline(Module):
    def __init__(self, C):
        super().__init__()
        if C.args.dataset == "jrdb":
            self.denoise_enc_pred =  getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 
        else:
            self.denoise_enc_pred =  getattr(model_module, C.args.dec_model)(C, in_dim = 4, out_dim=8 ) 

        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 ) 

        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
        B,L,P,_ = passersby_bbx4.shape

        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        future_kalman, denoised_kalman = self.kalman(interl_gps, B)
        denoised_kalman = denoised_kalman.cuda()
        denoised_bbx_observation = self.denoise_enc_pred(torch.cat([denoised_kalman, nearst_gps], dim=-1)).view(-1, 200, 4)      # has no gt of gps, because gt is noisy
        # denoised_bbx_observation = denoised_bbx_observation.view(B,L//2,4)
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss
        # in_pred_ob = torch.cat([denoised_bbx_observation,  interl_gps, nearst_gps],dim=-1)
        # pred_bbx = self.predictor(in_pred_ob) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss
        return denoised_bbx_observation, denoised_bbx_observation[:,:100,:]

    def kalman(self, interl_gps, B):
        denoised_bbx_observation=[]
        future_bbx_observation=[]
        interl_gps = interl_gps.cpu().numpy()
        for i in range(B):
            denoised_trajectory, future_predictions = torch.Tensor(KalmanFilter(interl_gps[i]))
            denoised_bbx_observation.append (denoised_trajectory)
            future_bbx_observation.append(future_predictions)
        denoised_bbx_observation = torch.stack(denoised_bbx_observation, dim=0)
        future_bbx_observation = torch.stack(future_bbx_observation, dim=0)
        future_bbx_observation = torch.cat([denoised_bbx_observation, future_bbx_observation], dim=1)
        return future_bbx_observation, denoised_bbx_observation

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
class Transformer_denoise_loc(Module):
    def __init__(self,C, input_feats=27,latent_dim=256,num_heads=4,ff_size=1024, dropout=0.1, num_layers=8, activation="gelu"):
        super().__init__()
        self.mask_ratio = C.args.mask_ratio

        C.IMU19_dim = C.IMU19_dim + 200 if C.args.dataset == "h3d" else  C.IMU19_dim
        self.latent_dim = latent_dim
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
        # self.input_process = nn.Linear(23, self.latent_dim)
        self.input_process = nn.Linear(4, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim ,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(self.seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        # self.output_process = nn.Linear(self.latent_dim, self.output_channels)
        self.output_process = nn.Linear(self.latent_dim, 2)
    
    def forward(self, interl_gps, nearst_gps, imu19):
        # bs, nframes, n_channels = x.shape  # should # frame, bs, njoints*nfeats
        # x = torch.cat([bbx_x,  imu_x], dim=2)


        # x = torch.cat([interl_gps, nearst_gps, imu19], dim=2)
        x = torch.cat([interl_gps, nearst_gps], dim=2)
        # x = bbx_x
        
        # x = x.permute((1,0,2))

        x = self.input_process(x)

        # for i in range(3):
            # t = torch.ones(x.shape[0:2]).unsqueeze(-1).cuda()
            # t[:]=i
        
        x = self.sequence_pos_encoder(x)  # [seqlen+1, bs, d]

        # x = torch.cat([t,x],-1)
        x = self.seqTransEncoder(x)#[1:] 

        output = self.output_process(x)
        return output.view(-1, int(200*self.mask_ratio) , 2)


class Transformer_cam_intrin(Module):#256,4
    def __init__(self,C, input_feats=27,latent_dim=256,num_heads=4,ff_size=1024, dropout=0.1, num_layers=8, activation="gelu"):
        super().__init__()
        self.mask_ratio = C.args.mask_ratio

        C.IMU19_dim = C.IMU19_dim + 200 if C.args.dataset == "h3d" else  C.IMU19_dim
        self.latent_dim = latent_dim
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
        self.input_process = nn.Linear(35, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim ,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(self.seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        # self.output_process = nn.Linear(self.latent_dim, self.output_channels)
        self.output_process = nn.Linear(self.latent_dim, 8//2)

        
    
    def forward(self, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
        # bs, nframes, n_channels = x.shape  # should # frame, bs, njoints*nfeats
        # x = torch.cat([bbx_x,  imu_x], dim=2)
        x = torch.cat([passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19], dim=2)
        # x = bbx_x
        
        # x = x.permute((1,0,2))

        x = self.input_process(x)

        # for i in range(3):
            # t = torch.ones(x.shape[0:2]).unsqueeze(-1).cuda()
            # t[:]=i
        
        x = self.sequence_pos_encoder(x)  # [seqlen+1, bs, d]

        # x = torch.cat([t,x],-1)
        x = self.seqTransEncoder(x)#[1:] 

        output = self.output_process(x)
        return output.view(-1, int(200*self.mask_ratio), 4, 2)




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
