import torch, torchvision
import torch.nn as nn
from torch.nn import Module
import torchvision.transforms as transforms
import numpy as np
import model.vitag as model_module

class VisionPosition(Module): #
    def __init__(self, C):
        super().__init__()
        

        self.denoise_enc = Transformer_denoise_loc(C)

        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        if C.args.dataset == "jrdb":
            self.estimate_camera_intrinsic = Transformer_cam_intrin(C, out_dim=6)
            self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 )

            self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 16, out_dim=4 ) 
            self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 7, out_dim=4 )
        else:
            self.estimate_camera_intrinsic = Transformer_cam_intrin(C)
            self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 )
            # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

            self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 29, out_dim=4 ) 
            self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 6, out_dim=4 )
        
        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19, gen_cam_intr=False):
        B,L,P,_ = passersby_bbx4.shape
        cam_intr = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy

        # if res:
        # if True:
        #     denoised_gps = denoised_gps + interl_gps

        denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) # has ground truth of bbx observation,          should apply loss

        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


        denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

        denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1))

        # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

        # return pred_bbx, denoised_bbx_observation
        if gen_cam_intr:
            return pred_bbx, denoised_2nd_bbx_observation, cam_intr
        else:
            return pred_bbx, denoised_2nd_bbx_observation

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

class VisionPosition_wopredictor_1stage(Module):
    def __init__(self, C):
        super().__init__()
        self.estimate_camera_intrinsic = Transformer_cam_intrin(C, out_dim=6)

        self.denoise_enc = Transformer_denoise_loc(C)

        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

        self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 16, out_dim=4 ) 


        self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 7, out_dim=8 )
        


        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19, gen_cam_intr=False):
        B,L,P,_ = passersby_bbx4.shape
        cam_intr = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy

        # if res:
        # if True:
        #     denoised_gps = denoised_gps + interl_gps

        denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) # has ground truth of bbx observation,          should apply loss

        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


        denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

        denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4)

        # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

        # return pred_bbx, denoised_bbx_observation
        # if gen_cam_intr:
        #     return pred_bbx, denoised_2nd_bbx_observation, cam_intr
        # else:
        return denoised_3rd_bbx_observation, denoised_3rd_bbx_observation[:, :100, :]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class VisionPosition_wodenoise(Module):
    def __init__(self, C):
        super().__init__()
        self.estimate_camera_intrinsic = Transformer_cam_intrin(C, out_dim=6)

        # self.denoise_enc = Transformer_denoise_loc(C) # ablation1_w/o_denoise

        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

        self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 16, out_dim=4 ) 


        self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 7, out_dim=4  )


        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
        B,L,P,_ = passersby_bbx4.shape
        cam_intr = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy
        denoised_gps = interl_gps
        # if res:
        # if True:
        #     denoised_gps = denoised_gps + interl_gps

        denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) # has ground truth of bbx observation,          should apply loss

        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


        denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

        denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1))

        # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

        # return pred_bbx, denoised_bbx_observation
        return pred_bbx, denoised_2nd_bbx_observation

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class VisionPosition_wocamerintrin(Module): # ablation2_w/o_camera_intrinsic
    def __init__(self, C):
        super().__init__()
        # self.estimate_camera_intrinsic = Transformer_cam_intrin(C)

        self.denoise_enc = Transformer_denoise_loc(C)

        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8  ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

        self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 16, out_dim=4  ) 


        self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 7, out_dim=4 )


        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
        B,L,P,_ = passersby_bbx4.shape
        # cam_intr = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
        cam_intr = torch.ones(B,100,4,3).cuda()
        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy

        # if res:
        # if True:
        #     denoised_gps = denoised_gps + interl_gps

        denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) # has ground truth of bbx observation,          should apply loss

        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


        denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

        denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1))

        # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

        # return pred_bbx, denoised_bbx_observation
        return pred_bbx, denoised_2nd_bbx_observation

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

class VisionPosition_wocamproj(Module): # ablation2_w/o_cam_projection
    def __init__(self, C):
        super().__init__()
        self.estimate_camera_intrinsic = Transformer_cam_intrin(C, out_dim=6)

        self.denoise_enc = Transformer_denoise_loc(C)

        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
        # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

        self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 15, out_dim=4 ) 


        self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 7, out_dim=4 )


        self.init_weights(self)
    
    def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
        B,L,P,_ = passersby_bbx4.shape
        cam_intr = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
        # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
        denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy

        # if res:
        # if True:
        #     denoised_gps = denoised_gps + interl_gps

        # denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) # has ground truth of bbx observation,          should apply loss
        denoised_bbx_observation = cam_intr[:,:,0,:] + denoised_gps
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


        denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

        denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1))

        # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
        # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

        # return pred_bbx, denoised_bbx_observation
        return pred_bbx, denoised_2nd_bbx_observation

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

# class VisionPosition(Module): #current SOTA version
#     def __init__(self, C):
#         super().__init__()
#         self.estimate_camera_intrinsic = Transformer_cam_intrin(C)

#         self.denoise_enc = Transformer_denoise_loc(C)

#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
#         self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 ) 
#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

#         self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 29, out_dim=4 ) 


#         self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 6, out_dim=4 )


#         self.init_weights(self)
    
#     def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19, gen_cam_intr=False):
#         B,L,P,_ = passersby_bbx4.shape
#         cam_intr = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
#         # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
#         denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy

#         # if res:
#         # if True:
#         #     denoised_gps = denoised_gps + interl_gps

#         denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) # has ground truth of bbx observation,          should apply loss

#         # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


#         denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

#         denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1))

#         # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

#         # return pred_bbx, denoised_bbx_observation
#         if gen_cam_intr:
#             return pred_bbx, denoised_2nd_bbx_observation, cam_intr
#         else:
#             return pred_bbx, denoised_2nd_bbx_observation

#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)
#         if type(m) == nn.Conv2d:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)

# class VisionPosition(Module): #camera projection + bias
#     def __init__(self, C):
#         super().__init__()
#         self.estimate_camera_intrinsic = Transformer_cam_intrin(C)

#         self.denoise_enc = Transformer_denoise_loc(C)

#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
#         self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 ) 
#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

#         self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 29, out_dim=4 ) 


#         self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 6, out_dim=4 )


#         self.init_weights(self)
    
#     def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19, gen_cam_intr=False):
#         B,L,P,_ = passersby_bbx4.shape
#         cam_intr , bias = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
#         # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
#         denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy

#         # if res:
#         # if True:
#         #     denoised_gps = denoised_gps + interl_gps

#         denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) + bias # has ground truth of bbx observation,          should apply loss

#         # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


#         denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

#         denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1))

#         # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

#         # return pred_bbx, denoised_bbx_observation
#         if gen_cam_intr:
#             return pred_bbx, denoised_2nd_bbx_observation, cam_intr
#         else:
#             return pred_bbx, denoised_2nd_bbx_observation
        

#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)
#         if type(m) == nn.Conv2d:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)



# class VisionPosition(Module):
#     def __init__(self, C):
#         super().__init__()
#         self.estimate_camera_intrinsic = Transformer_cam_intrin(C)

#         self.denoise_enc = Transformer_denoise_loc(C)

#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 29, out_dim=8 ) 
#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 10, out_dim=8 ) 
#         self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 8, out_dim=8 ) 
#         # self.predictor = getattr(model_module, C.args.dec_model)(C, in_dim = 6, out_dim=8 ) 

#         self.denoise_2nd_enc =  getattr(model_module, "Transformer")(C, in_dim = 29, out_dim=4 ) 


#         self.denoise_3rd_enc = getattr(model_module, "Transformer")(C, in_dim = 6, out_dim=4 )


#         self.init_weights(self)
    
#     def forward(self, interl_gps, nearst_gps, in_imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19):
#         B,L,P,_ = passersby_bbx4.shape
#         cam_intr = self.estimate_camera_intrinsic(passersby_bbx4.view(B,L,-1), passersby_interl_gps, passersby_nearst_gps, imu19) #imu19
#         # denoised_gps = self.denoise_enc(interl_gps, nearst_gps, in_imu19) #imu19      # has no gt of gps, because gt is noisy
#         denoised_gps = self.denoise_enc(interl_gps, nearst_gps, imu19)      # has no gt of gps, because gt is noisy

#         # if res:
#         # if True:
#         #     denoised_gps = denoised_gps + interl_gps

#         denoised_bbx_observation = torch.matmul(cam_intr, denoised_gps.unsqueeze(-1)).squeeze(-1) # has ground truth of bbx observation,          should apply loss

#         # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1), None, None) # final output of bbx, should apply loss


#         denoised_2nd_bbx_observation = self.denoise_2nd_enc(torch.cat([denoised_bbx_observation, denoised_gps,  interl_gps, nearst_gps, in_imu19],dim=-1))

#         denoised_3rd_bbx_observation = self.denoise_3rd_enc(torch.cat([denoised_2nd_bbx_observation, denoised_gps],dim=-1))

#         # pred_bbx = self.predictor(torch.cat([denoised_2nd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         pred_bbx = self.predictor(torch.cat([denoised_3rd_bbx_observation,  interl_gps, nearst_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         # pred_bbx = self.predictor(torch.cat([denoised_bbx_observation, denoised_gps],dim=-1)).view(-1, 200, 4) # final output of bbx, should apply loss
#         # self.predictor(trans_bbx_observation, denoised_gps) # has ground truth of bbx, should apply loss

#         # return pred_bbx, denoised_bbx_observation
#         return pred_bbx, denoised_2nd_bbx_observation

#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)
#         if type(m) == nn.Conv2d:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)

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
        if C.args.dataset == "jrdb":
            self.input_process = nn.Linear(6, self.latent_dim)
            self.output_process = nn.Linear(self.latent_dim, 3)
        else:
            self.input_process = nn.Linear(4, self.latent_dim)
            self.output_process = nn.Linear(self.latent_dim, 2)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim ,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(self.seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        # self.output_process = nn.Linear(self.latent_dim, self.output_channels)
        
    
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
        return output #.view(-1, int(200*self.mask_ratio) , 2)


class Transformer_cam_intrin(Module):#256,4
    def __init__(self,C, input_feats=27,latent_dim=256,num_heads=4,ff_size=1024, dropout=0.1, num_layers=8, activation="gelu",out_dim =  8//2):
        super().__init__()
        self.mask_ratio = C.args.mask_ratio
        self.out_dim = out_dim
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
        self.jrdb = False
        if C.args.dataset == "jrdb":
            self.input_process = nn.Linear(23, self.latent_dim)
            self.jrdb = True
        else:
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
        self.output_process = nn.Linear(self.latent_dim, out_dim)
        self.output_process_bias = nn.Linear(self.latent_dim, out_dim//2)

        
    
    def forward(self, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, imu19, is_bias=False):
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
        bias = self.output_process_bias(x)
        if self.jrdb:
            return output.view(-1, int(200*self.mask_ratio), 4 , 3)
        elif is_bias:
            return output.view(-1, int(200*self.mask_ratio), self.out_dim , 2), bias.view(-1, int(200*self.mask_ratio), 4)
        else:
            return output.view(-1, int(200*self.mask_ratio), self.out_dim , 2)
        




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
