import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import argparse, os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  
import torchmetrics
from config.config import Config
import pickle as pkl
from data.dataloader import Dataset_ViFi, load_data

from h3d.dataloader import Dataset_H3D, load_data_h3d

import random
import logging
from torchsummary import summary
from torch.optim.lr_scheduler import ExponentialLR

# from model.vitag import ViTag, LSTM, NMT
import model.vitag as model_module
seed = 42 # 2836 # 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class EarlyStopping():
    def __init__(self,logger,patience=7,verbose=False,delta=0):
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = 0

    def __call__(self,val_loss,model,path, epoch, phase):
        # self.logger("val_loss={}".format(val_loss))
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path, phase)
        elif score > self.best_score+self.delta:
            self.counter+=1
            # self.logger(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path, phase)
            self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self,val_loss,model,path, phase):
        if self.verbose:
            self.logger(
                f'Validation loss increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+f'phase_{phase}_model.pth')
        self.val_loss_min = val_loss

    def load_checkpoint(self, model, path, phase):
            model.load_state_dict(torch.load(path+'/'+f'phase_{phase}_model.pth'))



class Calculate():
    def __init__(self, model, device="cuda:0"):
        super(Calculate,self).__init__()

        self.device = device[0]
        pos = "noisy" if C.args.is_pos else ""
        self.ckpt_saving_path = os.path.join(f"checkpoints", f"{args.model}_{pos}")
        self.model = model.cuda()
        self.criterion = nn.MSELoss(reduction="mean").cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.learning_rate, weight_decay = 0.0005)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)# ExponentialLR(self.optimizer, gamma=0.99)

        # self.train_acc = torchmetrics.Accuracy().to(self.device)
        # self.test_acc = torchmetrics.Accuracy().to(self.device)
        self.train_acc = nn.MSELoss(reduction="mean").cuda()
        self.test_acc = nn.MSELoss(reduction="mean").cuda()

        self.train_loss = []
        self.valid_loss = []
        self.train_epochs_loss = []
        self.valid_epochs_loss = []

        
        self.log = ""        
        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if not os.path.exists(self.ckpt_saving_path):
            os.makedirs(self.ckpt_saving_path)
            
        logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(self.ckpt_saving_path, 'output.log'),
                        filemode='a',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.early_stopping = EarlyStopping(self.logger, patience=args.patience,verbose=False)

        isize = (40,6) if C.args.dataset == "h3d" else (50,5)
        self.ran_mask = RandomMaskingGenerator( input_size=isize) #input_size=(40,6)
        self.iou_d = IoU().cuda()


        

    # logger.info('This is a log info')
    # logger.debug('Debugging')
    # logger.warning('Warning exists')
    # logger.info('Finish')
     

    
    # def vifi_dataset(self, ):
    #     self.batch_size = 16
    #     self.trainset = torchvision.datasets.CIFAR10(root='/home/haichao/datasets/cifar/cifar10', train=True,
    #                                             download=True, transform=self.transform)
    #     self.train_dataloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
    #                                             shuffle=True, num_workers=4)
    #     self.testset = torchvision.datasets.CIFAR10(root='/home/haichao/datasets/cifar/cifar10', train=False,
    #                                         download=True, transform=self.transform)
    #     self.valid_dataloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
    #                                             shuffle=False, num_workers=2)
    def trans_matmul(self, tensor1, tensor2):
        # Create two tensors of the given shapes
        # tensor1 = torch.randn(128, 200, 2, 4)
        # tensor2 = torch.randn(128, 200, 2)
        # Reshape tensor2 to make it compatible for matrix multiplication
        # Reshape it to shape [128, 200, 2, 1]
        tensor2 = tensor2.unsqueeze(-1)

        # Transpose tensor1 to shape [128, 200, 4, 2] for matrix multiplication
        tensor1 = tensor1.permute(0, 1, 3, 2)

        # Perform matrix multiplication
        result = torch.matmul(tensor1, tensor2)

        # Transpose the result to obtain the desired shape [128, 200, 4]
        result = result.permute(0, 1, 3, 2)

        return result

    def train(self, config):
        # self.vifi_dataset(Dataset_ViFi)
        if C.args.dataset == "vifi":
            self.train_dataloader, self.valid_dataloader = load_data(config)
        elif C.args.dataset == "h3d":
            self.train_dataloader, self.valid_dataloader = load_data_h3d()
        else:
            raise ValueError("Dataset not supported")
        dsize = 6 if C.args.dataset == "h3d" else 5
        self.log = "Loop \t Train Loss \t Train Acc % \t Test Loss \t Test Acc % "
        self.logger.info(self.log)
        interl_gps, nearst_gps, imu19, bbx4, passersby_interl_gps, passersby_nearst_gps = next(iter(self.train_dataloader))
        # self.logger.info(summary(self.model, [list(traj.shape)[1:], list(ftm.shape)[1:], list(imu.shape)[1:]]))
        self.logger.info(self.model)

        # if config.args.phase !=1:
        #     self.early_stopping.load_checkpoint(self.model, path=self.ckpt_saving_path, phase=1)
        for epoch in range(args.epochs):
            self.model.train()
            train_epoch_loss = []
            for idx, (interl_gps, nearst_gps, imu, bbx, passersby_interl_gps, passersby_nearst_gps) in enumerate(self.train_dataloader): # interl_gps, nearst_gps, imu19, bbx4

                interl_gps = interl_gps.to(torch.float32).cuda()
                # noisy_traj = noisy_traj.to(torch.float32).cuda()
                nearst_gps = nearst_gps.to(torch.float32).cuda()
                imu = imu.to(torch.float32).cuda()
                bbx = bbx.type(torch.float32).cuda()

                passersby_interl_gps = passersby_interl_gps.to(torch.float32).cuda()
                passersby_nearst_gps = passersby_nearst_gps.to(torch.float32).cuda()


                # ratio = random.random() 

                # # ratio = 0.8 # ablation for w/o rms
                # mask = self.ran_mask.gen(batch_size=target.size(0), mask_ratio=ratio)
                # mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,dsize).cuda()
                # # mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,5).cuda()
                # mask = 1 - mask
                # # try:
                # # test_traj = target*mask # ~mask means visible
                # traj = target
                # traj[mask==0]=-1
                passersby_interl_gps = passersby_interl_gps.view(-1, 200, 4) # TODO: need update the dimension
                passersby_nearst_gps = passersby_nearst_gps.view(-1, 200, 4)

                pred_matrix = self.model(passersby_interl_gps, passersby_nearst_gps, imu)
                trans_bbx = self.trans_matmul(pred_matrix, interl_gps)
                # trans_bbx = pred_matrix * interl_gps #  interl_gps * pred_matrix
                outputs = trans_bbx.squeeze(2)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, bbx)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                # train_loss.append(loss.item())

                # self.train_acc(outputs, target)

                # if idx%(len(self.train_dataloader)//2)==0:
                #     self.logger("epoch={}/{},{}/{}of train, loss={}".format(
                #         epoch, args.epochs, idx, len(self.train_dataloader),loss.item()))
            self.train_epochs_loss.append(np.average(train_epoch_loss))

            total_train_acc = np.average(train_epoch_loss)
            # self.train_acc.reset()
            
            #=====================valid============================
            self.model.eval()
            valid_epoch_loss = []
            valid_iou_D_loss = []
            valid_iou_loss = []
            with torch.no_grad():
                for idx, (interl_gps, nearst_gps, imu, bbx, passersby_interl_gps, passersby_nearst_gps) in enumerate(self.train_dataloader): # interl_gps, nearst_gps, imu19, bbx4
                    interl_gps = interl_gps.to(torch.float32).cuda()
                    # noisy_traj = noisy_traj.to(torch.float32).cuda()
                    nearst_gps = nearst_gps.to(torch.float32).cuda()
                    imu = imu.to(torch.float32).cuda()
                    bbx = bbx.type(torch.float32).cuda()

                    passersby_interl_gps = passersby_interl_gps.to(torch.float32).cuda()
                    passersby_nearst_gps = passersby_nearst_gps.to(torch.float32).cuda()











                    # ratio = random.random() 
                    # # ratio = 0.8 # ablation for w/o rms
                    # mask = self.ran_mask.gen(batch_size=target.size(0), mask_ratio=ratio)
                    # mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,dsize).cuda()
                    # # mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,5).cuda()
                    # mask = 1 - mask
                    # # try:
                    # # test_traj = target*mask # ~mask means visible
                    # traj = target
                    # traj[mask==0]=-1


                    passersby_interl_gps = passersby_interl_gps.view(-1, 200, 4) # TODO: need update the dimension
                    passersby_nearst_gps = passersby_nearst_gps.view(-1, 200, 4)

                    pred_matrix = self.model(passersby_interl_gps, passersby_nearst_gps, imu)
                    trans_bbx = self.trans_matmul(pred_matrix, interl_gps)
                    # trans_bbx = pred_matrix * interl_gps #  interl_gps * pred_matrix
                    outputs = trans_bbx.squeeze(2)










                    # outputs = self.model(interl_gps, nearst_gps, imu)
                    # self.optimizer.zero_grad()
                    loss = self.criterion(outputs, bbx)
                    # loss.backward()
                    # self.optimizer.step()
                    valid_epoch_loss.append(loss.item())
                    # self.valid_loss.append(loss.item())

                    with open(f'vis/vis_{C.args.model}.pkl','wb') as f:
                        pkl.dump([outputs, bbx],f)


                    iou_loss, iou_d  = self.iou_d(outputs, bbx)
                    valid_iou_loss.append(torch.mean(iou_loss).item())
                    valid_iou_D_loss.append(torch.mean(iou_d).item())

                    # self.test_acc(outputs, labels)

            # self.valid_epochs_loss.append(np.average(valid_epoch_loss))
            total_test_acc = np.average(valid_epoch_loss)

            valid_iou_loss_acc = np.average(valid_iou_loss)
            valid_iou_D_loss_acc = np.average(valid_iou_D_loss)

            # total_test_acc = self.test_acc.compute()
            # self.test_acc.reset()

            
            self.log = f"{epoch}/{args.epochs} Phase_{config.args.phase} \t {np.average(train_epoch_loss):.8f} \t {total_train_acc:.8f} \t {np.average(valid_epoch_loss):.8f} \t {total_test_acc*2048:.8f} \t iou_d {valid_iou_D_loss_acc:.8f}  \t iou {valid_iou_loss_acc:.8f} "
            self.logger.info(self.log)

            #==================early stopping======================
            self.early_stopping(val_loss=total_test_acc, model=self.model,path=self.ckpt_saving_path,epoch=epoch, phase = config.args.phase) # (self.valid_epochs_loss[-1], model=self.model,path=r'./model')
            if self.early_stopping.early_stop:
                # self.logger("Early stopping")
                self.logger.info(f" Early stopping: best score is {self.early_stopping.best_score} best epoch is {self.early_stopping.best_epoch} ")
                break
            #====================adjust lr========================
            # lr_adjust = {
            #         200: 5e-2, 400: 1e-3, 600: 5e-4, 800: 1e-5,
            #         1000: 5e-6, 1500: 1e-7, 2000: 5e-8
            #     }
            # if epoch in lr_adjust.keys():
            #     lr = lr_adjust[epoch]
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = lr
            # if epoch>50:
            self.scheduler.step()
            self.logger.info('Updating learning rate to {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.logger.debug(f"best score is {self.early_stopping.best_score} best epoch is {self.early_stopping.best_epoch} ")
    
    def test_vis(self, config):
        # self.vifi_dataset(Dataset_ViFi)
        if C.args.dataset == "vifi":
            self.train_dataloader, self.valid_dataloader = load_data(config)
        elif C.args.dataset == "h3d":
            self.train_dataloader, self.valid_dataloader = load_data_h3d()
        else:
            raise ValueError("Dataset not supported")
        dsize = 6 if C.args.dataset == "h3d" else 5
        self.log = "Loop \t Train Loss \t Train Acc % \t Test Loss \t Test Acc % "
        self.logger.info(self.log)
        traj, ftm, imu, target = next(iter(self.train_dataloader))
        # self.logger.info(summary(self.model, [list(traj.shape)[1:], list(ftm.shape)[1:], list(imu.shape)[1:]]))
        self.logger.info(self.model)

        # if config.args.phase !=1:
        self.early_stopping.load_checkpoint(self.model.module, path=self.ckpt_saving_path, phase=2)

        #=====================valid============================
        self.model.eval()
        valid_epoch_loss = []
        valid_iou_D_loss = []
        valid_iou_loss = []
        with torch.no_grad():
            for idx, (traj, ftm, imu, target)  in enumerate(self.valid_dataloader):
                traj = traj.to(torch.float32).cuda()
                # noisy_traj = noisy_traj.to(torch.float32).cuda()
                ftm = ftm.to(torch.float32).cuda()
                imu = imu.to(torch.float32).cuda()
                target = target.type(torch.float32).cuda()

                ratio = random.random() 
                # ratio = 0.8 # ablation for w/o rms
                mask = self.ran_mask.gen(batch_size=target.size(0), mask_ratio=ratio)
                mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,dsize).cuda()
                # mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,5).cuda()
                mask = 1 - mask
                # try:
                # test_traj = target*mask # ~mask means visible
                traj = target
                traj[mask==0]=-1

                outputs = self.model(traj, ftm, imu)

                with open(f'vis/vis_{C.args.model}.pkl','wb') as f:
                    pkl.dump([outputs, target],f)

                loss = self.criterion(outputs, target)
                valid_epoch_loss.append(loss.item())
                # self.valid_loss.append(loss.item())


                iou_loss, iou_d  = self.iou_d(outputs, target)
                valid_iou_loss.append(torch.mean(iou_loss).item())
                valid_iou_D_loss.append(torch.mean(iou_d).item())

                # self.test_acc(outputs, labels)

            # self.valid_epochs_loss.append(np.average(valid_epoch_loss))
            total_test_acc = np.average(valid_epoch_loss)

            valid_iou_loss_acc = np.average(valid_iou_loss)
            valid_iou_D_loss_acc = np.average(valid_iou_D_loss)

            # total_test_acc = self.test_acc.compute()
            # self.test_acc.reset()

            self.log = f"{idx} \t {np.average(valid_epoch_loss):.4f} \t {valid_iou_loss_acc:.4f} \t {valid_iou_D_loss_acc:.4f} \t {total_test_acc:.4f} "
            self.logger.info(self.log)
        
    def test(self, imgpath):
        self.log=""

        image = Image.open(imgpath)
        image = self.transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(torch.float32).to(self.device)

        self.early_stopping.load_checkpoint(self.model, path=self.ckpt_saving_path, phase=1)
        self.model.eval()
        pred = self.model(image)
        _ ,pred=torch.max(pred,1)
        self.logger(f"prediction results: {self.classes[pred[0]]} ")
        # self.logger(f"prediction results: {[self.classes[pred[j]] for j in range(pred.shape[0])]}")

    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        # self.ran_mask = RandomMaskingGenerator()

        # ratio = random.random() 
        # # ratio = 0.8 # ablation for w/o rms
        # mask = self.ran_mask.gen(batch_size=target.size(0), mask_ratio=ratio)
        # mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,6).cuda()
        # # mask = torch.Tensor(mask).unsqueeze(-1).repeat(1,1,5).cuda()
        # mask = 1 - mask
        # # try:
        # # test_traj = target*mask # ~mask means visible
        # test_traj = target
        # test_traj[mask==0]=-1

class RandomMaskingGenerator(object):
    """
    input_size: 传入的为window_size=input_size//patch_size, 即224/16=14
    mask_ratio: mask的比例, 默认为0.75
    """
    def __init__(self, input_size=(40,6), mask_ratio=0.75):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.length, self.dim = input_size
        self.num_patches = self.length #* self.dim  # patch的总数即196
        self.num_mask = int(mask_ratio * self.num_patches)  # 196 * 0.75

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def gen(self, batch_size, mask_ratio=None):
        masks=[]
        for i in range(batch_size):
            if type (mask_ratio) is list:
                self.num_mask = int(mask_ratio[i] * self.num_patches)
            else:
                self.num_mask = int(mask_ratio * self.num_patches)
            
            mask = np.hstack([  # 水平方向叠起来
                np.zeros( (self.num_patches - self.num_mask)),  # 25%为0
                np.ones( self.num_mask),  # mask的部分设为1
            ])
            np.random.shuffle(mask)
            masks.append(mask)
        mask = np.array(masks)
        mask = mask.astype(np.int)
        mask = mask.reshape(batch_size, -1)  # [batch_size, 196]
        return mask # [196]

class IoU(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, a, b):
        """
        Calculate the intersection over union (IoU) between two sets of bounding boxes.
        a: a PyTorch tensor of shape [batch_size, num_boxes_a, 4] representing the ground truth bounding boxes,
        where the last dimension is [x1, y1, x2, y2]
        b: a PyTorch tensor of shape [batch_size, num_boxes_b, 4] representing the predicted bounding boxes,
        where the last dimension is [x1, y1, x2, y2]

                    # print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
                    # print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
                    # print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
                    # print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
                    # print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height 
        """
        # # calculate the minimum value across batch, length, and bounding box dimensions
        # min_val = torch.min(torch.min(a[...,:2]), torch.min(b[...,:2]))
        # # add the absolute minimum value to both a and b
        # a[...,:0] = a[...,:0] - min_val
        # b[...,:0] = b[...,:0] - min_val
        # a[...,:1] = a[...,:1] - min_val
        # b[...,:1] = b[...,:1] - min_val

        # depth = torch.clamp(1- torch.abs(a[..., 2] - b[..., 2])/torch.max(torch.abs(a[..., 2]), torch.abs(b[..., 2]) ),min=0.1, max=1)

        # # Get coordinates of intersection rectangles
        # x1 = torch.max(a[..., 0], b[..., 0])
        # y1 = torch.max(a[..., 1], b[..., 1])
        # x2 = torch.min(a[..., 0] + a[..., 3], b[..., 0] + b[..., 3])
        # y2 = torch.min(a[..., 1] + a[..., 4], b[..., 1] + b[..., 4])
        # # Compute intersection area
        # inter_area = torch.clamp((x2 - x1) * (y2 - y1), min=0, max=1)
        # # Compute union area
        # union_area = a[..., 3] * a[..., 4] + b[..., 3] * b[..., 4] - inter_area
        # # Compute IoU
        # iou = inter_area / union_area
        # iou = torch.clamp(iou, min=0.1, max=1.0)

        # torch.nan_to_num(iou, nan=0, posinf=0, neginf=0)
        # torch.nan_to_num(depth, nan=0, posinf=0, neginf=0)
        # return iou, iou * depth
        # calculate the minimum value across batch, length, and bounding box dimensions
        min_val = torch.min(torch.min(a[...,:3]), torch.min(b[...,:3]))
        # add the absolute minimum value to both a and b
        a[...,:3] = a[...,:3] - min_val
        b[...,:3] = b[...,:3] - min_val


        depth = torch.clamp(1- torch.abs(a[..., 2] - b[..., 2])/torch.abs(torch.max(a[..., 2], b[..., 2] )),min=0.1)

        # Get coordinates of intersection rectangles
        x1 = torch.max(a[..., 0], b[..., 0])
        y1 = torch.max(a[..., 1], b[..., 1])
        x2 = torch.min(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
        y2 = torch.min(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
        # Compute intersection area
        inter_area = torch.clamp((x2 - x1) * (y2 - y1), min=0)
        # Compute union area
        union_area = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter_area
        # Compute IoU
        iou = inter_area / union_area
        iou = torch.clamp(iou, min=0.1, max=1.0)

        torch.nan_to_num(iou, nan=0, posinf=1, neginf=0)
        torch.nan_to_num(depth, nan=0, posinf=1, neginf=0)
        return iou, iou * depth

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Toyota Project')
    parser.add_argument("mode", nargs="?", type=str, default= None, help="test an image, input image path")
    parser.add_argument("testpath", nargs="?", type=str, default= None, help="test an image, input image path")
    parser.add_argument("--predict", nargs="?", type=str, default= None, help="test additional images")
    parser.add_argument("--device", nargs="?", type=str, default= [torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),], help="gpu device")
    parser.add_argument("--learning_rate", nargs="?", type=float, default= '0.0001', help="learning_rate")
    parser.add_argument("--patience", nargs="?", type=int, default= 400, help="patience to stop")
    parser.add_argument("--epochs", nargs="?", type=int, default= 2000, help="total epochs to train") 
    parser.add_argument("--input_size", nargs="?", type=int, default= [32,32], help="input img size") 
    
    parser.add_argument('-tsid_idx', '--test_seq_id_idx', type=int, default='0', help='0-14') # edit
    parser.add_argument('-k', '--recent_K', type=int, default=50, help='Window length') # edit
    parser.add_argument('-l', '--loss', type=str, default='mse', help='mse: Mean Squared Error | b: Bhattacharyya Loss')
    parser.add_argument('-rt', '--resume_training', action='store_true', help='resume from checkpoint')
    parser.add_argument("--is_proc_data", action='store_true', help="if ture, process dataset")
    parser.add_argument('--random', action='store_true', help='random start')

    parser.add_argument("--phase", nargs="?", type=int, default= 1, choices=[1,2,3], help="Training Phase of our model, the flag is the masking strategies ")
    parser.add_argument("--gpus", nargs="?", type=str, default= "0,1", help="input img size e.g, 0,1")
    parser.add_argument("--is_pos", action='store_true', help="if ture calculate imu trajectory")

    parser.add_argument("--model", nargs="?", type=str, default= "Transformer",choices=["ViTag", "LSTM", "NMT", "UNet", "Transformer","SingleLayerViTag"], help="input img size") 
    parser.add_argument("--dataset", nargs="?", type=str, default= "vifi",choices=["vifi", "h3d"], help="input img size") 

    args = parser.parse_args()
    args.gpus = [int(gpu) for gpu in args.gpus.split(",")]
    C = Config(args)
    if len(args.gpus) == 1:
        torch.cuda.set_device(args.gpus[0])
    model =  getattr(model_module, args.model)(C)
    if len(args.gpus)>1:
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
        # model = model.module    
    
    cal = Calculate(model, args.device)
    
    if args.mode =="train":
        cal.train(config=C)
        if C.args.phase == 1:
            C.args.phase = 2
            cal = Calculate(model, args.device)
            cal.train(config=C)

    elif args.mode == "test":
        cal.test_vis(config=C)
    elif args.mode == "predict":
        cal.test(args.testpath)
    
