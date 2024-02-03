import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse, os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  
import torchmetrics
from config.config import Config

from data.dataloader import Dataset_ViFi, load_data
import random
import logging
from torchsummary import summary
from torch.optim.lr_scheduler import ExponentialLR
from model.dif.diffusion.resample import create_named_schedule_sampler
# from model.vitag import ViTag, LSTM, NMT
import model.vitag as model_module
seed = 2836 # 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import model.dif.parser_util as parser_util
from model.dif.mdm import MDM
from model.dif.diffusion import gaussian_diffusion as gd
from model.dif.model_util import SpacedDiffusion, space_timesteps
from model.dif.model_util import create_gaussian_diffusion, get_model_args

from model.dif.parser_util import train_args

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
    def __init__(self, device="cuda:0"):
        super(Calculate,self).__init__()

        self.model, self.diffusion = self.create_model_and_diffusion(args) # , data)
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        
        self.ddp_model = self.model.cuda()
        if len(args.gpus)>1:
            self.ddp_model = torch.nn.DataParallel(self.ddp_model , device_ids=args.gpus)


        self.device = device[0]
        pos = "noisy" if C.args.is_pos else ""
        self.ckpt_saving_path = os.path.join(f"checkpoints", f"{args.model}_{pos}")
        # self.model = model.cuda()
        self.criterion = nn.MSELoss(reduction="mean").cuda()
        # self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.learning_rate, weight_decay = 0.0005)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=args.learning_rate, weight_decay = 0.0005)

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
    def create_model_and_diffusion(self, args):#, data):
        model = MDM(**get_model_args(args))#, data))
        diffusion = create_gaussian_diffusion(args)
        return model, diffusion
    
    def create_incompelete_seq(self, t, whole_seq):
        # target TODO: rewrite'
        N = whole_seq.shape[1]
        mask = torch.ones_like(whole_seq)
        mask_out = torch.ones_like(whole_seq[:,:,0]) # 

        for idx, time in enumerate(t):
            time = self.diffusion.num_timesteps - time
            if time >= N:
                continue
            mask[idx,time:,:] = 0
            mask_out[idx,time:] = 0
            mask_out[idx,time] = -1
        masked_seq = torch.mul(whole_seq, mask)
        return masked_seq, mask_out

    def train(self, config):
        # self.vifi_dataset(Dataset_ViFi)
        self.train_dataloader, self.valid_dataloader = load_data(config)

        self.log = "Loop \t Train Loss \t Train Acc % \t Test Loss \t Test Acc % "
        self.logger.info(self.log)
        test_traj, ftm, imu, target = next(iter(self.train_dataloader))
        # self.logger.info(summary(self.model, [list(target.shape)[1:], list(ftm.shape)[1:], list(imu.shape)[1:]]))
        # TODO: uncomment
        self.logger.info(self.model)

        # if config.args.phase !=1:
        #     self.early_stopping.load_checkpoint(self.model, path=self.ckpt_saving_path, phase=1)
        self.logger.info("Loop \t Train Loss \t Train Diff  \t Train Pred \t Test Loss \t Test Diff \t Test Pred  ")
        for epoch in range(args.epochs):
            self.model.train()
            train_epoch_loss = []
            train_diff_epoch_loss = []
            train_pred_epoch_loss = []
            for idx, (test_traj, ftm, imu, target) in enumerate(self.train_dataloader):

                test_traj = test_traj.to(torch.float32).cuda()
                # noisy_traj = noisy_traj.to(torch.float32).cuda()
                ftm = ftm.to(torch.float32).cuda()
                imu = imu.to(torch.float32).cuda()
                target = target.type(torch.float32).cuda()

                t, weights = self.schedule_sampler.sample_after_N(target.shape[0], target.shape[1], "cuda")
                # t, weights = self.schedule_sampler.sample(target.shape[0], "cuda")   
                         
                # compute_losses = functools.partial(
                #         self.diffusion.training_losses,
                #         self.ddp_model,
                #         micro,  # [bs, ch, image_size, image_size]
                #         t,  # [bs](int) sampled timesteps
                #         model_kwargs=micro_cond,
                #         dataset=self.data.dataset
                #         )
                incomplete_seq, mask = self.create_incompelete_seq(t, target)

                noise_traj_context = torch.cat([ftm, imu], dim=2)
                # since the noisy sequence has already complete timestamps and divided in time dimension,
                # we don't mask them here, we align them in timestamps and pass them to transformer to provide time information,
                # we put them to 
                outputs, diff_loss = self.diffusion.training_losses( self.ddp_model,
                        incomplete_seq,  # [bs, ch, image_size, image_size]
                        t,  # [bs](int) sampled timesteps
                        noise_traj_context, mask=mask)
                if -1 in mask:
                    pred_loss = F.mse_loss(outputs[mask==-1,:], target[mask==-1,:], reduction='mean')
                    loss = pred_loss + diff_loss
                else:
                    with torch.no_grad():
                        pred_loss = torch.zeros_like(diff_loss)
                    loss = diff_loss

                # , dataset=self.data.dataset)
                # TODO: model inputs are t, m_t, cond, in_traj, n_traj

                # if last_batch or not self.use_ddp:
                #     losses = compute_losses()
                # else:
                #     with self.ddp_model.no_sync():
                #         losses = compute_losses()

                # if isinstance(self.schedule_sampler, LossAwareSampler):
                #     self.schedule_sampler.update_with_local_losses(
                #         t, losses["loss"].detach()
                #     )

                # loss = (losses["loss"] * weights).mean()
                # log_loss_dict(
                #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
                # )

                # outputs = self.model(traj, ftm, imu)
                self.optimizer.zero_grad()
                # loss = mse_loss # self.criterion(outputs, target)
                if torch.isnan(loss).any():
                    raise Exception("NaN :(")
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                train_diff_epoch_loss.append(diff_loss.item())
                train_pred_epoch_loss.append(pred_loss.item())
                # train_loss.append(loss.item())

                # self.train_acc(outputs, target)

                # if idx%(len(self.train_dataloader)//2)==0:
                #     self.logger("epoch={}/{},{}/{}of train, loss={}".format(
                #         epoch, args.epochs, idx, len(self.train_dataloader),loss.item()))
            self.train_epochs_loss.append(np.average(train_epoch_loss))

            total_train_acc = np.average(train_epoch_loss)
            total_diff_train_acc = np.average(train_diff_epoch_loss)
            total_pred_train_acc = np.average(train_pred_epoch_loss)
            # self.train_acc.reset()
            
            #=====================valid============================
            self.model.eval()
            valid_epoch_loss = []
            valid_diff_epoch_loss = []
            valid_pred_epoch_loss = []
            
            for idx, (test_traj, ftm, imu, target) in enumerate(self.valid_dataloader):

                test_traj = test_traj.to(torch.float32).cuda()
                # noisy_traj = noisy_traj.to(torch.float32).cuda()
                ftm = ftm.to(torch.float32).cuda()
                imu = imu.to(torch.float32).cuda()
                target = target.type(torch.float32).cuda()

                t, weights = self.schedule_sampler.sample_after_N(target.shape[0], target.shape[1], "cuda")
                incomplete_seq, mask = self.create_incompelete_seq(t, target)
                noise_traj_context = torch.cat([ftm, imu], dim=2)

                outputs, diff_loss = self.diffusion.training_losses( self.ddp_model,
                        incomplete_seq,  # [bs, ch, image_size, image_size]
                        t,  # [bs](int) sampled timesteps
                        noise_traj_context, mask=mask)
                if -1 in mask:
                    pred_loss = F.mse_loss(outputs[mask==-1,:], target[mask==-1,:], reduction='mean')
                    loss = pred_loss + diff_loss
                else:
                    with torch.no_grad():
                        pred_loss = torch.zeros_like(diff_loss)
                    loss = diff_loss
                
                valid_epoch_loss.append(loss.item())
                valid_diff_epoch_loss.append(diff_loss.item())
                valid_pred_epoch_loss.append(pred_loss.item())

                self.valid_loss.append(loss.item())

                # self.test_acc(outputs, labels)

            self.valid_epochs_loss.append(np.average(valid_epoch_loss))

            total_test_acc = np.average(valid_epoch_loss)
            total_diff_test_acc = np.average(valid_diff_epoch_loss)
            total_pred_test_acc = np.average(valid_pred_epoch_loss)
            # total_test_acc = self.test_acc.compute()
            # self.test_acc.reset()

            
            self.log = f"{epoch}/{args.epochs} Phase_{config.args.phase} \t {total_train_acc:.8f} \t {total_diff_train_acc:.8f} \t {total_pred_train_acc:.8f}  \t {total_test_acc:.8f} \t {total_diff_test_acc:.8f}  \t {total_pred_test_acc:.8f}"
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Toyota Project')
    parser.add_argument("mode", nargs="?", type=str, default= None, help="test an image, input image path")
    parser.add_argument("testpath", nargs="?", type=str, default= None, help="test an image, input image path")
    parser.add_argument("--predict", nargs="?", type=str, default= None, help="test additional images")
    parser.add_argument("--device", nargs="?", type=str, default= [torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),], help="gpu device")
    parser.add_argument("--learning_rate", nargs="?", type=float, default= '1e-4', help="learning_rate")
    parser.add_argument("--patience", nargs="?", type=int, default= 400, help="patience to stop")
    parser.add_argument("--epochs", nargs="?", type=int, default= 2000, help="total epochs to train") 
    parser.add_argument("--input_size", nargs="?", type=int, default= [32,32], help="input img size") 
    
    parser.add_argument('-tsid_idx', '--test_seq_id_idx', type=int, default='0', help='0-14') # edit
    parser.add_argument('-k', '--recent_K', type=int, default=50, help='Window length') # edit
    parser.add_argument('-l', '--loss', type=str, default='mse', help='mse: Mean Squared Error | b: Bhattacharyya Loss')
    parser.add_argument('-rt', '--resume_training', action='store_true', help='resume from checkpoint')
    parser.add_argument("--is_proc_data", action='store_true', help="if ture, process dataset")
    parser.add_argument('--random', action='store_true', help='random start')

    parser.add_argument("--phase", nargs="?", type=int, default= 2, choices=[1,2,3], help="Training Phase of our model, the flag is the masking strategies ")
    parser.add_argument("--gpus", nargs="?", type=str, default= "0", help="input img size e.g, 0,1")
    parser.add_argument("--is_pos", action='store_true', help="if ture calculate imu trajectory")

    parser.add_argument("--model", nargs="?", type=str, default= "Diffusion",choices=["Diffusion"], help="input img size") 
    parser_util.add_base_options(parser)
    parser_util.add_data_options(parser)
    parser_util.add_model_options(parser)
    parser_util.add_diffusion_options(parser)
    parser_util.add_training_options(parser)

    args = parser.parse_args()
    args.gpus = [int(gpu) for gpu in args.gpus.split(",")]
    C = Config(args)
    if len(args.gpus) == 1:
        torch.cuda.set_device(args.gpus[0])
    # model =  getattr(model_module, args.model)(C)
    # if len(args.gpus)>1:
    #     model = torch.nn.DataParallel(model, device_ids=args.gpus)
        # model = model.module    
    
    cal = Calculate(args.device)
    
    if args.mode =="train":
        cal.train(config=C)
        if C.args.phase == 1:
            C.args.phase = 2
            cal = Calculate(model, args.device)
            cal.train(config=C)

    elif args.mode == "test":
        cal.test(args.testpath)
    elif args.mode == "predict":
        cal.test(args.testpath)
    
