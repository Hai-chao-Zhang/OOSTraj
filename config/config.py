import argparse, os
import glob

class Config:
    def __init__(self, args):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        # self.parser = argparse.ArgumentParser()
        self.args = args # self.parser.parse_args()

        self.root_path = './' # '../../../..'

        # ------------------------------------------
        #  To be updated in prepare_training_data()
        self.model_id = 'X22_indoor_BBX5in_IMU19_FTM2_' # edit
        if self.args.loss == 'mse':self.model_id += 'test_idx_' + str(self.args.test_seq_id_idx)
        elif self.args.loss == 'b': self.model_id = self.model_id[:self.model_id.index('FTM2_') + len('FTM_2')] + \
            'Bloss_test_idx_' + str(self.args.test_seq_id_idx)
        print('self.model_id: ', self.model_id)
        # self.seq_root_path = self.root_path + '/Data/datasets/RAN/seqs/indoor'
        # self.seq_root_path_for_model = self.root_path + '/RAN4model/seqs/scene0'
        self.seq4model_root_path = self.root_path + 'datasets/RAN4model/seqs/scene0'
        if not os.path.exists(self.seq4model_root_path):
            os.makedirs(self.seq4model_root_path)

        print('self.seq4model_root_path: ', self.seq4model_root_path)
        self.seq_id_path_ls = sorted(glob.glob(self.seq4model_root_path + '/*'))
        self.seq_id_ls = sorted([seq_id_path[-15:] for seq_id_path in self.seq_id_path_ls])
        self.seq_id = self.seq_id_ls[0]
        self.test_seq_id = self.seq_id_ls[self.args.test_seq_id_idx]

        # -------
        #  Color
        # -------
        self.color_ls = ['crimson', 'lime green', 'royal blue', 'chocolate', 'purple', 'lemon']
        self.color_dict = {
            'crimson': (60,20,220),
            'lime green': (50,205,50),
            'royal blue': (225,105,65),
            'chocolate': (30,105,210),
            'purple': (128,0,128),
            'lemon': (0,247,255)
        }

        # --------------------------------------
        #  To be updated in update_parameters()
        self.seq_path = self.seq_id_path_ls[0]
        print(); print() # debug
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        print('self,seq_id_ls: ', self.seq_id_ls)

        self.img_type = 'RGBh_ts16_dfv2'
        self.img_path = self.seq_path + '/' + self.img_type
        self.RGBh_ts16_dfv3_ls = []

        self.subjects = [15, 46, 77, 70, 73]
        self.phone_time_offsets = [0] * len(self.subjects)

        print(); print() # debug
        print('self.subjects: ', self.subjects)

        # ----------------------------------------------
        #  Synchronized data: BBXC3,BBX5,IMU,_sync_dfv3
        # ----------------------------------------------
        self.BBXC3_sync_dfv3, self.BBX5_sync_dfv3, self.IMU19_sync_dfv3, self.FTM2_sync_dfv3= [], [], [], []
        self.seq_id = self.seq_id_ls[0]
        self.seq_path_for_model = self.seq4model_root_path + '/' + self.seq_id
        self.sync_dfv3_path = self.seq_path_for_model + '/sync_ts16_dfv3'
        if not os.path.exists(self.sync_dfv3_path): os.makedirs(self.sync_dfv3_path)
        self.BBXC3_sync_dfv3_path = self.sync_dfv3_path + '/BBXC3H_sync_dfv3.pkl'
        self.BBX5_sync_dfv3_path = self.sync_dfv3_path + '/BBX5H_sync_dfv3.pkl'
        self.IMU19_sync_dfv3_path = self.sync_dfv3_path + '/IMU19_sync_dfv3.pkl'
        self.FTM2_sync_dfv3_path = self.sync_dfv3_path + '/FTM2_sync_dfv3.pkl'

        # ------
        #  BBX5
        # ------
        self.BBX5_dim = 4
        self.BBX5_dummy = [0] * self.BBX5_dim

        # ------------
        #  FTM2
        # ------------
        self.FTM2_dim = 2
        self.FTM2_dummy = [0] * self.FTM2_dim

        # -----
        #  IMU
        # -----
        self.IMU_path = self.seq_path + '/IMU'
        self.IMU_dfv3_path = self.seq_path + '/IMU_dfv3' # ts13_dfv3 with offsets (os)
        if not os.path.exists(self.IMU_dfv3_path): os.makedirs(self.IMU_dfv3_path)

        # -------
        #  IMU19
        # -------
        self.IMU19_data_types = ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
        self.IMU19_dim =  (3 + 3 + 3 + 4 + 3 + 3 )# if self.args.is_pos (3 + 3 + 3 + 4 )
        self.IMU19_dummy = [0] * self.IMU19_dim

        # --------------
        #  Video Window
        # --------------
        self.crr_ts16_dfv3_ls_all_i = 0
        self.video_len = 0 # len(self.ts12_BBX5_all)
        self.recent_K = self.args.recent_K
        self.n_wins = 0

        self.sub_tracklets = None # (win_i, subj_i, first_f_i_in_win_in_view, last_f_i_in_win_in_view) with len <= K

        # -------
        #  Model
        # -------
        self.n_batch = 32 # 32
        self.n_epochs = 200 # 1000000000000000 # 200 # 100000 # 100000
        self.h_BBX5_dim = 32 # X8: 32
        self.h_FTM2_dim = 32
        self.h_IMU19_dim = 32 # X8: 32
        self.h_fused_dim = 32 # X8: 32
        self.n_filters = 32 # X8: 32
        self.kernel_size = 16 # X8: 16
        self.seq_in_BBX5, self.seq_in_IMU19, self.seq_in_FTM2 = None, None, None
        self.seq_out_BBX5, self.seq_out_IMU19, self.seq_out_FTM2 = None, None, None
        self.model = None
        self.checkpoint_root_path = self.root_path + '/Data/checkpoints/' + self.model_id # exp_id
        if not os.path.exists(self.checkpoint_root_path): os.makedirs(self.checkpoint_root_path)
        self.model_path_to_save = self.checkpoint_root_path + '/model.h5'
        self.model_weights_path_to_save = self.checkpoint_root_path + '/w.ckpt'
        self.start_training_time = ''
        self.start_training_time_ckpt_path = ''
        self.history_callback_path_to_save = self.checkpoint_root_path + '/history_callback.p' # self.seq_path + '/' + self.model_id + '_history_callback.p'
        self.history_callback = None
        self.loss_lambda = 1
        # self.opt = None
        self.learning_rate = 0.01 # edit
        self.save_weights_interval = 2
        # self.model_checkpoint = ModelCheckpoint(self.model_weights_path_to_save, \
        #     monitor='loss', verbose=1, \
        #     save_weights_only=True, \
        #     save_best_only=True, mode='auto', \
        #     period=self.save_weights_interval)
        #  To de updated in prepare_training_data()
        # ------------------------------------------

        # ---------------
        #  Visualization
        # ---------------
        self.vis = True #False # edit

        # -------
        #  Color
        # -------
        self.color_ls = ['crimson', 'lime green', 'royal blue', 'chocolate', 'purple', 'lemon']
        self.color_dict = {
            'crimson': (60,20,220),
            'lime green': (50,205,50),
            'royal blue': (225,105,65),
            'chocolate': (30,105,210),
            'purple': (128,0,128),
            'lemon': (0,247,255)
        }