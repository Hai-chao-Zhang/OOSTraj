import torch, torchvision
from torchvision import transforms
import json
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pickle, random, copy
from .noisytraj.imu_cal import imu_cal


def prepare_training_data(C):
    seq_in_BBX5_dfv3_ls, seq_in_FTM2_dfv3_ls, seq_in_IMU19_dfv3_ls = [], [], []
    seq_in_BBX5_dfv3_ls_mask = []
    '''
    C.BBX5_sync_dfv3 e.g. (589, 5, 5)
    C.IMU19_sync_dfv3 e.g. (589, 5, 19)
    '''
    C.img_type = 'RGBh_ts16_dfv2'
    print(); print() # debug
    print('C.seq_id_ls: ', C.seq_id_ls)
    # ---------------------------------------
    #  Iterate Over All Train Seq_id - Start
    # ---------------------------------------
    for C.seq_id_idx, C.seq_id in enumerate(C.seq_id_ls):
        if C.seq_id != C.test_seq_id:   #TODO: currently we only use 14 of these 15 datasets, skip the test_seq_id
            C.seq_path  = C.seq_root_path + '/' + C.seq_id
            C.img_path = C.seq_path + '/' + C.img_type
            if C.seq_id  == "20201229_162235":
                print(C.seq_id)
            C.seq_date = C.seq_id[:8]
            C.seq_path_for_model = C.seq4model_root_path + '/' + C.seq_id
            # C.img_path = C.seq_path + '/' + C.img_type
            C.RGBh_ts16_dfv3_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv3_ls.json'
            with open(C.RGBh_ts16_dfv3_ls_path, 'r') as f:
                C.RGBh_ts16_dfv3_ls = json.load(f)
                print(C.RGBh_ts16_dfv3_ls_path, 'loaded!')
                print('C.RGBh_ts16_dfv3_ls[:5]: ', C.RGBh_ts16_dfv3_ls[:5])
            
            if C.vis: C.img_path = C.seq_path + '/' + C.img_type

            # ------------------------------------------
            #  Synchronized data: BBX5,IMU19_sync_dfv3
            # ------------------------------------------
            C.sync_dfv3_path = C.seq_path_for_model + '/sync_ts16_dfv3'
            # -----------------
            #  Load BBX5 Data
            # -----------------
            C.BBX5_dim = 5
            C.BBX5_sync_dfv3_path = C.sync_dfv3_path + '/BBX5H_sync_dfv3.pkl'
            C.BBX5_sync_dfv3 = pickle.load(open(C.BBX5_sync_dfv3_path, 'rb'))
            C.BBX5_sync_dfv3 = np.nan_to_num(C.BBX5_sync_dfv3, nan=0)        #########TODO: Mask algorithm
            print(); print() # debug
            print('np.shape(C.BBX5_sync_dfv3): ', np.shape(C.BBX5_sync_dfv3))
            # e.g. (535, 5, 5)

            # -----------------
            #  Load IMU19 Data
            # -----------------
            C.IMU19_dim = 3 + 3 + 3 + 4 + 3 + 3 # 19
            C.IMU19_data_types = ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO'] # Acceleration:3, Gravity:3, Linear:3, Quaternion:4, Magnetic:3, Gyroscope:3
            C.IMU19_sync_dfv3_path = C.sync_dfv3_path + '/IMU19_sync_dfv3.pkl'  #  linear accelerations (three-axis accelerometer) and rotational velocities (three-axis gyroscope)
            #TODO: what is Linear?
            C.IMU19_sync_dfv3 = pickle.load(open(C.IMU19_sync_dfv3_path, 'rb'))
            print(); print() # debug
            print('np.shape(C.IMU19_sync_dfv3): ', np.shape(C.IMU19_sync_dfv3))
            # e.g. (535, 5, 19)

            # ----------------
            #  Load FTM2 Data
            # ----------------
            C.FTM2_dim = 2
            C.FTM2_sync_dfv3_path = C.sync_dfv3_path + '/FTM_sync_dfv3.pkl'
            C.FTM2_sync_dfv3 = pickle.load(open(C.FTM2_sync_dfv3_path, 'rb'))
            print(); print() # debug
            print('np.shape(C.FTM2_sync_dfv3): ', np.shape(C.FTM2_sync_dfv3))
            # e.g. (535, 5, 2)

            # --------------
            #  Video Window
            # --------------
            C.crr_ts16_dfv3_ls_all_i = 0
            C.video_len = len(C.RGBh_ts16_dfv3_ls) # len(C.ts12_BBX5_all)
            print(); print() # debug
            print('C.video_len: ', C.video_len) # e.g. 1800
            C.n_wins = C.video_len - C.recent_K + 1
            print('C.n_wins: ', C.n_wins) # e.g. 1791

            # --------------
            #  Prepare BBX5
            # --------------
            curr_in_view_i_ls = []

            
            Training_Phase = C.args.phase
            is_Random = C.args.random

            for win_i in range(C.n_wins):
                win_seq = C.BBX5_sync_dfv3[win_i : win_i + C.recent_K, :, :]  # TODO: recent_K is the total length of the sequence, should be set to 50      length, subj_id, dim[col, row, depth, width, height]
                mask_subj_id = [random.randint(0, len(C.subjects)-1)] if is_Random else range(len(C.subjects)-1) # choose a subject to be masked]

                # for subj_i in range(len(C.subjects) - 1):
                for subj_i in range(len(C.subjects) -1):
                    # seq_in_BBX5_ = C.BBX5_sync_dfv3[subj_i, win_i : win_i + C.recent_K, :] # old
                    seq_in_BBX5_ = win_seq[:,subj_i,:]
                    # length, dim[col, row, depth, width, height].    for subj_id \belongto [0, C.subjects]
                    seq_in_BBX5_masked_ = copy.deepcopy(seq_in_BBX5_) 


                    if Training_Phase == 1:
                        length = 10
                    elif Training_Phase == 2:
                        length = 1
                    
                    if subj_i in mask_subj_id:
                        
                        mask_start_point = random.randint(0, C.recent_K - 10) if is_Random else 0 # mask algorithm, now random start, length = 10. TODO: change mask algorithm, start_point?
                        # seq_in_BBX5_masked_[ mask_start_point: mask_start_point + length, :] = 0 # padding as zero. TODO: padding method needs update
                        seq_in_BBX5_masked_[ : mask_start_point, :] = 0 
                        seq_in_BBX5_masked_[ mask_start_point + length:, :] = 0

                    seq_in_BBX5_dfv3_ls_mask.append(seq_in_BBX5_masked_ if subj_i in mask_subj_id else seq_in_BBX5_)

                    # print(); print() # debug
                    # print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 5)
                    # print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
                    # print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
                    # print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
                    # print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
                    # print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height
                    '''
                    e.g.
                    seq_in_BBX5_[:, 0]:  [641. 631. 618. 604. 592. 583. 577. 570. 565. 562.]
                    seq_in_BBX5_[:, 1]:  [635. 630. 627. 623. 619. 615. 611. 607. 604. 602.]
                    seq_in_BBX5_[:, 2]:  [1.73513258 1.75361669 1.78351653 1.84246898 1.86370301 1.86906254
                     1.90441883 1.93990803 1.98963535 2.04343772]
                    seq_in_BBX5_[:, 3]:  [157. 152. 147. 146. 147. 148. 149. 145. 142. 140.]
                    seq_in_BBX5_[:, 4]:  [163. 173. 180. 188. 195. 203. 211. 218. 225. 228.]
                    '''
                    # -----------------------------------------------------------------------------
                    #  Note that RGB_ts16_dfv3_valid_ls only works for New Dataset in this version
                    # -----------------------------------------------------------------------------
                    # if C.vis:
                    #     subj_i_RGB_ots26_img_path = C.img_path + '/' + ts16_dfv3_to_ots26(C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]) + '.png'
                    #     print(); print() # debug
                    #     print('subj_i_RGB_ots26_img_path: ', subj_i_RGB_ots26_img_path)
                    #     img = cv2.imread(subj_i_RGB_ots26_img_path)
                    #     print(); print() # debug
                    #     print(C.subjects[subj_i], ', np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 5)
                    #     print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
                    #     print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
                    #     print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
                    #     print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
                    #     print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height
                    #     '''
                    #     e.g.
                    #     Sid , np.shape(seq_in_BBX5_):  (10, 5)
                    #     seq_in_BBX5_[:, 0]:  [866. 833. 809.   0.   0.   0.   0. 676. 653. 638.]
                    #     seq_in_BBX5_[:, 1]:  [427. 427. 432.   0.   0.   0.   0. 446. 451. 485.]
                    #     seq_in_BBX5_[:, 2]:  [9.25371265 9.26818466 8.887537   0.         0.         0.
                    #      0.         7.5010891  8.03569031 8.17784595]
                    #     seq_in_BBX5_[:, 3]:  [40. 35. 32.  0.  0.  0.  0. 34. 64. 67.]
                    #     seq_in_BBX5_[:, 4]:  [ 46.  46.  62.   0.   0.   0.   0.  63.  77. 144.]
                    #     '''
                    #     subj_color = C.color_dict[C.color_ls[subj_i]]
                    #
                    #     for k_i in range(C.recent_K):
                    #         top_left = (int(seq_in_BBX5_[k_i, 0]) - int(seq_in_BBX5_[k_i, 3] / 2), \
                    #                     int(seq_in_BBX5_[k_i, 1]) - int(seq_in_BBX5_[k_i, 4] / 2))
                    #         bottom_right = (int(seq_in_BBX5_[k_i, 0]) + int(seq_in_BBX5_[k_i, 3] / 2), \
                    #                     int(seq_in_BBX5_[k_i, 1]) + int(seq_in_BBX5_[k_i, 4] / 2))
                    #         img = cv2.circle(img, (int(seq_in_BBX5_[k_i, 0]), int(seq_in_BBX5_[k_i, 1])), 4, subj_color, 4) # Note (col, row) or (x, y) here
                    #         img = cv2.rectangle(img, top_left, bottom_right, subj_color, 2)
                    #     cv2.imshow('img', img); cv2.waitKey(0)

                    curr_in_view_i_ls.append(win_i * 5 + subj_i)
                    seq_in_BBX5_dfv3_ls.append(seq_in_BBX5_)
            # for win_i in range(C.n_wins):
            #     if Training_Phase == 1:
            #         mask_subj_id = random.randint(0, len(C.subjects) - 1) # choose a subject to be masked
            #         for subj_i in range(len(C.subjects) - 1):
            #             # seq_in_BBX5_ = C.BBX5_sync_dfv3[subj_i, win_i : win_i + C.recent_K, :] # old
            #             seq_in_BBX5_ = C.BBX5_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]  # TODO: recent_K should be set to 50      length, subj_id, dim[col, row, depth, width, height]
            #             # length, dim[col, row, depth, width, height].    for subj_id \belongto [0, C.subjects]
            #             if subj_i == mask_subj_id:
            #                 mask_start_point = random.randint(0, C.recent_K - 10) # mask algorithm, now random start, length = 10. TODO: change mask algorithm, start_point?
            #                 seq_in_BBX5_masked_ = copy.deepcopy(seq_in_BBX5_) 
            #                 seq_in_BBX5_masked_[ mask_start_point: mask_start_point + 10, :] = 0 # padding as zero. TODO: padding method needs update
            #                 seq_in_BBX5_masked_[ : mask_start_point, :] = 0 
            #                 seq_in_BBX5_masked_[ mask_start_point + 10:, :] = 0
            #             seq_in_BBX5_dfv3_ls_mask.append(seq_in_BBX5_masked_ if subj_i == mask_subj_id else seq_in_BBX5_)
            #     elif Training_Phase == 2:
            #         pass
            #     elif Training_Phase == 3:
            #         pass
            #     else:
            #         raise ValueError("Wrong Traing Phase")
            
            # ---------------
            #  Prepare IMU19
            # ---------------
            for win_i in range(C.n_wins):
                for subj_i in range(len(C.subjects) - 1):
                    curr_in_view_i = win_i * 5 + subj_i
                    if curr_in_view_i in curr_in_view_i_ls:
                        # seq_in_BBX5_ = C.BBX5_sync_dfv3[subj_i, win_i : win_i + C.recent_K, :] # old
                        seq_in_BBX5_ = C.BBX5_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
                        # print('seq_in_BBX5_: ', seq_in_BBX5_)
                        k_start_i = C.recent_K
                        for k_i in range(C.recent_K):
                            # print(type(seq_in_BBX5_[k_i])) # numpy.ndarray
                            if 0 not in seq_in_BBX5_[k_i]:
                                k_start_i = k_i
                                break

                        # seq_in_IMU19_ = C.IMU19_sync_dfv3[subj_i, win_i : win_i + C.recent_K, :] # old
                        seq_in_IMU19_ = C.IMU19_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
                        for k_i in range(C.recent_K):
                            if k_i < k_start_i:
                                IMU19_not_in_view = np.full((1, C.IMU19_dim), 0)
                                # print('IMU19_not_in_view: ', IMU19_not_in_view)
                                seq_in_IMU19_[k_i] = IMU19_not_in_view

                        # --------------------------------------------
                        # When subject appears in the middle of the K
                        #   frames, not from the beginning.
                        # --------------------------------------------
                        # if k_start_i > 0:
                        #     print('seq_in_IMU19_: ', seq_in_IMU19_)
                        # print('np.shape(seq_in_IMU19_): ', np.shape(seq_in_IMU19_)) # e.g. (10, 19)
                        seq_in_IMU19_dfv3_ls.append(seq_in_IMU19_)

                        # >>> FTM2 >>>
                        seq_in_FTM2_ = C.FTM2_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
                        for k_i in range(C.recent_K):
                            if k_i < k_start_i:
                                FTM2_not_in_view = np.full((1, C.FTM2_dim), 0)
                                seq_in_FTM2_[k_i] = FTM2_not_in_view
                        seq_in_FTM2_dfv3_ls.append(seq_in_FTM2_)
                        # <<< FTM2 <<<

            print(); print() # debug
            print('len(seq_in_IMU19_dfv3_ls): ', len(seq_in_IMU19_dfv3_ls))
            print('len(seq_in_FTM2_dfv3_ls): ', len(seq_in_FTM2_dfv3_ls))

    # -------------------------------------
    #  Iterate Over All Train Seq_id - End
    # -------------------------------------
    # C.seq_in_BBX5 = np.array(seq_in_BBX5_dfv3_ls) ######################
    # C.seq_out_BBX5 = copy.deepcopy(C.seq_in_BBX5) ######################
    C.seq_in_BBX5 = np.array(seq_in_BBX5_dfv3_ls_mask) 
    C.seq_out_BBX5 = np.array(seq_in_BBX5_dfv3_ls) 


    # C.seq_in_BBX5 = tf.convert_to_tensor(C.seq_in_BBX5)
    # C.seq_out_BBX5 = tf.convert_to_tensor(C.seq_out_BBX5)
    print('np.shape(C.seq_in_BBX5): ', np.shape(C.seq_in_BBX5)) # e.g. (27376, 10, 5)
    print('np.shape(C.seq_out_BBX5): ', np.shape(C.seq_out_BBX5))

    C.seq_in_FTM2 = np.array(seq_in_FTM2_dfv3_ls)
    C.seq_out_FTM2 = copy.deepcopy(C.seq_in_FTM2)
    # C.seq_in_FTM2 = tf.convert_to_tensor(C.seq_in_FTM2)
    # C.seq_out_FTM2 = tf.convert_to_tensor(C.seq_out_FTM2)
    print('np.shape(C.seq_in_FTM2): ', np.shape(C.seq_in_FTM2)) # e.g. (27376, 10, 2)
    print('np.shape(C.seq_out_FTM2): ', np.shape(C.seq_out_FTM2))

    C.seq_in_IMU19 = np.array(seq_in_IMU19_dfv3_ls)
    C.seq_out_IMU19 = copy.deepcopy(C.seq_in_IMU19)
    # C.seq_in_IMU19 = tf.convert_to_tensor(C.seq_in_IMU19)
    # C.seq_out_IMU19 = tf.convert_to_tensor(C.seq_out_IMU19)
    print('np.shape(C.seq_in_IMU19): ', np.shape(C.seq_in_IMU19)) # e.g. (27376, 10, 19)
    print('np.shape(C.seq_out_IMU19): ', np.shape(C.seq_out_IMU19))

    assert np.shape(C.seq_in_BBX5)[1] == np.shape(C.seq_in_FTM2)[1]
    assert np.shape(C.seq_in_BBX5)[1] == np.shape(C.seq_in_IMU19)[1]
    assert np.shape(C.seq_in_BBX5)[2] == C.BBX5_dim
    assert np.shape(C.seq_in_FTM2)[2] == C.FTM2_dim
    assert np.shape(C.seq_in_IMU19)[2] == C.IMU19_dim

    print(); print() # debug
    print('seq_id_path_ls: ', C.seq_id_path_ls)
    print('seq_id_ls: ', C.seq_id_ls)
    print('C.BBX5_sync_dfv3_path: ', C.BBX5_sync_dfv3_path)

    return C.seq_in_BBX5, C.seq_in_FTM2, C.seq_in_IMU19, C.seq_out_BBX5