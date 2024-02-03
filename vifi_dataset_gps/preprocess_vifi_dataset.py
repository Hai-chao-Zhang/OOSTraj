import json
import numpy as np
import os
from collections import OrderedDict
# from utils import ots26_to_ts16_dfv3
# from read_wireless import read_wirless
from glob import glob
import pickle
from tqdm import tqdm
import csv
import torch

import datetime
from collections import defaultdict

# This method convert the Original TimeStamp (ots26) 2021-10-07 13:46:34.773741 into ts16_dfv3.
def ots26_to_ts16_dfv3(ots26):
    year, month, day = int(ots26[:4]), int(ots26[5:7]), int(ots26[8:10])
    hour, minute, second = int(ots26[11:13]), int(ots26[14:16]), int(ots26[17:19])
    after_second = int(ots26[20:26])
    dt = datetime.datetime(year, month, day, hour, minute, second, after_second)

    # print();print() # Debug
    # print('dt.timestamp(): ', dt.timestamp())
    # e.g. dt.timestamp():  1633628794.773

    # Debug
    dt_ts_str = str(dt.timestamp()) # ts16_dfv3
    while len(dt_ts_str) < 17:
        dt_ts_str += '0'
    # print('dt_ts_str: ', dt_ts_str, ', len(dt_ts_str): ', len(dt_ts_str))
    # e.g. dt_ts_str:  1633628794.773000 , len(dt_ts_str):  17
    ts16_dfv3 = dt_ts_str
    return ts16_dfv3

    '''
    # debug
    ots26 = '2021-10-07 13:46:34.773000'
    ts16_dfv3 = ots26_to_ts16_dfv3(ots26)
    print('ts16_dfv3: ', ts16_dfv3, ', len(ts16_dfv3): ', len(ts16_dfv3))
    # e.g. ts16_dfv3:  1633628794.773000 , len(ts16_dfv3):  17
    '''

def read_wirless(csv_file):
# Initialize an empty list to store the filtered data
    wi_flags = ['STEPCOUNT','STEPDET', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'STEPCOUNT', 'GYRO', 'GPS', 'ACCEL']
    # Open and read the CSV file

    data = {i:dict() for i in wi_flags}
    with open(csv_file, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Check if the row has at least two elements (0-based index)
            if row[1] in wi_flags:
                tm = str(row[0][:10]+"."+row[0][10:])
                if row[1] not in ['STEPCOUNT','STEPDET', 'GPS']:
                    data[row[1]][ float(tm) ] = row[2:]
                elif row[1] == 'GPS':
                    data[row[1]][ float(tm) ] = row[2:4]
                # Append the row to the filtered_data list
                # filtered_data.append(row)
    
    overlap_start = 0 # max(A[0], B[0])
    overlap_end = 9e20 # min(A[-1], B[-1])

    for sensor, signals in data.items():
        if sensor in ['STEPCOUNT','STEPDET']:
            continue
        data[sensor] = OrderedDict(sorted(signals.items(), key=lambda obj: obj[0]))

        ls = list(data[sensor].keys())
        if ls[0]>overlap_start:
            overlap_start =  ls[0]
        if ls[-1]< overlap_end:
            overlap_end = ls[-1]
    # Now, 'filtered_data' contains only the rows with at least two elements
    # print(data)
    # wi = dict()
    # for key, value in data.items():
    #     if key in ['STEPCOUNT','STEPDET']:
    #         continue
    #     data[key] = np.array(value)
    # wi['wireless'] = np.concatenate([ data['GRAV'], data['LINEAR'], data['Quaternion'], data['MAG'], data['GYRO'], data['ACCEL']])
    
    return data, overlap_start, overlap_end

# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
# def closest(ls, K):
#     return ls[min(range(len(ls)), key = lambda i: abs(float(ls[i])-float(K)))]
def closest(ls, K): #  binary search to find the nearest elemnt in ls
    low = 0
    high = len(ls) - 1
    
    while low <= high:
        mid = (low + high) // 2
        mid_value = float(ls[mid])
        
        if mid_value == float(K):
            return ls[mid]
        elif mid_value < float(K):
            low = mid + 1
        else:
            high = mid - 1
    
    # At this point, low and high have crossed, and low points to the closest element.
    # Check if low is out of bounds and return the closest element.
    if low >= len(ls):
        return ls[-1]
    elif low == 0:
        return ls[0]
    else:
        left_value = float(ls[low - 1])
        right_value = float(ls[low])
        if abs(left_value - float(K)) <= abs(right_value - float(K)):
            return ls[low - 1]
        else:
            return ls[low]

def extract_bbx_wireless_fromvifi():
    # Directory where JSON files are stored
    dataset_folder = "dataset"


    # Iterate through each JSON file in the dataset folder
    for folder in tqdm(os.listdir(dataset_folder)):
        # Create a dictionary to store object points for each frame
        object_points_by_frame = {}
        ts_img_map = {}

        # if filename.endswith(".json"):
        #     # Construct the full path to the JSON file
        json_filepath = os.path.join(dataset_folder, folder, "GND/vott-json-export/")

        # /20211004_142757/IMU/A_Subject1_Phone_Oct_4,_2021_2_28_03_PM.csv'
        if not os.path.exists(json_filepath):
            continue    
        else:
            json_filepath = os.path.join(json_filepath, os.listdir(json_filepath)[0])
        # Open and parse the JSON data
        with open(json_filepath, "r") as json_file:
            parsed_data = json.load(json_file)
        

        # Parse the JSON data
        # parsed_data = json.loads(json_data)

        # Get the list of tags
        tags = parsed_data["tags"]
        identities = [diction['name'] for diction in tags if diction['name']!='Others']
        # Create a dictionary to store object points for each frame
        object_points_by_frame = dict()

        # Iterate through each asset/frame
        for asset_id, asset_info in parsed_data["assets"].items():
            # Initialize a list to store object points for this frame
            frame_object_points = OrderedDict()
            
            # Get the regions for this asset/frame
            regions = asset_info["regions"]

            ots26_time = asset_info['asset']["name"].split(".png")[0].replace("%20"," ")
            ts16_time = ots26_to_ts16_dfv3(ots26_time)
            
            # # Iterate through each tag and find its corresponding region in this frame

            # for tag_name in identities:

            #     # tag_name = tag["name"]
            #     # if tag_name not in identities:
            #     #     continue
            #     # Initialize a zero tensor for this object
            object_points = np.zeros((4, 2))
            
            # Find the region with the matching tag in this frame
            # count_id = 0
            for region in regions:
                if region["tags"][0] not in identities:
                    continue
                # else:
                #     count_id = count_id + 1
                # Extract the four points of the bounding box
                bounding_box_points = region["points"] #### 'boundingBox': {'height': 218, 'width': 82, 'left': 607, 'top': 221}, 'points': [{'x': 607, 'y': 221}, {'x': 689, 'y': 221}, {'x': 689, 'y': 439}, {'x': 607, 'y': 439}]}
                # Convert the points to a NumPy array
                # object_points = np.array([(point["x"], point["y"]) for point in bounding_box_points])
                # object_points = np.array([(point["x"], point["y"]) for point in bounding_box_points])
                object_points = region["boundingBox"] # top-right & bot-left
                # break
                
                # Append the object points to the frame list
                frame_object_points[region["tags"][0]] = object_points
            
            # Add the frame's object points to the dictionary
            object_points_by_frame[float(ts16_time)] = frame_object_points
            ts_img_map[float(ts16_time)] = asset_info['asset']["name"]
        

        csv_file = os.path.join(dataset_folder, folder, "IMU/*")

        wireless = OrderedDict()
        csv_files = glob(csv_file)
        csv_files.sort()
        for csv in csv_files:
            subject_id = csv.split("_Subject")[-1][0]
            # wireless[f"Subject{subject_id}"], overlap_start, overlap_end =read_wirless(csv)
            wireless[f"Subject{subject_id}"], overlap_start, overlap_end = read_wirless(csv)
        
        subject_ids  = [ f'Subject{csv.split("_Subject")[-1][0]}' for csv in csv_files]



        object_points_by_frame = OrderedDict(sorted(object_points_by_frame.items(), key=lambda obj: obj[0]))
        timelist = list(object_points_by_frame.keys())

        overlap_start = max(overlap_start, timelist[0])
        overlap_end = min(overlap_end, timelist[-1])

        object_points_by_frame =  OrderedDict({ i: object_points_by_frame[i] for i in timelist if i > overlap_start and i < overlap_end})
        timelist = [i for i in timelist if i > overlap_start and i < overlap_end]


        aligned_wire={i:dict() for i in ['STEPCOUNT','STEPDET', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO', 'GPS', 'ACCEL']}
        
        imu_by_sub = OrderedDict()
        gps_by_sub = OrderedDict()
        for subject, wire in wireless.items():
            for sensor, signals in wire.items():
                if sensor in ['STEPCOUNT','STEPDET']:
                    continue
                IMU_ts13_dfv4_ls = list(signals.keys())
                
                IMU_ts13_dfv4 = [closest(IMU_ts13_dfv4_ls, RGB_ts16_dfv4) for RGB_ts16_dfv4 in timelist] #TODO: subjects order of bbx , fill in the missing subject frames. GPS needs to be estimated by each 段落 in video. save in pkl as before 

                aligned_wire[sensor][subject] = np.array([wireless[subject][sensor][i] for i in IMU_ts13_dfv4])
            
            imu_by_sub[subject] = np.concatenate([ aligned_wire[sensor][subject] for sensor in ['GRAV', 'LINEAR', 'Quaternion', 'MAG','GYRO', 'ACCEL'] ],axis=1) #'GPS', 
            gps_by_sub[subject] = aligned_wire['GPS'][subject]
            #GPS up sample, while others down sample
        # print(); print() # debug
        # print('IMU_ts13_dfv4: ', IMU_ts13_dfv4, ', RGB_ts16_dfv4: ', RGB_ts16_dfv4)
        '''
        e.g.
        IMU_ts13_dfv4:  1608750656.696 , RGB_ts16_dfv4:  1608750656.689327
        IMU_ts13_dfv4:  1608750657.023 , RGB_ts16_dfv4:  1608750657.022574
        IMU_ts13_dfv4:  1608750657.364 , RGB_ts16_dfv4:  1608750657.356593
        '''
        # aligned_wire
        

        # subject_ids_points = OrderedDict({i:object_points_by_frame[i] for i in subject_ids for Object, object_points in object_points_by_frame.items() if i in object_points_by_frame else np.zeros_like() })

        # backup plan resize and crop
        # # If an object is missing in a frame, its points will be represented as a 0 tensor.
        gps_crop_by_sub = {}
        for subject in subject_ids:
            signals = wireless[subject]["GPS"]
            sensor_stamp = list(signals.keys())
            gps_crop_by_sub[subject] = np.array([signals[i] for i in sensor_stamp if i > overlap_start and i < overlap_end])


        bbx_by_time = OrderedDict()
        bbx = []
        for stamp, frame in object_points_by_frame.items():
            # missingsubjects = [i for i in subject_ids if i not in frame]
            Sub = []
            for subj in subject_ids:
                if subj not in frame:
                    rect = {'height': -1, 'width': -1, 'left': -1, 'top': -1}
                else:
                    rect = frame[subj]
                # 
                Sub.append([rect['left'], rect['top'], rect['height'], rect['width']])
            bbx_by_time[stamp] = Sub
            bbx.append(Sub)
        bbx = np.array(bbx)

        for subj in subject_ids:
            imu = np.stack([imu_by_sub[subject] for subj in subject_ids], axis=1)

            nearest_gps = np.stack([gps_by_sub[subject] for subj in subject_ids], axis=1)

            gps_crop = np.stack([gps_crop_by_sub[subject] for subj in subject_ids], axis=1)

            # for Object, points in frame.items():
            #     pass
        
        # Save the object
        with open(f'/home/haichao/code/gps/pkl/{folder}.pkl', 'wb') as f:
            pickle.dump({"bbx":bbx, "imu":imu, "nearest_gps":nearest_gps, "gps_crop":gps_crop, "bbx_by_frame":object_points_by_frame, "timestamp2img_map":ts_img_map}, f) 

def load_savedbbx(step = 50, length = 200):
    data = {"video":[], "timestamp":[], "bbx":[], "imu":[], "nearst_gps":[], "interl_gps":[]}

    pklfiles = os.listdir("./pkl/")
    for pklfile in pklfiles:
        with open(os.path.join( "./pkl/",pklfile), 'rb') as f:
            info_dict = pickle.load(f)
        bbx, imu, nearst_gps, gps = info_dict["bbx"], info_dict["imu"], info_dict["nearest_gps"], info_dict[ "gps_crop"]
        l = bbx.shape[0]
        bbx, imu, nearst_gps, gps = torch.Tensor(bbx),torch.Tensor(imu.astype(float)),torch.Tensor(nearst_gps.astype(float)),torch.Tensor(gps.astype(float))
        timestamps = list(info_dict["bbx_by_frame"].keys())
        
        gps = torch.nn.functional.interpolate(gps.permute(1,2,0), size=nearst_gps.shape[0], mode='linear').permute(2,0,1)
        

        for i in range(0,(l//step -1)*step, step):
            data["video"].append(pklfile.split(".")[0])
            data["timestamp"].append(timestamps[i:i+length])
            data["bbx"].append( bbx[i:i+length])
            data["imu"].append( imu[i:i+length])
            data["nearst_gps"].append( nearst_gps[i:i+length])
            data["interl_gps"].append( gps[i:i+length])
    # data
    # data["bbx"] = torch.cat(data["bbx"])
    # data["imu"] = torch.cat(data["imu"])
    # data["nearst_gps"] = torch.cat(data["nearst_gps"])
    # data["interl_gps"] = torch.cat(data["interl_gps"])
    with open(f'./vifi_data.pkl', 'wb') as f:
        pickle.dump(data, f) 
    # import pickle
    # with open(os.path.join( "/home/haichao/code/gps/vifi_data.pkl"), 'rb') as f:
    #     vifi_data = pickle.load(f)
    # video, timestamp, bbx, imu, nearst_gps, interl_gps = data["video"], data["timestamp"], data["bbx"], data["imu"], data["nearst_gps"], data["interl_gps"]
      

def read_martice(json_file_path):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    # Now, you can access the data using the keys in the JSON object
    R_w2c = data["R_w2c"]
    t_w2c = data["t_w2c"]
    T_w2c = data["T_w2c"]
    T_c2w = data["T_c2w"]
    Estimated_RSU_lla = data["Estimated RSU lla"]
    Surveyed_RSU_lla = data["Surveyed RSU lla"]

    # You can then work with the data as needed
    print("R_w2c:", R_w2c)
    print("t_w2c:", t_w2c)
    print("T_w2c:", T_w2c)
    print("T_c2w:", T_c2w)
    print("Estimated RSU lla:", Estimated_RSU_lla)
    print("Surveyed RSU lla:", Surveyed_RSU_lla)
    return T_c2w # (latitude,longitude,altitude)

if __name__ =="__main__":
    # extract_bbx_wireless_fromvifi()
    load_savedbbx()