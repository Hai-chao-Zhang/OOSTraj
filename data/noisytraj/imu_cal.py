import os.path
import time
import math
import numpy as np
import quaternion

g = 9.80665
# gravity = np.array([0, 0, 1])


dt = 1 #TODO: set to 1, should be 1/fps

c = 1e-6 # (1 + alpha) / 2.
alpha = 0.9
# file_pos = open("positions.txt","w")
# file_pos.write('{},{}\n'.format(position[0],position[1]))

#INFO: Acceleration:0:3, Gravity: 3:6, Linear: 6:9, Quaternion: 9:13, Magnetic: 13:16, Gyroscope: 16:19
#INFO: linear accelerations (three-axis accelerometer) and rotational velocities (three-axis gyroscope)

def imu_cal(seq_in_IMU19):


    num_frame, num_object, num_data = seq_in_IMU19.shape
    seq_in_IMU19_pos = np.zeros([num_frame, num_object, num_data+5])
    
    for o in range(num_object):

        speed = np.array([0, 0, 0])
        position = np.array([0, 0, 0])
        min_x, min_y, min_z = 10, 10, 10 
        max_x, max_y, max_z = -10, -10, -10
        
        for f in range(num_frame):
            # if f%4==0:
            #     continue
            data = seq_in_IMU19[f,o,:]
            seq_in_IMU19_pos[f,o,:num_data] = seq_in_IMU19[f,o,:]
            # x, y, z = imu.getFusionData()
            # print("%f %f %f" % (x,y,z))

            # data = imu.getIMUData()
            # fusionPose = data["fusionPose"]
            # pitch, roll, yaw = fusionPose
            
            # acc_x, acc_y, acc_z = (i * g for i in data["accel"])

            # acc_x, acc_y, acc_z = (i * g for i in imu.getAccelResiduals())
            #acc_z = (1 - alpha) * acc_z + alpha * sample_old
            #print(acc_z)
        
            # local_coordinate_system:
            
            lcs = data[9:13] # ["fusionQPose"]
            q_lcs = np.quaternion(lcs[0], lcs[1], lcs[2], lcs[3])
            # accel quaternion:
            acc_lcs = data[0:3] # ["accel"]
            q_acc = np.quaternion(acc_lcs[0], acc_lcs[1], acc_lcs[2])
            # global_coordinate_system:
            q_gcs = q_lcs * q_acc * q_lcs.inverse()
            acc_gcs = q_gcs.vec
            #print("LCS_X: %6.3f LCS_Y: %6.3f LCS_Z: %6.3f" % (acc_lcs[0], acc_lcs[1], acc_lcs[2]))
            #print("GCS_X: %6.3f GCS_Y: %6.3f GCS_Z: %6.3f" % (acc_gcs[0], acc_gcs[1], acc_gcs[2]))
            if np.isnan(acc_gcs).any():
                continue
            # rotate global coordinate system
            # on z axis by θ degrees
            angle = 160 # 90 -25
            angle_rad = math.radians(angle)
            # https://math.stackexchange.com/questions/2261003/rotating-a-quaternion-around-its-z-axis-to-point-its-x-axis-towards-a-given-poin
            q_rotator = np.quaternion(math.cos(angle_rad/2.0), 0, 0, math.sin(angle_rad/2.0))
            q_spiti = q_rotator * q_gcs * q_rotator.inverse()
            acc_spiti = q_spiti.vec
            # SCS = spiti coordinate system
            #print("SCS_X: %6.3f SCS_Y: %6.3f SCS_Z: %6.3f" % (acc_spiti[0], acc_spiti[1], acc_spiti[2]))
            min_x = min(min_x, acc_spiti[0])
            min_y = min(min_y, acc_spiti[1])
            min_z = min(min_z, acc_spiti[2])
            max_x = max(max_x, acc_spiti[0])
            max_y = max(max_y, acc_spiti[1])
            max_z = max(max_z, acc_spiti[2])
            #print("Min X: %6.3f Y: %6.3f Z:%6.3f" % (min_x, min_y, min_z))
            #print("Max X: %6.3f Y: %6.3f Z:%6.3f" % (max_x, max_y, max_z))
            #r, p, y = map(math.degrees,quaternion.as_euler_angles(q_lcs))
            #print("Roll: %10.3f Pitch: %10.3f Yaw: %10.3f" % (r, p, y))
            gravity = data[3:6]
            acc_spiti = -((acc_spiti - gravity) * g)
            speed = c * (acc_spiti * dt) + speed #alpha * speed
            position = c * (speed * dt) + position #alpha * position
            # file_pos.write('{:.3f},{:.3f}\n'.format(position[0],position[1]))
            #file_pos.write('{:.3f},{:.3f},{:.3f}\n'.format(position[0],position[1],position[2]))
            
            seq_in_IMU19_pos[f,o,num_data:] = [position[0], position[1], speed[0], speed[1], speed[2]]
            # print(position)
    assert np.isnan(seq_in_IMU19_pos).any()==False
    return seq_in_IMU19_pos

        
