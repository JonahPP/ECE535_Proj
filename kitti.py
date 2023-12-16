import argparse
import numpy as np
import cv2 
from os import mkdir 
from os.path import isdir 
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import datetime
import pandas as pd


def SO2quat(SO_data):
    rr = R.from_matrix(SO_data)
    return rr.as_quat()

def SE2pos_quat(SE_data):
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat

def kitti2tartan(traj):
    '''
    traj: in kitti style, N x 12 np array, in camera frame
    output: in TartanAir style, N x 7 np array, in NED frame
    '''
    T = np.array([[0,0,1,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3,:] = pose.reshape(3,4)
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(ttt))
        
    return np.array(new_traj)

# def pos_quats2SE_matrices(quat_datas):
#     data_len = quat_datas.shape[0]
#     SEs = []
#     for quat in quat_datas:
#         SO = R.from_quat(quat[3:7]).as_matrix()
#         SE = np.eye(4)
#         SE[0:3,0:3] = SO
#         SE[0:3,3]   = quat[0:3]
#         SEs.append(SE)
#     return SEs

# def evaluate(gt_traj, est_traj, scale):
#     gt_xyz = np.matrix(gt_traj[:,0:3].transpose())
#     est_xyz = np.matrix(est_traj[:, 0:3].transpose())

#     rot, trans, trans_error, s = align(gt_xyz, est_xyz)
#     print('  ATE scale: {}'.format(s))
#     error = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))

#     # align two trajs 
#     est_SEs = pos_quats2SE_matrices(est_traj)
#     T = np.eye(4) 
#     T[:3,:3] = rot
#     T[:3,3:] = trans 
#     T = np.linalg.inv(T)
#     est_traj_aligned = []
#     for se in est_SEs:
#         se[:3,3] = se[:3,3] * s
#         se_new = T.dot(se)
#         se_new = SE2pos_quat(se_new)
#         est_traj_aligned.append(se_new)

#     est_traj_aligned = np.array(est_traj_aligned)
#     return error, gt_traj, est_traj_aligned

# def align(model,data):
#     """Align two trajectories using the method of Horn (closed-form).
    
#     Input:
#     model -- first trajectory (3xn)
#     data -- second trajectory (3xn)
    
#     Output:
#     rot -- rotation matrix (3x3)
#     trans -- translation vector (3x1)
#     trans_error -- translational error per point (1xn)
    
#     """
#     np.set_printoptions(precision=3,suppress=True)
#     model_zerocentered = model - model.mean(1)
#     data_zerocentered = data - data.mean(1)
    
#     W = np.zeros( (3,3) )
#     for column in range(model.shape[1]):
#         W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
#     U,d,Vh = np.linalg.linalg.svd(W.transpose())
#     S = np.matrix(np.identity( 3 ))
#     if(np.linalg.det(U) * np.linalg.det(Vh)<0):
#         S[2,2] = -1
#     rot = U*S*Vh
#     trans = data.mean(1) - rot * model.mean(1)
    
#     model_aligned = rot * model + trans
#     alignment_error = model_aligned - data
    
#     trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        
#     return rot,trans,trans_error


def evaluate(gt_SEs, est_SEs, kittitype):
    error = kittievaluate(gt_SEs, est_SEs, kittitype=kittitype)
    return error

def trajectory_distances(poses):
    distances = []
    distances.append(0)
    for i in range(1,len(poses)):
        p1 = poses[i-1]
        p2 = poses[i]
        delta = p1[0:3,3] - p2[0:3,3]
        distances.append(distances[i-1]+np.linalg.norm(delta))
    return distances

def last_frame_from_segment_length(dist,first_frame,length):
    for i in range(first_frame,len(dist)):
        if dist[i]>dist[first_frame]+length:
            return i
    return -1

def rotation_error(pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5*(a+b+c-1)
    rot_error = np.arccos(max(min(d,1.0),-1.0))
    return rot_error

def translation_error(pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx*dx+dy*dy+dz*dz)

# def line2matrix(pose_line):
#     pose_line = np.array(pose_line)
#     pose_m = np.matrix(np.eye(4))
#     pose_m[0:3,:] = pose_line.reshape(3,4)
#     return pose_m
    
def calculate_sequence_error(poses_gt,poses_result,lengths=[10,20,30,40,50,60,70,80]):
    # error_vetor
    errors = []

    # paramet
    step_size = 1 #09; # every second
    num_lengths = len(lengths)

    # import ipdb;ipdb.set_trace()
    # pre-compute distances (from ground truth as reference)
    dist = trajectory_distances(poses_gt)
    # for all start positions do
    for  first_frame in range(0, len(poses_gt), step_size):
    # for all segment lengths do
        for i in range(0,num_lengths):
            #  current length
            length = lengths[i];

            # compute last frame
            last_frame = last_frame_from_segment_length(dist,first_frame,length);
            # continue, if sequence not long enough
            if (last_frame==-1):
                continue;

            # compute rotational and translational errors
            pose_delta_gt     = np.linalg.inv(poses_gt[first_frame]).dot(poses_gt[last_frame])
            pose_delta_result = np.linalg.inv(poses_result[first_frame]).dot(poses_result[last_frame])
            pose_error        = np.linalg.inv(pose_delta_result).dot(pose_delta_gt)
            r_err = rotation_error(pose_error);
            t_err = translation_error(pose_error);

            # compute speed
            num_frames = (float)(last_frame-first_frame+1);
            speed = length/(0.1*num_frames);

            # write to file
            error = [first_frame,r_err/length,t_err/length,length,speed]
            errors.append(error)
            # return error vector
    return errors

def calculate_ave_errors(errors,lengths=[10,20,30,40,50,60,70,80]):
    rot_errors=[]
    tra_errors=[]
    for length in lengths:
        rot_error_each_length =[]
        tra_error_each_length =[]
        for error in errors:
            if abs(error[3]-length)<0.1:
                rot_error_each_length.append(error[1])
                tra_error_each_length.append(error[2])

        if len(rot_error_each_length)==0:
            # import ipdb;ipdb.set_trace()
            continue
        else:
            rot_errors.append(sum(rot_error_each_length)/len(rot_error_each_length))
            tra_errors.append(sum(tra_error_each_length)/len(tra_error_each_length))
    return np.array(rot_errors)*180/np.pi, tra_errors

def evaluate(gt, data,kittitype=True):
    if kittitype:
        lens =  [100,200,300,400,500,600,700,800] #
    else:
        lens = [5,10,15,20,25,30,35,40] #[1,2,3,4,5,6] # 
    errors = calculate_sequence_error(gt, data, lengths=lens)
    rot,tra = calculate_ave_errors(errors, lengths=lens)
    return np.mean(rot), np.mean(tra)


    
""""
to change easily 

"KITTI\\09.txt"
"KITTI_gt\\09.txt"
"Kitti09_updated.txt"

"""



# loading estimate result data, shifting kitti (12 column) to tartanVO (7 column)
# 7 x 4541
est = np.loadtxt("KITTI\\09.txt")
# print(len(est[0]))



# 12 x 4541
# loading ground truth (12 column)
gt = np.loadtxt("KITTI_gt\\09.txt")
# print(len(gt[0]))
new_gt = kitti2tartan(gt)
np.savetxt("Kitti09_updated.txt", new_gt)
gt = np.loadtxt("Kitti09_updated.txt")


# only need the first 3 for x, y
cut_est = est[:, :3]
trans_est = cut_est.T

cut_gt = new_gt[:, :3]
trans_gt = cut_gt.T

# rot, trans, error, gt_traj, est_traj_aligned = evaluate(new_gt, est, True)


fig = plt.figure(figsize=(7,6))
ax= fig.add_subplot(111)

plt.plot(trans_est[0], trans_est[1], label='Estimated', color='orange')
plt.plot(trans_gt[0], trans_gt[1], label='Ground Truth', color='black', linestyle='dashed')

ax.set_title('TartanVO Graph for Kitti09.txt')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend()
plt.show()