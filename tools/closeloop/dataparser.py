import cv2
import mmcv
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SCR
import math

OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
OPENCV2LIDAR = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def get_intrinsic(intrin_dict):
    fovx, fovy = intrin_dict['fovx'], intrin_dict['fovy']
    h, w = intrin_dict['H'], intrin_dict['W']
    K = np.eye(4)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrin_dict['cx'], intrin_dict['cy']
    return K

def parse_raw(raw_data, cameras, img_norm_cfg, history):
    obs, info = raw_data
    cam_params = info['cam_params']
    imu_rot = np.array([0,0,0], dtype=np.float32)
    imu_rot[2] = -info['ego_rot'][1]
    lidar_rot = np.array([0,0,0], dtype=np.float32)
    lidar_rot[2] = info['ego_rot'][1]
    
    imgs_shape = []
    imgs = []
    raw_imgs = {}
    l2imgs, intrinsics = [], []
    
    for cam in cameras:
        im = obs['rgb'][cam]
        raw_imgs[cam] = im
        # x_scale, y_scale = 380 / im.shape[0], 640 / im.shape[1]
        im = cv2.resize(im, (1600, 900))
        im_norm = mmcv.imnormalize(im, img_norm_cfg['mean'], img_norm_cfg['std'], img_norm_cfg['to_rgb'])
        imgs_shape.append(list(im_norm.shape))
        im_norm = torch.from_numpy(im_norm).permute(2, 0, 1)
        imgs.append(im_norm)
        l2c = cam_params[cam]['l2c']
        K = get_intrinsic(cam_params[cam]['intrinsic'])
        # training gs downsample 2x, recover scale
        rescale_K = np.copy(K)
        rescale_K[0, 0] *= 2
        rescale_K[0, 2] *= 2
        rescale_K[1, 1] *= 2
        rescale_K[1, 2] *= 2
        l2imgs.append(rescale_K @ l2c)
        intrinsics.append(K) 

    imgs = torch.stack(imgs)[None, ...]
    imgs = imgs.cuda()
    timestamp = info['timestamp']
    
    f2w = np.eye(4)
    f2w[:3, :3] = SCR.from_euler('XYZ', info['ego_rot']).as_matrix()
    f2w[:3, 3] = info['ego_pos']
    
    v2w = np.eye(4)
    v2w[:3, :3] = SCR.from_euler('XYZ', imu_rot).as_matrix()
    v2w[:3, 3] = OPENCV2IMU @ f2w[:3, 3]

    history.put(v2w[:3,3])
    while history.qsize() < 3:
        history.put(v2w[:3,3])
    if history.qsize() > 3:
        history.get()
    ego_his_trajs = np.stack(list(history.queue))
    ego_his_trajs -= v2w[:3,3]
    ego_his_trajs = np.dot(v2w[:3, :3], ego_his_trajs.T).T
    ego_his_trajs = ego_his_trajs[1:, :2] - ego_his_trajs[:-1, :2]
    ego_his_trajs = ego_his_trajs[None, None, :, [1,0]]

    forward_acc = info['accelerate']
    forward_velo = info['ego_velo']
    can_bus = np.zeros(18)
    can_bus[:3] = v2w[:3, 3]
    v_quat = SCR.from_euler('XYZ', imu_rot).as_quat()[[3,0,1,2]]
    can_bus[3:7] = v_quat
    # can_bus[7:10] = np.array([0.05252802768009661, 0.9291747528573647, 9.513756543139106])
    # rotation_rate
    can_bus[10:13] = np.array([0, 0, -info['steer_rate']])
    # vel
    can_bus[13:16] = np.array([forward_velo, 0, 0])
    yaw = imu_rot[2]
    # yaw = -info['ego_steer']
    if yaw < 0:
        yaw += np.pi * 2
    can_bus[-2] = yaw
    can_bus[-1] = yaw / np.pi * 180
    
    img_metas = {
        'scene_token': '062',
        'can_bus': can_bus,
        'img_shape': imgs_shape,
        'lidar2img': l2imgs,
        'intrinsics': intrinsics,
    }
    img_metas = [[img_metas]]
    cmd = [0,0,0]
    cmd[info['command']] = 1
    data = {
        'img': imgs, 
        'img_metas': img_metas,
        'ego_his_trajs': torch.from_numpy(ego_his_trajs).cuda(),
        'ego_fut_cmd': torch.tensor([[[cmd]]]).cuda(),
        'raw_imgs': raw_imgs,
    }

    return data

