import cv2 
import numpy as np
from open3d import geometry
import math
from dataparser import get_intrinsic
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import os
from glob import glob

nusc_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def draw_bev(pred_bboxes, out):
    rot_axis=2
    canvas_h, canvas_w = 400, 200
    zoom = 5
    canvas = np.ones((canvas_h, canvas_w, 3)) * 255
    for i in range(len(pred_bboxes)):
        center = pred_bboxes[i, 0:3]
        dim = pred_bboxes[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -pred_bboxes[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        center[rot_axis] += dim[rot_axis] / 2
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        box3d_verts = np.asarray(box3d.get_box_points())
        bev_box_verts = box3d_verts[[0, 1, 2, 7], :2] * zoom
        bev_box_verts[:, 0] = -bev_box_verts[:, 0] + canvas_w // 2
        bev_box_verts[:, 1] += canvas_h // 2
        bev_box_verts = bev_box_verts.astype(np.int)
        bev_box_verts = bev_box_verts[:, [1,0]]
        connects = [[0, 1], [1, 3], [3, 2], [2, 0]]
        for conn in connects:
            cv2.line(canvas, bev_box_verts[conn[0]], bev_box_verts[conn[1]], (255, 0, 0), 2)
    cv2.imwrite(out, canvas)

def draw_proj(results, raw_data, parse_data, out, cameras):
    obs, info, cam_params, _ = raw_data
    l2i_mats = parse_data['img_metas'][0][0]['lidar2img']

    # box3d  = results['boxes_3d_det']
    # scores = results['scores_3d_det'].numpy()
    # labels = results['labels_3d_det'].numpy()
    box3d = results['pts_bbox']['boxes_3d']
    scores = results['pts_bbox']['scores_3d']
    labels = results['pts_bbox']['labels_3d']

    box_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    box_yaw = -box_yaw - np.pi / 2

    imgs = []
    for cam, l2i in zip(cameras, l2i_mats):
        l2c = cam_params[cam]['l2c']
        K = get_intrinsic(cam_params[cam]['intrinsic'])
        im = obs['rgb'][cam]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # im = cv2.resize(im, (640, 380))

        for i in range(len(box3d)):
            if scores[i] < 0.2:
                continue
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*box3d.tensor[i, 7:9], 0.0)
            box = Box(
                box_center[i],
                box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            # import ipdb; ipdb.set_trace()
            # box.render_cv2(im, K @ l2c, normalize=True)

            box3d_verts = box.corners().T
            cam_verts = (l2c[:3, :3] @ box3d_verts.T).T + l2c[:3, 3]
            uvz_verts = (K[:3, :3] @ cam_verts.T).T
            # uvz_verts = (l2i[:3, :3] @ box3d_verts.T).T + l2i[:3, 3]
            uv_verts = uvz_verts[:, :2] / uvz_verts[:, 2][:, None]

            mask = (uvz_verts[:, 2] > 0) & (uv_verts[:, 0] >= 0) & (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < im.shape[0]) & (uv_verts[:, 0] < im.shape[1])
            if mask.sum() < 4:
                continue

            connections = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                            [0, 4], [1, 5], [2, 6], [3, 7]]

            for connection in connections:
                line = uv_verts[connection, :].astype(int).tolist()
                cv2.line(im, line[0], line[1], color=(0,0,255), thickness=1)

        imgs.append(im)

    cat_images = cv2.vconcat([
        cv2.hconcat([imgs[2], imgs[0], imgs[1]]),
        cv2.hconcat([imgs[5], imgs[3], imgs[4]]),
    ])
    
    cv2.imwrite(out, cat_images)
    
def save_frame(raw_image, traj_poses, cam_params, save_fn):
    plan_traj = np.stack([
        traj_poses[:, 0],
        1.5*np.ones_like(traj_poses[:, 0]),
        traj_poses[:, 1],
    ], axis=1)
    
    intrinsic_dict = cam_params['CAM_FRONT']['intrinsic']
    intrinsic = np.eye(4)
    fx = fov2focal(intrinsic_dict['fovx'], intrinsic_dict['W'])
    fy = fov2focal(intrinsic_dict['fovy'], intrinsic_dict['H'])
    intrinsic[0, 0], intrinsic[1, 1] = fx, fy
    intrinsic[0, 2], intrinsic[1, 2] = intrinsic_dict['cx'], intrinsic_dict['cy']
    uvz_traj = (intrinsic[:3, :3] @ plan_traj.T).T
    uv_traj = uvz_traj[:, :2] / uvz_traj[:, 2][:, None]
    uv_traj_int = uv_traj.astype(int)
    
    images = []
    for cam in nusc_cameras:
        im = raw_image[cam]
        if cam == 'CAM_FRONT':
            for point in uv_traj_int:
                im = cv2.circle(im, point, radius=5, color=(192, 152, 7), thickness=-1)
        images.append(im)
    front_cams = cv2.hconcat(images[:3])
    back_cams = cv2.hconcat(images[3:])
    images = cv2.cvtColor(cv2.vconcat([front_cams, back_cams]), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_fn+'.jpg', images)
    
def to_video(folder_path, fps=5, downsample=1):
    imgs_path = glob(os.path.join(folder_path, '*.jpg'))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height//downsample))
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
        
    # media.write_video(os.path.join(folder_path, 'video.mp4'), img_array, fps=10)
    mp4_path = os.path.join(folder_path, 'video.mp4')
    if os.path.exists(mp4_path): 
        os.remove(mp4_path)
    out = cv2.VideoWriter(
        mp4_path, 
        cv2.VideoWriter_fourcc(*'DIVX'), fps, size
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
