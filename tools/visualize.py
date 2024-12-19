from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import color_map
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
import numpy as np
import math
import os

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

# def format_results(det):
#     boxes = output_to_nusc_box(det)
#     plan_annos = [det['ego_fut_preds'], det['ego_fut_cmd']]
#     boxes = lidar_nusc_box_to_global()

# def render_sample_data(data):
#     bbox_anns = data['results'][sample_token]
#     for content in bbox_anns:
#         bbox_pred_list.append(CustomDetectionBox(
#             sample_token=content['sample_token'],
#             translation=tuple(content['translation']),
#             size=tuple(content['size']),
#             rotation=tuple(content['rotation']),
#             velocity=tuple(content['velocity']),
#             fut_trajs=tuple(content['fut_traj']),
#             ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
#             else tuple(content['ego_translation']),
#             num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
#             detection_name=content['detection_name'],
#             detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
#             attribute_name=content['attribute_name']))
#     pred_annotations = EvalBoxes()
#     pred_annotations.add_boxes(sample_token, bbox_pred_list)
#     # print('green is ground truth')
#     # print('blue is the predited result')
#     visualize_sample(nusc, sample_token, gt_annotations, pred_annotations,
#                      savepath=out_path, traj_use_perstep_offset=traj_use_perstep_offset, pred_data=data)
    # return

def traj_vis(results, out, data, front_im):
    plan_cmd = results[0]['pts_bbox']['ego_fut_cmd']
    plan_cmd = np.argmax(results[0]['pts_bbox']['ego_fut_cmd'])
    plan_traj = results[0]['pts_bbox']['ego_fut_preds'][plan_cmd]
    print(plan_traj)
    plan_traj[abs(plan_traj) < 0.01] = 0.0
    plan_traj = plan_traj.cumsum(axis=0).detach().cpu().numpy()
    plan_traj = np.concatenate((
        plan_traj[:, [0]],
        1.5*np.ones((plan_traj.shape[0], 1)),
        plan_traj[:, [1]],
    ), axis=1)
    # # add the start point in lcf
    # plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
    # # plan_traj[0, :2] = 2*plan_traj[1, :2] - plan_traj[2, :2]
    # plan_traj[0, 0] = 0.3
    # plan_traj[0, 1] = 1.0
    # plan_traj[0, 3] = 1.0

    img_metas = data['img_metas'][0][0]
    intrinsic_dict = img_metas['intrinsics'][0]
    intrinsic = np.eye(4)
    fx = fov2focal(intrinsic_dict['fovx'], intrinsic_dict['W'])
    fy = fov2focal(intrinsic_dict['fovy'], intrinsic_dict['H'])
    intrinsic[0, 0], intrinsic[1, 1] = fx, fy
    intrinsic[0, 2], intrinsic[1, 2] = intrinsic_dict['cx'], intrinsic_dict['cy']
    l2c = img_metas['lidar2img'][0]
    cam_traj = (l2c[:3, :3] @ plan_traj.T).T + l2c[:3, 3]
    uvz_traj = (intrinsic[:3, :3] @ cam_traj.T).T
    uv_traj = uvz_traj[:, :2] / uvz_traj[:, 2][:, None]
    uv_traj = np.stack((uv_traj[:-1], uv_traj[1:]), axis=1)
    
    plan_vecs = None
    for i in range(uv_traj.shape[0]):
        plan_vec_i = uv_traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, linestyles='solid', cmap=cmap)
    _, ax = plt.subplots(1, 1, figsize=(6, 12))
    ax.imshow(front_im/255.)
    ax.add_collection(line_segments)
    ax.set_xlim(0, front_im.shape[1])
    ax.set_ylim(front_im.shape[0], 0)
    ax.axis('off')

    plt.savefig(out)

    return uv_traj


# def vis(results, out):
    # format_results(results[0]['pts_bbox'])
    # render_sample_data(sample_token_list[id], pred_data=results, out_path=out)