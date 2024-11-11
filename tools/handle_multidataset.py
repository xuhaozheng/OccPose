import os
from plyfile import PlyData
import numpy as np
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
from lib.utils import base_utils
import json
from lib.utils.pvnet import pvnet_pose_utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml


def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def sample_fps_points(data_root):
    ply_path = os.path.join(data_root, 'model.ply')
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def record_ann(model_meta, img_id, ann_id, images, annotations, split):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']

    dataset_range = os.path.join(data_root, 'ranges', '{}.txt'.format(split))
    dataset_range = np.loadtxt(dataset_range, dtype=str)
    print("npy path",os.path.join(data_root,'{}.npy'.format(split)))
    folders = np.load(os.path.join(data_root,'{}.npy'.format(split)),allow_pickle=True)
    print("number of folders", len(folders))
    for subfolder in folders: 
        subfolder_name = subfolder['folder_name']
        img_list = subfolder['img_list']
        K = subfolder['K']
    # for idx in range(5): 
        for img_path in img_list:
            # print(img_path)
            ds_path,folder_path, img_idx = img_path.split('/')
            folder_path = ds_path +'/'+folder_path
            # print(folder_path,img_idx)
            rgb_path = os.path.join(data_root,folder_path,'undistort', '{}.jpg'.format(img_idx))
            pose_path = os.path.join(data_root,folder_path,'homo', '{}.npy'.format(img_idx))
            mask_path = os.path.join(data_root,folder_path,'mask', '{}.jpg'.format(img_idx))
            # rgb = Image.open(rgb_path)
            # img_size = rgb.size
            img_size = [1280, 720]
            img_id += 1
            info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
            images.append(info)
            pose = np.load(pose_path)
            corner_2d = base_utils.project(corner_3d, K, pose)
            # print("corner2d",corner_2d.shape)
            center_2d = base_utils.project(center_3d[None], K, pose)
            # print("center 2d",center_2d)
            fps_2d = base_utils.project(fps_3d, K, pose)
            # fps_2d_gt_path = os.path.join(data_root,folder_path,'fps_x0', '{}.npy'.format(img_idx))
            # fps_2d_gt = np.load(fps_2d_gt_path)
            # print('from orignal fps 2d pts',fps_2d_gt,'reprojected 2d pts',fps_2d)

            ann_id += 1
            anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
            anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
            anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
            anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
            anno.update({'K': K.tolist(), 'pose': pose.tolist()})
            anno.update({'data_root': data_root})
            anno.update({'type': 'real', 'cls': 'cat'})
            annotations.append(anno)
            # input()

    return img_id, ann_id


def custom_to_coco(data_root,split):
    model_path = os.path.join(data_root, 'convert_Tube45mm_53mm.npy')

    model = np.load(model_path)
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.load(os.path.join(data_root, 'convert_Tube45mm_37mm_fps.npy'))

    model_meta = {
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations,split)
    categories = [{'supercategory': 'none', 'id': 1, 'name': 'shaft'}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    print("num of imgs",len(annotations))

    anno_path = os.path.join(data_root, '{}.json'.format(split))
    with open(anno_path, 'w') as f:
        json.dump(instance, f)

def generate_custom(data_root):
    # custom_to_coco(data_root,'train')
    # custom_to_coco(data_root,'test')
    custom_to_coco(data_root,'all')
    
