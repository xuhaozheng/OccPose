import os
from plyfile import PlyData
import numpy as np
# from lib.csrc.fps import fps_utils
# from lib.utils.linemod.opengl_renderer import OpenGLRenderer
# import tqdm
from PIL import Image
from lib.utils import base_utils
import json
# from lib.utils.pvnet import pvnet_pose_utils
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import yaml

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


def record_ann(model_meta, img_id, ann_id, images, annotations, split, add_occlusion=False):
    img_format = '.png'
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    # print('fps3d',fps_3d)

    dataset_range = os.path.join(data_root, 'ranges', '{}.txt'.format(split))
    dataset_range = np.loadtxt(dataset_range, dtype=str)
    print("total num of images", dataset_range.shape[0])
    for idx in range(dataset_range.shape[0]):
        # for idx in range(5):
        # ds_path,folder_path, img_idx = dataset_range[idx].split('/')
        # folder_path = ds_path +'/'+folder_path
        folder_path, img_idx = dataset_range[idx].split('/')
        config_path = data_root+'/'+folder_path+'/config.yaml'
        with open(config_path) as f_tmp:
            config = yaml.load(f_tmp, Loader=yaml.FullLoader)
        K = np.array(config['cam']['camera_matrix']['data'],
                     dtype=np.float32).reshape((3, 3))
        # print(K)
        rgb_path = os.path.join(data_root, folder_path,
                                'undistort', '{}{}'.format(img_idx, img_format))
        pose_path = os.path.join(
            data_root, folder_path, 'homo', '{}.npy'.format(img_idx))
        mask_path = os.path.join(
            data_root, folder_path, 'mask', '{}{}'.format(img_idx, img_format))
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path,
                'height': img_size[1], 'width': img_size[0], 'id': img_id}
        # print(info)
        images.append(info)
        # print(rgb_path)
        pose = np.load(pose_path)
        # print('pose',pose)
        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)
        fps_2d = base_utils.project(fps_3d, K, pose)
        # fps_2d_gt_path = os.path.join(
        #     data_root, folder_path, 'fps', '{}.npy'.format(img_idx))
        # fps_2d_gt = np.load(fps_2d_gt_path)
        # print('from orignal fps 2d pts', fps_2d_gt.reshape(-1, 2))
        # print('reprojected 2d pts', fps_2d.reshape(-1, 2))
        # input()
        # print("center 2d",center_2d)
        if add_occlusion:
            mask_occ_path = os.path.join(
                data_root, folder_path, 'mask_occ', '{}{}'.format(img_idx, img_format))   
            mask_occ_per =  np.load(os.path.join(
                data_root, folder_path, 'mask_per', '{}.npy'.format(img_idx)))   
            # print(mask_occ_per)
            # print(mask_occ_per.tolist())
        # input()      


        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id,
                'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(),
                    'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(),
                    'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': data_root})
        anno.update({'type': 'real', 'cls': 'cat'})
        if add_occlusion:
            # anno.update({'mask_occ_path': mask_occ_path, 'mask_per': mask_occ_per.tolist()})
            anno.update({'mask_path': mask_occ_path, 'mask_per': mask_occ_per.tolist()})
        annotations.append(anno)

    return img_id, ann_id


def custom_to_coco(data_root, split):
    config_path = data_root+'/config.yaml'
    with open(config_path) as f_tmp:
        config = yaml.load(f_tmp, Loader=yaml.FullLoader)
    K = np.array(config['cam']['camera_matrix']['data'],
                    dtype=np.float32).reshape((3, 3))

    # model_path = os.path.join(data_root, 'LND_cut_notip.npy')
    model_path = os.path.join(data_root, config['dataset']['3d_model'])
    # model_path = os.path.join(data_root, 'PG_cut_notip_short.npy')


    model = np.load(model_path)
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.load(os.path.join(data_root, config['dataset']['kp_model']))

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

    img_id, ann_id = record_ann(
        model_meta, img_id, ann_id, images, annotations, split, add_occlusion=False)
    print("num of annotations", len(annotations))
    categories = [{'supercategory': 'none', 'id': 1, 'name': 'shaft'}]
    instance = {'images': images,
                'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, '{}.json'.format(split))
    with open(anno_path, 'w') as f:
        json.dump(instance, f)


def generate_custom(data_root):
    custom_to_coco(data_root, 'trainAB')
    custom_to_coco(data_root, 'testC')
    # custom_to_coco(data_root,'all')
    # custom_to_coco(data_root,'all_ABC')
    # custom_to_coco(data_root,'train_occlusion500')
