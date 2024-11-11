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
import cv2


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
    K = model_meta['K']

    dataset_range = os.path.join(data_root, 'ranges', '{}.txt'.format(split))
    dataset_range = np.loadtxt(dataset_range, dtype=str)
    for idx in range(dataset_range.shape[0]): 
    # for idx in range(5): 
        folder_path, img_idx = dataset_range[idx].split('/')
        # print(folder_path,img_idx)
        rgb_path = os.path.join(data_root,folder_path,'images', '{}.jpg'.format(img_idx))
        pose_path = os.path.join(data_root,folder_path,'homo_x0', '{}.npy'.format(img_idx))
        mask_path = os.path.join(data_root,folder_path,'mask', '{}.jpg'.format(img_idx))
        # print(rgb_path)
        # print(pose_path)
        # print(mask_path)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)
        pose = np.load(pose_path)
        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]
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

    return img_id, ann_id

def generate_from_clips(data_root):
    model_path = os.path.join(data_root, 'convert_Tube45mm_53mm.npy')

    model = np.load(model_path)
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.load(os.path.join(data_root, 'convert_Tube45mm_37mm_fps.npy'))

    img_id = 0
    images = []

    # epoch=44
    # epochs = [19,24,29,34,39,44]
    epochs = [15]
    for epoch in epochs:
        print("epoch",epoch)

        # dataset_range = os.path.join(data_root, 'inference/charuco_5d_inference','refine_dataset_44', 'clips.npy')
        dataset_range = '/home/neurobeast/Documents/haozheng/clean-pvnet/data/clips/clips_ABC.npy'
        dataset_range = np.load(dataset_range,allow_pickle=True)

        # pd_data_root = 'data/shaft_charuco_45mm_kp_ested/'
        # pd_data_root = 'data/inference/charuco_5d_inference/refine_dataset_{}'.format(epoch)
        pd_data_root = 'data/inference/charuco_5d_occlusion500_inference/refine_dataset_{}'.format(epoch)

        good_frames = 0
        abnormal_frames = 0

        clip_list = []
        idx_list = []
        num_frame_list=[]
        dist_list= []
        for clip_file in dataset_range:
            # print(len(dataset_range[key]))
            # print(clip_file.keys())
            img_list = clip_file['img_list']
            poses_dict ={}
            K = clip_file['K']
            if len(img_list)>30:
                # print("folder name",clip_file['folder_name'])
                gt = []
                pd = []
                for img_path in img_list: 
                    # print(img_path)
                    ds_name, folder_path, img_idx = img_path.split('/')
                    folder_path = os.path.join(ds_name,folder_path)
                    gt_pose_path = os.path.join(data_root,folder_path,'homo_x0', '{}.npy'.format(img_idx))
                    gt_pose = np.load(gt_pose_path)
                    center_2d = base_utils.project(center_3d[None], K, gt_pose)[0].reshape((1,2))
                    fps_2d = base_utils.project(fps_3d, K, gt_pose)
                    # print('fps2d',fps_2d.shape,'center2d',center_2d.shape)
                    kpt_2d_gt = np.concatenate([fps_2d, center_2d], axis=0)  ###original code, add center point
                    pd_pose_path = os.path.join(pd_data_root,folder_path,'pd_pose', '{}.npy'.format(img_idx))
                    kpt_2d_pd = np.load(pd_pose_path).reshape(-1,) 

                    # if folder_path =='shaft_keydots/exp0823B' and img_idx =='529':
                    #     print(kpt_2d_pd)
                    #     # print(pose_pred)
                    #     print("cam",K)
                    #     input()
                    # print("gt",kpt_2d_gt.shape)
                    # print("pd",kpt_2d_pd.shape)
                    gt.append(kpt_2d_gt)
                    pd.append(kpt_2d_pd)
                gt = np.array(gt).reshape(-1,20)
                pd = np.array(pd).reshape(-1,20)
                dist = np.linalg.norm(gt-pd,axis=1).mean()
                # print("gt",gt.shape)
                # print("pd",pd.shape)
                # print(clip_file['idx'])
                # print("number of clips",len(img_list))
                # print("dist",dist)
                idx_list.append(clip_file['idx'])
                num_frame_list.append(len(img_list))
                dist_list.append(dist)
                if dist<1500:
                    good_frames+=len(img_list)
                else:
                    abnormal_frames+=len(img_list)
                poses_dict['gt'] = gt
                poses_dict['pd'] = pd
                poses_dict['paths'] = img_list
                poses_dict['K'] = K
                poses_dict['idx'] = clip_file['idx']
                clip_list.append(poses_dict)
        # print(poses_dict)
        print("total num of good frame: {} | abnormal frames: {}".format(good_frames,abnormal_frames))
        evalute_dict = {}
        evalute_dict['idxs']=idx_list
        evalute_dict['num_frames']=num_frame_list
        evalute_dict['dist']=dist_list
        evalute_dict['avg_dist']=np.average(dist_list)
        evalute_dict['good_frames']=good_frames
        evalute_dict['abnormal_frames'] = abnormal_frames
        print("avg dist",evalute_dict['avg_dist'])
        # print("evalute dict",evalute_dict)
        np.save(os.path.join(pd_data_root, 'evaluate_2ds.npy'),evalute_dict)
        np.save(os.path.join(pd_data_root, 'poses_2ds.npy'),clip_list)

def evaluation_analyze(data_root):
    # epochs = [4,9,19,24,29,34,39,44]
    epochs = [4,9,14,19,24]
    import matplotlib.pyplot as plt
    import numpy as np


    for epoch in epochs:
        print("epoch",epoch)

        # pd_data_root = 'data/shaft_charuco_45mm_kp_ested/'
        pd_data_root = 'data/refine_dataset_{}'.format(epoch)
        evalute_dict_path = os.path.join(pd_data_root,'evaluate.npy')
        evalute_dict = np.load(evalute_dict_path,allow_pickle=True).item()
        # print(evalute_dict)
        dist_list = evalute_dict['dist']
        idx_list = evalute_dict['idxs']
        # for i in range(len(dist_list)):
            # print("{} | {}".format(idx_list[i],dist_list[i]))
        plt.plot(idx_list, dist_list,label="epoch {}".format(epoch))
    plt.legend()
    plt.show()
        

def visualize_from_prediction(data_root):
    model_path = os.path.join(data_root, 'convert_Tube45mm_53mm.npy')

    model = np.load(model_path)
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.load(os.path.join(data_root, 'convert_Tube45mm_37mm_fps.npy'))
    kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)

    img_id = 0
    images = []

    dataset_range = os.path.join(data_root, 'refine_dataset_44', 'clips.npy')
    dataset_range = np.load(dataset_range,allow_pickle=True)

    epochs = [4,9,19,24]
    for epoch in epochs:
        pd_data_root = 'data/refine_dataset_{}'.format(epoch)
        save_img_folder_path = 'data/test_result/initial_pred_{}'.format(epoch)

        good_frames = 0
        abnormal_frames = 0

        clip_list = []
        for clip_file in dataset_range:
            # print(len(dataset_range[key]))
            # print(clip_file.keys())
            img_list = clip_file['img_list']
            poses_dict ={}
            K = clip_file['K']
            clip_idx = str(clip_file['idx'])
            if len(img_list)>30:
                gt = []
                pd = []
                file_paths = []
                if not os.path.exists(os.path.join(save_img_folder_path,clip_idx)):
                    os.makedirs(os.path.join(save_img_folder_path,clip_idx))
                for img_path in img_list: 
                    # print(img_path)
                    ds_name, folder_path, img_idx = img_path.split('/')
                    folder_path = os.path.join(ds_name,folder_path)
                    gt_pose_path = os.path.join(data_root,folder_path,'homo_x0', '{}.npy'.format(img_idx))
                    gt_pose = np.load(gt_pose_path)
                    rgb_path = os.path.join(data_root,folder_path,'undistort', '{}.jpg'.format(img_idx))
                    im = cv2.imread(rgb_path)
                    center_2d = base_utils.project(center_3d[None], K, gt_pose)[0].reshape((1,2))
                    fps_2d = base_utils.project(fps_3d, K, gt_pose)
                    # print('fps2d',fps_2d.shape,'center2d',center_2d.shape)
                    kpt_2d_gt = np.concatenate([fps_2d, center_2d], axis=0)  ###original code, add center point
                    pd_pose_path = os.path.join(pd_data_root,folder_path,'pd_pose', '{}.npy'.format(img_idx))
                    kpt_2d_pd = np.load(pd_pose_path)
                    pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_pd, K)

                    gt.append(kpt_2d_gt)
                    pd.append(kpt_2d_pd)

                    length = 15
                    short = 4
                    axis_3d = np.float32([[0, 0, 0], [length,0,0], [0,short,0], [0,0,short]]).reshape(-1,3)
                    axis_2d_gt = pvnet_pose_utils.project(axis_3d, K, gt_pose)
                    axis_2d_pred = pvnet_pose_utils.project(axis_3d, K, pose_pred)

                    thickness =2

                    im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[1,0]),int(axis_2d_gt[1,1])), (255,0,0), thickness, cv2.LINE_AA)
                    im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[2,0]),int(axis_2d_gt[2,1])), (255,0,0), thickness, cv2.LINE_AA)
                    im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[3,0]),int(axis_2d_gt[3,1])), (255,0,0), thickness, cv2.LINE_AA)
                    im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[1,0]),int(axis_2d_pred[1,1])), (0,255,0), thickness, cv2.LINE_AA)
                    im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[2,0]),int(axis_2d_pred[2,1])), (0,255,0), thickness, cv2.LINE_AA)
                    im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[3,0]),int(axis_2d_pred[3,1])), (0,255,0), thickness, cv2.LINE_AA)
                    # cv2.imshow('Estimated Pose', im)
                    # cv2.waitKey(0)
                    save_img_path = os.path.join(save_img_folder_path,clip_idx,'{}.jpg'.format(img_idx))
                    cv2.imwrite(save_img_path,im)
        
                gt = np.array(gt).reshape(-1,20)
                pd = np.array(pd).reshape(-1,20)
                dist = np.linalg.norm(gt-pd,axis=1).mean()
                # print("gt",gt.shape)
                # print("pd",pd.shape)
                print(clip_file['idx'])
                print("number of clips",len(img_list))
                print("dist",dist)
                if dist<1500:
                    good_frames+=len(img_list)
                else:
                    abnormal_frames+=len(img_list)
                # poses_dict['gt'] = gt
                # poses_dict['pd'] = pd
                # poses_dict['paths'] = file_paths
                # poses_dict['K'] = K
                # clip_list.append(poses_dict)
        # print(poses_dict)
        print("total num of good frame: {} | abnormal frames: {}".format(good_frames,abnormal_frames))
        # np.save(os.path.join(pd_data_root, 'poses.npy'),poses_dict)




def visualize_opencv(data_root, pose_flag):
    model_path = os.path.join(data_root, 'convert_Tube45mm_53mm.npy')
    config_path = data_root+'/config.yaml'
    with open(config_path) as f_tmp:
        config =  yaml.load(f_tmp, Loader=yaml.FullLoader)
    K = np.array(config['cam']['camera_matrix']['data'],dtype=np.float32).reshape((3,3))

    model = np.load(model_path)
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.load(os.path.join(data_root, 'convert_Tube45mm_37mm_fps.npy'))
    kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)

    img_id = 0
    images = []

    pd_data_root = 'data/shaft_charuco_45mm_test_kp_est/'
    refine_data_root = 'data/shaft_charuco_45mm_test_kp_est/predicted'
    img_save_root = 'data/shaft_charuco_45mm_test_pred/'

    dataset_range = os.path.join(pd_data_root, 'poses20.npy')
    dataset_range = np.load(dataset_range, allow_pickle=True).item()

    gt_list = []
    pred_list = []
    refine_list = []

    keys = list(dataset_range.keys())


    poses_dict = {}
    for key in keys[7:]:
        print('key',key)
        clip = dataset_range[key]
        file_paths = clip['paths']
        print(len(file_paths))
        gt = clip['gt']
        pd = clip['pd']
        # print(gt.shape)
        # print(pd.shape)
        refine_path = refine_data_root+'/{}.npy'.format(key)
        refine_pose = np.load(refine_path)
        # print(refine_pose.shape)
        for i in range(len(file_paths)):
            folder_path, img_idx = file_paths[i].split('/')
            # print(folder_path,img_idx)
            rgb_path = os.path.join(data_root,folder_path,'undistort', '{}.jpg'.format(img_idx))
            im = cv2.imread(rgb_path)
            pose_path = os.path.join(data_root,folder_path,'homo_x0', '{}.npy'.format(img_idx))
            save_img_folder_path = os.path.join(img_save_root,folder_path,pose_flag)
            if not os.path.exists(save_img_folder_path):
                os.makedirs(save_img_folder_path)
            save_img_path = os.path.join(save_img_folder_path,'{}.jpg'.format(img_idx))
            gt_pose = np.load(pose_path)
            inp = Image.open(rgb_path)
            kpt_gt = gt[i]
            pose_gt = pvnet_pose_utils.pnp(kpt_3d, kpt_gt, K)
            kpt_pred = pd[i]
            # print("kpt_3d",kpt_3d.shape)
            # print("kpt_pred",kpt_pred.shape)
            pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_pred, K)
            # print(pose_pred)
            kpt_refine = refine_pose[i]
            pose_refine = pvnet_pose_utils.pnp(kpt_3d, kpt_refine, K)
            # print(pose_refine)
            gt_list.append(pose_gt)
            pred_list.append(pose_pred)
            refine_list.append(pose_refine)


            length = 15
            short = 4
            axis_3d = np.float32([[0, 0, 0], [length,0,0], [0,short,0], [0,0,short]]).reshape(-1,3)
            axis_2d_gt = pvnet_pose_utils.project(axis_3d, K, pose_gt)
            axis_2d_pred = pvnet_pose_utils.project(axis_3d, K, pose_pred)
            axis_2d_refine = pvnet_pose_utils.project(axis_3d, K, pose_refine)

            thickness =2


            if pose_flag == 'gt':###B
                im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[1,0]),int(axis_2d_gt[1,1])), (255,0,0), thickness, cv2.LINE_AA)
                im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[2,0]),int(axis_2d_gt[2,1])), (255,0,0), thickness, cv2.LINE_AA)
                im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[3,0]),int(axis_2d_gt[3,1])), (255,0,0), thickness, cv2.LINE_AA)
            # if pose_flag == 'pd':###G
            if pose_flag == 'refine':###R
                im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[1,0]),int(axis_2d_pred[1,1])), (0,255,0), thickness, cv2.LINE_AA)
                im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[2,0]),int(axis_2d_pred[2,1])), (0,255,0), thickness, cv2.LINE_AA)
                im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[3,0]),int(axis_2d_pred[3,1])), (0,255,0), thickness, cv2.LINE_AA)
                im = cv2.line(im, (int(axis_2d_refine[0,0]),int(axis_2d_refine[0,1])), (int(axis_2d_refine[1,0]),int(axis_2d_refine[1,1])), (0,0,255), thickness, cv2.LINE_AA)
                im = cv2.line(im, (int(axis_2d_refine[0,0]),int(axis_2d_refine[0,1])), (int(axis_2d_refine[2,0]),int(axis_2d_refine[2,1])), (0,0,255), thickness, cv2.LINE_AA)
                im = cv2.line(im, (int(axis_2d_refine[0,0]),int(axis_2d_refine[0,1])), (int(axis_2d_refine[3,0]),int(axis_2d_refine[3,1])), (0,0,255), thickness, cv2.LINE_AA)
            # cv2.imshow('Estimated Pose', im)
            # cv2.waitKey(0)
            cv2.imwrite(save_img_path,im)

def visualize_kp(data_root):
    model_path = os.path.join(data_root, 'convert_Tube45mm_53mm.npy')

    model = np.load(model_path)
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.load(os.path.join(data_root, 'convert_Tube45mm_37mm_fps.npy'))

    img_id = 0
    images = []

    # epoch=44
    # epochs = [19,24,29,34,39,44]
    epochs = [9]
    for epoch in epochs:
        print("epoch",epoch)

        dataset_range = os.path.join(data_root, 'refine_dataset_44', 'clips.npy')
        dataset_range = np.load(dataset_range,allow_pickle=True)

        # pd_data_root = 'data/shaft_charuco_45mm_kp_ested/'
        pd_data_root = 'data/refine_dataset_{}'.format(epoch)

        good_frames = 0
        abnormal_frames = 0

        clip_list = []
        idx_list = []
        num_frame_list=[]
        dist_list= []
        for clip_file in dataset_range:
            # print(len(dataset_range[key]))
            # print(clip_file.keys())
            img_list = clip_file['img_list']
            poses_dict ={}
            K = clip_file['K']
            for img_path in img_list: 
                # print(img_path)
                ds_name, folder_path, img_idx = img_path.split('/')
                folder_path = os.path.join(ds_name,folder_path)
                rgb_path = os.path.join(data_root,folder_path,'undistort', '{}.jpg'.format(img_idx))
                im = cv2.imread(rgb_path)
                gt_pose_path = os.path.join(data_root,folder_path,'homo_x0', '{}.npy'.format(img_idx))
                gt_pose = np.load(gt_pose_path)
                center_2d = base_utils.project(center_3d[None], K, gt_pose)[0].reshape((1,2))
                fps_2d = base_utils.project(fps_3d, K, gt_pose)
                print('fps2d',fps_2d.shape,'center2d',center_2d.shape)
                kpt_2d_gt = np.concatenate([fps_2d, center_2d], axis=0)  ###original code, add center point

                im = cv2.resize(im,(900,600),interpolation = cv2.INTER_AREA)

                img_shape = np.array([720,1280])

                fps_2d = (fps_2d-img_shape/2)/(img_shape/2) # normalization
                fps_2d = fps_2d*900

                for i in range(fps_2d.shape[0]):
                    pt_x = int(fps_2d[i,0])
                    pt_y = int(fps_2d[i,1])
                    # use the BGR format to match the original image type
                    cv2.circle(im,(pt_x, pt_y), 2, 255, 2)
                cv2.imshow('Estimated Pose', im)
                cv2.waitKey(0)


def generate_custom(data_root):
    generate_from_clips(data_root)
    # visualize_from_prediction(data_root)
    # visualize_opencv(data_root, 'gt')
    # visualize_opencv(data_root, 'pd')
    # visualize_opencv(data_root, 'refine')
    # evaluation_analyze(data_root)
    # visualize_kp(data_root)
