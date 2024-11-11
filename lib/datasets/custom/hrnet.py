import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
# from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
from lib.datasets.aug_utils import augment_lm, occlude_obj,blackout
import random
import torch
from lib.config import cfg


class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        # print("annotaion file:",ann_file)
        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        print("num of images",self.img_ids.shape)
        # input()
        self._transforms = transforms
        self.cfg = cfg

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], anno['center_2d']], axis=0)    ###original code, add center point
        # kpt_2d = np.array(anno['fps_2d'])    ###original code, add center point

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)
        
        if self.split=='test':
            gt_pose = np.array(anno['pose'])
            # print("gt_pose",gt_pose.shape)
            return inp, kpt_2d, mask, path, gt_pose

        return inp, kpt_2d, mask, path, None

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask, path, gt_pose = self.read_data(img_id)
        # print('img',img.shape,'kpt_2d',kpt_2d.shape,'mask',mask.shape)
        ### cancel augmentation firstly!!
        
        ### convert PIL image to numpy array
        inp = np.asarray(img).astype(np.uint8)
        if self.split == 'train':
            inp, kpt_2d, mask = self.augment(inp, mask, kpt_2d,path)

        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        if self.split=='test':
            ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'meta': {}, 'path': path, 'pose':gt_pose, 'kpt_2d':kpt_2d}
        else:
            ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'meta': {}, 'path': path}

        return ret

    def __len__(self):
        return len(self.img_ids)
    
    def augment(self, img, mask, kpt_2d,path, p_occlude=0.6, p_blackout=0.2):
        img = torch.tensor(img).permute(2,0,1)###[3, 540, 960]
        mask = torch.tensor(mask)

        img1, pts2d, mask1, bbox = augment_lm(path,img, kpt_2d, mask ,'cpu')
        if bbox is not None:
            if torch.rand(1) < p_occlude:
                img1, mask1 = occlude_obj(img1,mask1,bbox, p_white_noise=0.4,p_occlude=(0.15, 0.5))
            if torch.rand(1) < p_blackout:
                img1 = blackout(img1, bbox)

        return img1.permute(1,2,0).numpy(), pts2d, mask1.numpy()
