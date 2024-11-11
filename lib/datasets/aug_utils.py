import torch
# import logging
# import os
import numpy as np
# import fnmatch
# from PIL import Image
# from libs.utils import batch_project
# from scipy.io import loadmat, savemat
# from torch.utils.data import Dataset
import imgaug.augmenters as iaa
# import imgaug as ia
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2
# from torchvision.ops import masks_to_boxes
import torchvision


def divide_box(bbox, n_range=(3,6), p_range=(0.25, 0.7), img_w=640, img_h=480):
    # bbox: size [4], format [x,y,w,h]
    n = torch.randint(n_range[0], n_range[1], (1,)).item()
    p = (p_range[1]-p_range[0])*torch.rand(1).item()+p_range[0]
    cells = torch.zeros(n, n, 2)
    occlude = torch.rand(n,n)<=p
    X = bbox[0]
    Y = bbox[1]
    W = bbox[2]
    H = bbox[3]
    if W%n != 0:
        W = W - W%n
    if H%n != 0:
        H = H - H%n
    assert W%n == 0
    assert H%n == 0
    assert X+W <= img_w, 'X: {}, W: {}, img_w: {}'.format(X, W, img_w)
    assert Y+H <= img_h, 'Y: {}, H: {}, img_h: {}'.format(Y, H, img_h)
    w = int(W/n)
    h = int(H/n)
    for i in range(n):
        for j in range(n):
            cells[i,j,0] = X + i*w
            cells[i,j,1] = Y + j*h
    return cells.view(-1,2).long(), occlude.view(-1), w, h

def get_patch_xy(num_patches, img_w, img_h, obj_bbox, cell_w, cell_h):
    patch_xy = torch.zeros(num_patches, 2)
    max_w = img_w - cell_w
    max_h = img_h - cell_h
    X = obj_bbox[0]
    Y = obj_bbox[1]
    XX = X + obj_bbox[2]
    YY = Y + obj_bbox[3]
    assert XX>X and X>=0 and XX<=img_w, 'X {}, XX {}, Y {}, YY {}, cell_w {}, cell_h {}, img_w {}, img_h {}.'.format(X, XX, Y, YY, cell_w, cell_h, img_w, img_h)
    assert YY>Y and Y>=0 and YY<=img_h, 'X {}, XX {}, Y {}, YY {}, cell_w {}, cell_h {}, img_w {}, img_h {}.'.format(X, XX, Y, YY, cell_w, cell_h, img_w, img_h)
    for i in range(num_patches):
        x = torch.randint(0, max_w-1, (1,))
        y = torch.randint(0, max_h-1, (1,))
        trial = 0
        # while x>=X and x<XX and y>=Y and y<YY:### select the patches inside the bbox
        while (x+cell_w<X or x>XX) and (y+cell_h<Y or y>YY):### select the patches outside the bbox
            x = torch.randint(0, max_w-1, (1,))
            y = torch.randint(0, max_h-1, (1,))
            trial += 1
            if trial > 1000:
                print('Can find patch! X {}, XX {}, Y {}, YY {}, cell_w {}, cell_h {}, img_w {}, img_h {}.'
                    .format(X, XX, Y, YY, cell_w, cell_h, img_w, img_h))
        patch_xy[i,0] = x
        patch_xy[i,1] = y
    return patch_xy

def get_bbox(pts2d, img_size, coco_format=False):
    W = img_size[-2]
    H = img_size[-1]
    # print("w",W,"h",H)
    xmin = int(max(pts2d[:,0].min().round().item()-15, 0))
    xmax = int(min(pts2d[:,0].max().round().item()+15, W))
    # assert xmax>xmin
    ymin = int(max(pts2d[:,1].min().round().item()-15, 0))
    ymax = int(min(pts2d[:,1].max().round().item()+15, H))
    # assert ymax>ymin
    if ymax<=ymin or xmax<=xmin:
        print(pts2d)
        print(xmin, ymin, xmax, ymax)
        return None
    if coco_format:
        return [xmin, ymin, xmax, ymax]
    else:
        return [xmin, ymin, xmax-xmin, ymax-ymin]

def get_bbox_from_mask(mask,path, coco_format=False):
    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(mask)
    # print(obj_ids)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = mask == obj_ids[:, None, None]
    # masks = mask == obj_ids[:, None]
    boxes = torchvision.ops.masks_to_boxes(masks)
    # print(boxes.size()[0])
    if boxes.size()[0]!=1:
        # print("wrong path",path)
        # np.save('wrong_mask.npy',mask.numpy())
        return None

    xmin, ymin, xmax, ymax = boxes[0].tolist()
    xmin, ymin, xmax, ymax = int(xmin),int(ymin),int(xmax),int(ymax)
    # assert xmax>xmin
    # assert ymax>ymin
    if ymax<=ymin or xmax<=xmin:
        return None
    if coco_format:
        return [xmin, ymin, xmax, ymax]
    else:
        return [xmin, ymin, xmax-xmin, ymax-ymin]

def check_if_inside(pts2d, x1, x2, y1, y2):
    r1 = pts2d[:, 0]-0.5 >= x1 -0.5
    r2 = pts2d[:, 0]-0.5 <= x2 -1 + 0.5
    r3 = pts2d[:, 1]-0.5 >= y1 -0.5
    r4 = pts2d[:, 1]-0.5 <= y2 -1 + 0.5
    return r1*r2*r3*r4

def obj_out_of_view(W, H, pts2d):
    xmin = pts2d[:,0].min().item()
    xmax = pts2d[:,0].max().item()
    ymin = pts2d[:,1].min().item()
    ymax = pts2d[:,1].max().item()
    if xmin>W or xmax<0 or ymin>H or ymax<0:
        return True
    else:
        return False

def occlude_obj(img,mask, bbox, p_white_noise=0.1, p_occlude=(0.25, 0.7)):
    # img: image tensor of size [3, h, w]
    _, img_h, img_w = img.size()
    # bbox = get_bbox_from_mask(mask,path)
    if bbox is None:
        return img, _
    cells, occ_cell, cell_w, cell_h = divide_box(bbox, p_range=p_occlude, img_w=img_w, img_h=img_h)
    '''
    cells: the divided cells (image patch)
    occ_cell: list of True/False represents whether the corresponding cells are occluded
    cell_w/h: width/height of each cells
    '''
    num_cells = cells.size(0)
    noise_occ_id = torch.rand(num_cells) <= p_white_noise   
    ### list of True/False represents whether the corresponding cells are noise occulued
    actual_noise_occ = noise_occ_id * occ_cell
    ### list of True/False represents whether the corresponding cells are real noise occulued
    num_patch_occ = occ_cell.sum() - actual_noise_occ.sum()
    patches_xy = get_patch_xy(num_patch_occ, img_w, img_h, bbox, cell_w, cell_h)
    j = 0
    for i in range(num_cells):
        if occ_cell[i]:
            x1 = cells[i,0].item()
            x2 = x1 + cell_w
            y1 = cells[i,1].item()
            y2 = y1 + cell_h

            mask[y1:y2, x1:x2] = 0

            # if vis is not None:
            #     vis = vis*(~check_if_inside(pts2d, x1, x2, y1, y2))

            if noise_occ_id[i]: # white_noise occlude
                img[:, y1:y2, x1:x2] = torch.rand(3, cell_h, cell_w) *255
            else: # patch occlude
                xx1 = patches_xy[j, 0].long().item()
                xx2 = xx1 + cell_w
                yy1 = patches_xy[j, 1].long().item()
                yy2 = yy1 + cell_h
                img[:, y1:y2, x1:x2] = img[:, yy1:yy2, xx1:xx2].clone()
                # img[:, yy1:yy2, xx1:xx2] = 255
                j += 1
    assert num_patch_occ == j
    return img, mask


def kps2tensor(kps):
    n = len(kps.keypoints)
    pts2d = np.array([kps.keypoints[i].coords for i in range(n)])
    return torch.tensor(pts2d, dtype=torch.float).squeeze()


def augment_lm(path,img, pts2d, mask, device):
    assert len(img.size()) == 3

    H, W = img.size()[-2:]
    # print('H',H,'W',W)
    bbox = get_bbox(pts2d, [W, H])
    if bbox is None:
        print(path)
    min_x_shift = int(-bbox[0]+10)              ###-xmin+10
    max_x_shift = int(W-bbox[0]-bbox[2]-10)     ###W-xmax-10
    min_y_shift = int(-bbox[1]+10)              ###-ymin+10
    max_y_shift = int(H-bbox[1]-bbox[3]-10)     ###H-ymax-10
    # assert max_x_shift > min_x_shift, 'path:{},max_x_shift:{}, min_x_shift:{},H: {}, W: {}, bbox: {}, {}, {}, {}'.format(path,max_x_shift,min_x_shift,H, W, bbox[0], bbox[1], bbox[2], bbox[3])
    # assert max_y_shift > min_y_shift, 'path:{},max_y_shift:{}, min_y_shift:{}, H: {}, W: {}, bbox: {}, {}, {}, {}'.format(path,max_y_shift,min_y_shift,H, W, bbox[0], bbox[1], bbox[2], bbox[3])
    if max_x_shift <= min_x_shift:
        max_x_shift, min_x_shift = 0,0
    if max_y_shift <= min_y_shift:
        max_y_shift, min_y_shift = 0,0

    img = img.permute(1,2,0).numpy()
    # mask = mask.permute(1,2,0).numpy()
    # print("img shape",img.shape)
    mask = mask.numpy()
    nkp = pts2d.shape[0]
    kp_list = [Keypoint(x=pts2d[i][0].item(), y=pts2d[i][1].item()) for i in range(nkp)] 
    kps = KeypointsOnImage(kp_list, shape=img.shape)
    mask = SegmentationMapsOnImage(mask, shape=img.shape)

    rotate = iaa.Affine(rotate=(-30, 30))
    scale = iaa.Affine(scale=(0.8, 1.2))
    trans = iaa.Affine(translate_px={"x": (min_x_shift, max_x_shift), "y": (min_y_shift, max_y_shift)})
    bright = iaa.MultiplyAndAddToBrightness(mul=(0.7, 1.3))
    hue_satu = iaa.MultiplyHueAndSaturation(mul_hue=(0.95,1.05), mul_saturation=(0.5,1.5))
    contrast = iaa.GammaContrast((0.8, 1.2))
    random_aug = iaa.SomeOf((3, 6), [rotate, trans, scale, bright, hue_satu, contrast])
    img1, kps1, mask1 = random_aug(image=img, keypoints=kps, segmentation_maps=mask)

    img1 = torch.tensor(img1).permute(2,0,1).to(device)
    pts2d1 = kps2tensor(kps1).to(device)
    # mask1 = torch.tensor(mask1.get_arr()).permute(2,0,1).to(device)
    mask1 = torch.tensor(mask1.get_arr()).to(device)

    bbox = get_bbox_from_mask(mask1,path)
    if bbox is None:
        np.save("wrong_img.npy",img1.permute(1,2,0).numpy())
        np.save("wrong_mask.npy",mask1.numpy())

    if pts2d1[:,0].min()>W or pts2d1[:,0].max()<0 or pts2d1[:,1].min()>H or pts2d1[:,1].max()<0 or bbox is None:
        img1 = torch.tensor(img).permute(2,0,1).to(device)
        pts2d1 = kps2tensor(kps).to(device)
        mask1 = torch.tensor(mask.get_arr()).to(device)

    return img1, pts2d1, mask1, bbox


def blackout(img, bbox):
    assert len(img.size()) == 3
    H, W = img.size()[-2:]
    # bbox = get_bbox_from_mask(mask,path)
    if bbox is None:
        return img
    x, y, w, h = bbox
    img2 = torch.zeros_like(img)
    # img2 = torch.ones_like(img)*255
    img2[:, y:y+h, x:x+w] = img[:, y:y+h, x:x+w].clone()
    return img2