from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
import cv2 as cv
import os


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        # print(inp.shape)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        # print(kpt_2d.shape)
        # print(kpt_2d)

        img_id = int(batch['img_id'][0])
        img_path = '/home/neurobeast/Documents/haozheng/clean-pvnet/data/result/custom_edge/'
        npy_path = '/home/neurobeast/Documents/haozheng/clean-pvnet/data/result/custom_edge_npy/'
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        if not os.path.exists(npy_path):
            os.mkdir(npy_path)
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        print('pose_gt',pose_gt)
        print('pose_pd',pose_pred)
        pose = {'gt':pose_gt,'pred':pose_pred}
        np.save(npy_path+'{}.npy'.format(img_id), pose)

        kpt_2d_gt = pvnet_pose_utils.project(kpt_3d, K, pose_gt)

        length = 15
        short = 4
        axis_3d = np.float32([[0, 0, 0], [short,0,0], [0,short,0], [0,0,length]]).reshape(-1,3)
        axis_2d_gt = pvnet_pose_utils.project(axis_3d, K, pose_gt)
        axis_2d_pred = pvnet_pose_utils.project(axis_3d, K, pose_pred)


        dpi = 100
        height, width, depth = inp.shape
        figsize = 2* width / float(dpi), 1.3*height / float(dpi)


        fig, (ax, bx) = plt.subplots(1, 2,figsize=figsize)
        # _, ax = plt.subplots(1)
        ax.imshow(inp,interpolation=None)
        # ax.set_aspect(height/width)
        ax.add_patch(patches.Polygon(xy=axis_2d_gt[[0, 1]], fill=False, linewidth=2, edgecolor='r'))
        ax.add_patch(patches.Polygon(xy=axis_2d_gt[[0, 2]], fill=False, linewidth=2, edgecolor='r'))
        ax.add_patch(patches.Polygon(xy=axis_2d_gt[[0, 3]], fill=False, linewidth=2, edgecolor='r'))
        ax.add_patch(patches.Polygon(xy=axis_2d_pred[[0, 1]], fill=False, linewidth=2, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=axis_2d_pred[[0, 2]], fill=False, linewidth=2, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=axis_2d_pred[[0, 3]], fill=False, linewidth=2, edgecolor='b'))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        bx.imshow(inp,interpolation=None)
        # bx.set_aspect(height/width)
        for i in range(kpt_2d.shape[0]):
            bx.add_patch(plt.Circle(kpt_2d[i], 5, color='b'))
        for i in range(kpt_2d_gt.shape[0]):
            bx.add_patch(plt.Circle(kpt_2d_gt[i], 5, color='r'))
        bx.axes.xaxis.set_visible(False)
        bx.axes.yaxis.set_visible(False)
        # plt.show()
        plt.savefig(img_path+'{}.png'.format(img_id),dpi=dpi)

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.savefig('test.jpg')
        plt.close(0)





