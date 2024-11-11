from operator import gt
from lib.config import cfg, args
import numpy as np
import os
from lib.utils.pvnet import visualize_utils
import time
from PIL import Image
import yaml
from lib.utils.pvnet import pvnet_pose_utils
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket

class ToTensor(object):

    def __call__(self, img):
        return np.asarray(img).astype(np.float32) / 255.
    
class Normalize(object):

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img):
        img -= self.mean
        img /= self.std
        if self.to_bgr:
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img

def unpack_homo(homo):
    R = homo[:,:3][:3]
    t = homo[:,-1][:3]
    return R,t

# Set up client to send pose data
def setup_client(host='localhost', port=12345):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    return client_socket


def vis_pose(pose_pred, gt_pose, K, im, require_gt=False):
    length = 15
    short = 4
    thickness = 2
    axis_3d = np.float32([[0, 0, 0], [length,0,0], [0,short,0], [0,0,short]]).reshape(-1,3)
    axis_2d_pred = pvnet_pose_utils.project(axis_3d, K, pose_pred)
    # im = cv2.imread(rgb_path)
    if require_gt:
        axis_2d_gt = pvnet_pose_utils.project(axis_3d, K, gt_pose)
        im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[1,0]),int(axis_2d_gt[1,1])), (255,0,0), thickness, cv2.LINE_AA)
        im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[2,0]),int(axis_2d_gt[2,1])), (255,0,0), thickness, cv2.LINE_AA)
        im = cv2.line(im, (int(axis_2d_gt[0,0]),int(axis_2d_gt[0,1])), (int(axis_2d_gt[3,0]),int(axis_2d_gt[3,1])), (255,0,0), thickness, cv2.LINE_AA)
    im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[1,0]),int(axis_2d_pred[1,1])), (255,0,0), thickness, cv2.LINE_AA)
    im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[2,0]),int(axis_2d_pred[2,1])), (0,255,0), thickness, cv2.LINE_AA)
    im = cv2.line(im, (int(axis_2d_pred[0,0]),int(axis_2d_pred[0,1])), (int(axis_2d_pred[3,0]),int(axis_2d_pred[3,1])), (0,0,255), thickness, cv2.LINE_AA)
    cv2.imshow('Estimated Pose', im)
    cv2.waitKey(10)

def vis_pose_3d(pose_pred):
    rot_max, tvec = unpack_homo(pose_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original point
    ax.scatter(tvec[0], tvec[1], tvec[2], color='r', s=100)

    # Define the axis length
    axis_length = 1.0

    # Plot the orientation
    # origin = np.array([tvec[0], tvec[1], tvec[2]])
    origin = np.array([2.5, 2.5, 2.5])
    x_axis = origin + rot_max @ np.array([axis_length, 0, 0])
    y_axis = origin + rot_max @ np.array([0, axis_length, 0])
    z_axis = origin + rot_max @ np.array([0, 0, axis_length])

    ax.quiver(*origin, *(x_axis-origin), color='r', length=axis_length, normalize=True)
    ax.quiver(*origin, *(y_axis-origin), color='g', length=axis_length, normalize=True)
    ax.quiver(*origin, *(z_axis-origin), color='b', length=axis_length, normalize=True)

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set limits
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])

    plt.show()


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

def get_kpt3d(model_path, fps_path):
    model = np.load(model_path)
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.load(fps_path)
    kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)
    return kpt_3d

def run_inference_folder():
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from torchvision import models, transforms
    import torch
    import warnings
    import glob
    from natsort import natsorted
    warnings.filterwarnings("ignore")
    preprocess = transforms.Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
    # Connect to server
    use_server = True
    if use_server:
        client_socket = setup_client()

    try:
        network = make_network(cfg).cuda()
        load_network(network, cfg.model_dir,esume=cfg.resume, epoch=cfg.test.epoch)
        network.eval()

        model_path = cfg.cad_model_path
        fps_path = cfg.fps_path

        kpt_3d = get_kpt3d(model_path, fps_path)
        config_path = '{}/config.yaml'.format(cfg.infer_dir)
        with open(config_path) as f_tmp:
            config =  yaml.load(f_tmp, Loader=yaml.FullLoader)
        K = np.array(config['cam']['camera_matrix']['data'],dtype=np.float32).reshape((3,3))


        img_folder = cfg.infer_dir
        infer_imgs = natsorted(glob.glob(os.path.join(img_folder,cfg.img_format)))

        for rgb_path in infer_imgs:

            gt_pose=None
            
            input_image = Image.open(rgb_path)
            input_tensor = torch.from_numpy(preprocess(input_image)).cuda()
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = network(input_batch)
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

            if use_server:
                data = pose_pred.tobytes()
                client_socket.sendall(data)

            im = cv2.imread(rgb_path)
            vis_pose(pose_pred, gt_pose, K, im, require_gt=False)
    except KeyboardInterrupt:
        pass
    finally:
        if use_server:
            client_socket.close()

def run_cam():
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from torchvision import models, transforms
    import torch
    import warnings
    import cv2
    import numpy as np
    warnings.filterwarnings("ignore")
    preprocess = transforms.Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    '''
    model configuration
    '''
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir,esume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    # Connect to server
    use_server = False
    if use_server:
        client_socket = setup_client()

    model_path = cfg.cad_model_path
    fps_path = cfg.fps_path

    kpt_3d = get_kpt3d(model_path, fps_path)
    config_path = '{}/config.yaml'.format(cfg.infer_dir)
    with open(config_path) as f_tmp:
        config =  yaml.load(f_tmp, Loader=yaml.FullLoader)
    K = np.array(config['cam']['camera_matrix']['data'],dtype=np.float32).reshape((3,3))

    ### Camera Configuration
    width = 1280
    height = 720
    cam = cv2.VideoCapture(0)  # camera index (default = 0) (added based on Randyr's comment).

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, height)

    # Lets check start/open your cam!
    if cam.read() == False:
        cam.open()

    if not cam.isOpened():
        print('Cannot open camera')

    try:
        while True:
            ret,frame = cam.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(img)

            input_tensor = torch.from_numpy(preprocess(input_image)).cuda()
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = network(input_batch)
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

            if use_server:
                data = pose_pred.tobytes()
                client_socket.sendall(data)

            vis_pose(pose_pred, None, K, frame, require_gt=False)

            if cv2.waitKey(10)&0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
    finally:
        if use_server:
            client_socket.close()







    

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    globals()['run_'+args.type]()
