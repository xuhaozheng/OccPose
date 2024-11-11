from operator import gt
from lib.config import cfg, args
import numpy as np
import os
from lib.utils.pvnet import visualize_utils


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'], batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from torchviz import make_dot

    torch.manual_seed(0)
    epochs = [44,54,64,74,84,94,104]
    # epochs = [164]
    # for i in range(5,10):
    for epoch in epochs:
        path_list = []

        network = make_network(cfg).cuda()
        # load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
        print("model dir", cfg.model_dir)
        evaluate_root_path = 'data/evaluate'
        root_path = 'data/model/pvnet/'
        # root_path = 'data/model/hrnet/'
        model_name = 'costom_PG1'
        # model_name = 'costom_LND_hrnet1'
        # model_name = 'costom_LND1'
        model_dir = root_path+model_name
        load_network(network, model_dir, epoch=epoch)
        #load_network(network, cfg.model_dir, epoch=epoch)
        network.eval()
        save_path = os.path.join(evaluate_root_path,model_name,'epoch_{}'.format(epoch))

        if not os.path.exists(save_path):
                os.makedirs(save_path)

        # data_loader = make_data_loader(cfg, is_train=False)
        data_loader = make_data_loader(cfg, is_train=False, inference=True)
        evaluator = make_evaluator(cfg)
        test_num = 0
        for batch in tqdm.tqdm(data_loader):
            # if test_num > 10:
                # break
            data_path = batch['path'][0]
            inp = batch['inp'].cuda()
            with torch.no_grad():
                output = network(inp)
                # print(output)
            evaluator.evaluate(output, batch)
            path_list.append(data_path)
            test_num += 1
        evaluator.save(save_path)
        evaluator.summarize(save_path)
        np.save(save_path+"/path_list.npy",path_list)


def run_model_vis():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from torchviz import make_dot
    from torchsummary import summary

    epoch = 4

    network = make_network(cfg).cuda()
    # load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    print("model dir", cfg.model_dir)
    evaluate_root_path = 'data/evaluate'
    root_path = 'data/model/pvnet/'
    # model_dir = root_path+'custom_charuco6d'
    model_name = 'costom_LND_aug2'
    model_dir = root_path+model_name
    load_network(network, model_dir, epoch=epoch)
    #load_network(network, cfg.model_dir, epoch=epoch)
    network.eval()

    # data_loader = make_data_loader(cfg, is_train=False)
    test = 0
    data_loader = make_data_loader(cfg, is_train=False, inference=True)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            print(inp.size())
            # (3, 540, 960)
            summary(network, (3, 540, 960))
            # output = network(inp)
            # make_dot(output, params=dict(list(network.named_parameters()))).render("pvnet", format="png")
            # print(output)
        break


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir,
                 resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
            # print(output)
        visualizer.visualize(output, batch)

def run_inference():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    import warnings
    warnings.filterwarnings("ignore")
    '''
    Generate Pose Estimation for Refinement Module Training
    '''
    # epoch = 44
    epochs = [74]
    for epoch in epochs:
        root_path = 'data/model/pvnet/'
        model_name = 'costom_LND_aug4'
        model_dir = root_path+model_name
        network = make_network(cfg).cuda()
        load_network(network, model_dir, resume=cfg.resume, epoch=epoch)
        network.eval()
        # evaluator = make_evaluator(cfg)
        data_loader = make_data_loader(cfg, is_train=False, inference=True)
        # num_test=0
        for batch in tqdm.tqdm(data_loader):
            # if num_test>10:
            # break
            folder_path = batch['path'][0].split('/')
            # print(folder_path)
            folder_path[0] = 'data/inference/{}_inference_rect/refine_dataset_{}'.format(model_name,
                epoch)
            folder_path[-2] = 'pd_pose'
            folder_path[-1] = folder_path[-1].replace(".png", ".npy")
            # print("save path",os.path.join(*folder_path[:4]))
            if not os.path.exists(os.path.join(*folder_path[:4])):
                os.makedirs(os.path.join(*folder_path[:4]))
            save_path = os.path.join(*folder_path)
            for k in batch:
                if k != 'meta' and k != 'path':
                    batch[k] = batch[k].cuda()
            with torch.no_grad():
                output = network(batch['inp'])
            # evaluator.evaluate(output, batch)
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            np.save(save_path, kpt_2d)

def run_test():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import glob
    import tqdm
    import torch

    epochs = [4,9,14,19]
    for epoch in epochs:
        print("epoch!", epoch)
        model_name = 'custom_charuco_5d_49'
        save_path = os.path.join('data/test_result', model_name, str(epoch))
        cfg.task = 'pvnet_npy'
        evaluator = make_evaluator(cfg)

        test_files = glob.glob(save_path+"/*.npy")

        # for file_path in test_files:
        for file_path in tqdm.tqdm(test_files):
            # print(file_path)
            predict = np.load(file_path, allow_pickle=True).item()
            evaluator.evaluate(predict)
        result_dict = evaluator.summarize()
        np.save(save_path+'result.npy', result_dict)

def run_custom_multifolder():
    from tools import handle_multifolder_dataset
    data_root = 'data/LND'
    handle_multifolder_dataset.generate_custom(data_root)


def run_custom_multiset():
    from tools import handle_multidataset
    data_root = 'data/'
    handle_multidataset.generate_custom(data_root)


def run_vis_refinement():
    from tools import refinement_visualize
    data_root = 'data/LND'
    refinement_visualize.vis(data_root)


def run_refinement_data():
    from tools import handle_refinement_dataset
    data_root = 'data'
    handle_refinement_dataset.generate_custom(data_root)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    globals()['run_'+args.type]()
