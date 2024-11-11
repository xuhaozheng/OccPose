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

    network = make_network(cfg).cuda()

    load_network(network, cfg.model_dir,esume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()
    '''
    Define the save path
    '''
    save_path = ''

    if not os.path.exists(save_path):
            os.makedirs(save_path)

    data_loader = make_data_loader(cfg, is_train=False, inference=True)
    evaluator = make_evaluator(cfg)
    test_num = 0
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
        test_num += 1
    evaluator.save(save_path)
    evaluator.summarize(save_path)

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
        visualizer.visualize(output, batch)

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
