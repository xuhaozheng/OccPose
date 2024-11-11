from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'LinemodTest': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/test.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'LinemodTrain': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/train.json'.format(cfg.cls_type),
            'split': 'train'
        },
        'TlessTest': {
            'id': 'tless_test',
            'ann_file': 'data/tless/test_primesense/test.json',
            'split': 'test'
        },
        
        'CustomTrain': {
            'id': 'custom',
            'data_root': 'data/custom_LND',
            'ann_file': 'data/custom_LND/train.json',
            'split': 'train'
        },
        'CustomTest': {
            'id': 'custom',
            'data_root': 'data/custom_LND',
            'ann_file': 'data/custom_LND/test.json',
            'split': 'test'
        },
        'CustomAll': {
            'id': 'custom',
            'data_root': 'data/custom_LND',
            'ann_file': 'data/custom_LND/all.json',
            'split': 'test'
        },
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
