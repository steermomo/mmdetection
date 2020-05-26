import mmcv
import numpy as np
import pickle
from .builder import DATASETS
from .custom import CustomDataset

# https://github.com/steermomo/mmdetection/blob/master/docs/tutorials/new_dataset.md
@DATASETS.register_module()
class GHDDataset(CustomDataset):

    CLASSES = ('wheat')

    def load_annotations(self, ann_file):

        with open(ann_file, 'rb') as infile:
            anno_info = pickle.load(infile)
        # ann_list = mmcv.list_from_file(ann_file)
        return anno_info

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']