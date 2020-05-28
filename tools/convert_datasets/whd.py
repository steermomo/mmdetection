import numpy as np
import os.path as osp
import random
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import pickle
import json
try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

random.seed(42)
# np.seed(42)

# https://discuss.mxnet.io/t/make-ndarray-json-serializable/1627
class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_to_mmdect_middle(df : pd.DataFrame):
    data_list = []
    df_grouped = df.groupby('image_id')
    for image_id, group_data in tqdm(df_grouped):
        bboxes = []
        
        for idx, row_data in group_data.iterrows():
            __, width, height, bbox, source = row_data
            # print(type(bbox))
            bbox = eval(bbox)  # convert str to list object
            x, y, w, h = bbox
            if w > width // 2 or h > height // 2:
                # bad annotation
                continue
            
            # https://github.com/jwyang/faster-rcnn.pytorch/issues/136
            # Getting Nan loss while training
            if w < 3 or h < 3:
                # print(f'find error in {image_id}, {width, height, bbox}')
                continue

            # https://github.com/open-mmlab/mmdetection/blob/master/docs/compatibility.md
            # 从0开始
            # assert bbox[2] + bbox[0] < width and bbox[3] + bbox[1] < height, f'{bbox}, {width, height}'
            # bbox[2] = max(bbox[2], width - bbox[0])
            # bbox[3] = max(bbox[3], height - bbox[1])
            # assert bbox[0] >= 0 and bbox[1] >= 0
            # bbox[2] = bbox[2] - 1
            # bbox[3] = bbox[3] - 1
            # 这里需要制定的是对角线坐标
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            bboxes.append(bbox)
            # print(bbox)
        
        # https://github.com/open-mmlab/mmdetection/blob/master/docs/compatibility.md
        # In MMDetection 2.0, label "K" means background, and labels [0, K-1] correspond to the K = num_categories object categories.
        ant_dict = {
            'filename': f'{image_id}.jpg',
            'width': width,
            'height': height,
            'ann': {
                'bboxes': np.array(bboxes),
                'labels': np.zeros(len(bboxes)).astype(np.int64),
                # 'labels': np.ones(len(bboxes)).astype(np.int64),
            },
            'source': source
        }
        # print(np.array(bboxes).shape)
        data_list.append(ant_dict)
    return data_list


def convert_global_wheat_detection_data(annot_path):
    annot_file = open(annot_path, 'rt', encoding='utf-8')
    
    df = pd.read_csv(annot_path)

    data_list = convert_to_mmdect_middle(df)

    save_fp = osp.join(osp.dirname(annot_path), osp.basename(annot_path).replace('.csv', '.json'))
    with open(save_fp, 'wt', encoding='utf-8') as outfile:
        json.dump(data_list, outfile, indent=4, cls=NDArrayEncoder)

    save_fp = osp.join(osp.dirname(annot_path), osp.basename(annot_path).replace('.csv', '.pkl'))
    with open(save_fp, 'wb') as outfile:
        pickle.dump(data_list, outfile)


def kfold_split(csv_path):
    df = pd.read_csv(csv_path)
    save_dir = osp.dirname(csv_path)
    skf = StratifiedShuffleSplit(n_splits=5, random_state=42)
    print(f'Total :\n{df.source.value_counts()}\n')
    for f_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['source'])):
        print(f'Fold {f_idx}')
        data_train = df.loc[train_idx]
        data_val = df.loc[val_idx]
        
        print(data_train.source.value_counts())
        print('-' * 20)
        print(data_val.source.value_counts())

        save_fp = osp.join(save_dir, f'fold{f_idx}_train.csv')
        data_train.to_csv(save_fp, index=None) 
        data_val.to_csv(osp.join(save_dir, f'fold{f_idx}_val.csv'), index=None) 

        print(f'=>>.  save to : {save_fp}')
        print('=' * 20)

if __name__ == "__main__":
    img_path = ''
    annot_path = r'D:\Dataset\ghd\fold0_train.csv'
    annot_path = '/mnt/d/Dataset/ghd/fold0_train.csv'

    dataset_dir = '/mnt/d/Dataset/ghd'
    dataset_dir = '/Users/steer/Documents/dataset/global-wheat-detection'
    dataset_dir = '/data1/hangli/gwd/data'

    train_csv_fp = osp.join(dataset_dir, 'train.csv')
    kfold_split(train_csv_fp)

    for fold_idx in range(5):
        for trv in ['train', 'val']:
            annot_path = osp.join(dataset_dir, f'fold{fold_idx}_{trv}.csv')
            print(f'=>>> {annot_path}')
            convert_global_wheat_detection_data(annot_path)


