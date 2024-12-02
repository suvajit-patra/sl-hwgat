import numpy as np
from pathlib import Path
import os, pickle, argparse, csv
from multiprocessing import Pool
from tqdm import tqdm
from configs import dataCFG
from constants import *
from pose_modules.keypoint_extract_models import *

def arg_parser():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--root', type=str, required=True,
                        help='Enter the root directory of the dataset')
    parser.add_argument('-ds', type=str, required=True,
                        help='Enter the dataset\'s name')
    parser.add_argument('--meta', type=str, required=True,
                        help='Enter the dataset\'s metadata.csv')
    parser.add_argument('-dr', '--dataroot', type=str, required=False, default='',
                        help='Enter data root relative path')
    parser.add_argument('-ft', '--feature', type=str, required=False, default=feature_type_list[1],
                        help=f'Enter wheather {feature_type_list} data')
    parser.add_argument('-kpm', '--kp_model', type=str, required=False, default='mediapipe', help=f'Choose from {keypoint_model_dict.keys()}')
    # Parse the argument
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()
    root = args.root
    dataset_name = args.ds
    meta = args.meta
    data_root = args.dataroot
    feature = args.feature
    kp_model = args.kp_model

    cfg = dataCFG(dataset_name, feature, kp_model)

    if feature == 'keypoints' and data_root == '':
        print("Specify dataroot for keypoints")
        exit(0)

    meta_csv = meta
    data_root_path = os.path.join(root, data_root)

    if not os.path.exists(f'./input/{dataset_name}'):
        # making input folder
        os.makedirs(f'./input/{dataset_name}')

    if not os.path.exists(f'./output/{dataset_name}'):
        # making output folder
        os.makedirs(f'./output/{dataset_name}')

    vid_splits = {'train': [], 'val': [], 'test': []}
    vid_class = {}
    class_map = {}
    data_info = {}

    # creates the video split and video class files
    c = 0
    with open(meta_csv, newline='') as csvfile:  # id, video_dir, video_name, class, split
        reader = csv.reader(csvfile, delimiter=',')
        reader.__next__()  # ignore header
        for row in tqdm(reader):
            vid_name = row[0]
            word = row[3].strip()
            if class_map.get(word) == None:  # target encoding
                class_map[word] = c
                c += 1
            vid_class[vid_name] = class_map[word]
            if feature == 'keypoints':
                # data = pickle.load(open(os.path.join(root, data_root_path, *row[0].split("/")[1:-1],row[2][:-4] + '.pkl'), "rb"))[feature]
                data = pickle.load(open(os.path.join(root, data_root_path, row[0] + '.pkl'), "rb"))
                try:
                    feat = data['feat']
                except:
                    feat = data[feature]
                if 1 in feat.shape or 0 in feat.shape or feat.sum() == 0:
                    continue
                # data_info[row[0]] = os.path.join(root, data_root_path, *row[1].split("/")[1:-1],row[2][:-4] + '.pkl')
                # data_info[row[0]] = os.path.join(root, data_root_path, row[0] + '.pkl')
                data = cfg.data_transform(data)
                # print(data.shape)
                data_info[row[0]] = data
            else:
                data_info[row[0]] = os.path.join(root, row[1])
            if row[4] == 'train':
                vid_splits['train'].append(vid_name)
                
            elif row[4] == 'val':
                vid_splits['val'].append(vid_name)
            elif row[4] == 'test':
                vid_splits['test'].append(vid_name)
            else:
                print(f'Not In SPLIT {vid_name}')

    print(f'Unique Words: {len(class_map)}')

    with open(cfg.vid_split_path, 'wb') as f:
        pickle.dump(vid_splits, f)
    with open(cfg.vid_class_path, 'wb') as f:
        pickle.dump(vid_class, f)

    with Path(cfg.class_map_path).open('w', newline='') as file:
        writer_object = csv.writer(file)
        header = ['class', 'word']
        writer_object.writerow(header)
        for w, c in class_map.items():
            writer_object.writerow([c, w])
            

    with open(cfg.data_map_path, 'wb') as f:
        pickle.dump(data_info, f)
