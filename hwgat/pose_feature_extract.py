"""
Extract mediapipe keypoints as features from all the videos present in the dataset
"""

import csv
from decord import VideoReader
from decord import cpu
import numpy as np
from tqdm import tqdm
import pickle
from multiprocessing import Pool, Manager
import os
import argparse
from numpy import ndarray
from pose_modules.keypoint_extract_models import *
import warnings
import importlib
warnings.filterwarnings('ignore')

# vid_features = {}  # {'vid_name' : np.array(fps, 1662)}
# vid_num_frames = {}    # {'vid_name' : fps}

# id, vid_dir+vid_name, vid_name, cls, split

out_path = ''

def get_model(name:str):
    module = importlib.import_module('pose_modules.'+keypoint_model_dict[name]['module'])
    model = getattr(module, 'Model')()
    return model

def init(root: str, meta: str) -> list:
    """Collects all the video names with corresponding locations

    Args:
        root (str): Path of root directory of the dataset
        meta (str): Path of the metadata.csv file

    Returns:
        vid_names (list): A list of each video's [location, name, root, id] 
    """
    vid_names = []
    with open(meta, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.__next__()
        for row in reader:
            vid_names.append(
                [(os.path.split(row[1])[0]), row[2], root, row[0]])
    return vid_names


def get_video_features(vid_name) -> list:
    """It calculates the mediapipe keypoints from a video

    Args:
        vid_name (list): It contains one video's [location, name, root, id]

    Returns:
        (list): A list of video [location, name, features, number of frames, id]
    """
    if os.path.exists(os.path.join(out_path, vid_name[3]+".pkl")):
        return [None]*7

    pose_model = get_model(model)
    kp_shape = keypoint_model_dict[model]['shape']

    if type(vid_name) is str:
        cap = VideoReader(vid_name, ctx=cpu(0))
    else:
        cap = VideoReader(os.path.join(vid_name[2], vid_name[0], vid_name[1]), ctx=cpu(0))

    num_frames = len(cap)
    vid_height, vid_width = cap[0].shape[:2]

    features = np.zeros((num_frames, *kp_shape))

    i_th_frame = 0

    for image in cap:
        # saving the i-th frame feature
        features[i_th_frame] = pose_model(image.asnumpy())[0]
        i_th_frame += 1

    return vid_name[0], vid_name[1], features, i_th_frame, vid_name[3], vid_width, vid_height

def arg_parser():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--root', type=str, required=True,
                        help='Enter the root directory of the dataset')
    parser.add_argument('--meta', type=str, required=True,
                        help='Enter the dataset\'s generated metadata.csv')
    parser.add_argument('--out_path', type=str,
                        required=True, help='Enter the output path')
    parser.add_argument('-m', '--model', type=str, required=False, default='mediapipe', help=f'Choose from {keypoint_model_dict.keys()}')
    parser.add_argument('--num_processes', '-p', type=int, required=False, default=10, help='Enter number of processes')
    # Parse the argument
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parser()
    global root
    root = args.root
    meta = args.meta
    out_path = os.path.join(args.root, args.out_path)
    model = args.model
    num_processes = args.num_processes

    try:
        # making mediapipe output storing folder
        os.makedirs(out_path)
    except:
        pass

    storing_data = {}

    vid_names = init(root, meta)

    # vid_loc, vid_name, features, num_frames, id, vid_width, vid_height = get_video_features(vid_names[0])
    # print(features)
    # filename = os.path.join(out_path, id+".pkl")
    # storing_data = {
    #             'feat': features,
    #             'num_frames': num_frames,
    #             'vid_loc': vid_loc,
    #             'vid_name': vid_name,
    #             'vid_width': vid_width,
    #             'vid_height': vid_width
    #         }
    
    #pickle.dump(storing_data, open(filename, "wb"))

    # creating multiple processes to reduce the processing time
    with Pool(processes=num_processes) as pool:
        results = tqdm(pool.imap_unordered(get_video_features,
                       vid_names), total=len(vid_names), desc='Videos')
        for vid_loc, vid_name, features, num_frames, id, vid_width, vid_height in results:
            if features is None:
                continue
            storing_data = {
                'feat': features,
                'num_frames': num_frames,
                'vid_loc': vid_loc,
                'vid_name': vid_name,
                'vid_width': vid_width,
                'vid_height': vid_width
            }
            # saving the calculated mediapipe features with video id
            filename = os.path.join(out_path, id+".pkl")
            with open(filename, "wb") as outfile:
                pickle.dump(storing_data, outfile)
