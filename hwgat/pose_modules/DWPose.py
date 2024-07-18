import numpy as np
from typing import List, Optional, Union
import cv2
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules


class Model:
    def __init__(self, config_file=None, checkpoint_file=None, device='cuda:0') -> None:
        register_all_modules()
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        if self.config_file is None:
            self.config_file = '/mmpose/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        if self.checkpoint_file is None:
            self.checkpoint_file = 'pose_modules/weights/dw-ll_ucoco_384.pth'
        self.device = device
        self.model = init_model(self.config_file, self.checkpoint_file, self.device)

    def __call__(self, image:Union[np.ndarray, str]):
        results = inference_topdown(self.model, image)[0]
        result = np.concatenate([results.pred_instances.keypoints[0], np.expand_dims(results.pred_instances.keypoint_scores[0], axis=1)], axis=1) if results.pred_instances.keypoints.any() else np.zeros(133, 3)
        return result, *results.img_shape #return result_kps 2d with confidence, height width
    