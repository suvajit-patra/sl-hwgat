import math
from typing import Any
import random
import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R


class Compose:
    """
    Compose a list of pose transforms
    
    Args:
        transforms (list): List of transforms to be applied.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: np.ndarray):
        """Applies the given list of transforms

        Args:
            x (np.ndarray): input data

        Returns:
            np.ndarray: data after the transforms
        """
        for transform in self.transforms:
            x = transform(x)
        return x

class DWPoseDataProcess:
    def __init__(self) -> None:
        self.left_idx = [9, 91, 95, 96, 99, 100, 103, 104, 107, 108, 111]
        self.right_idx =[10, 91+21, 95+21, 96+21, 99+21, 100+21, 103+21, 104+21, 107+21, 108+21, 111+21]
        self.visibility_threshold = 0.5

    def __call__(self, data):
        feat = data['feat']
        max_y = np.max(feat[:, self.left_idx[0], 1])
        frame_idx = np.where(feat[:, self.left_idx[0], 1] > 0.95*max_y)[0]
        if frame_idx.size > 1:
            for f in np.squeeze(frame_idx):
                feat[f, self.left_idx[1:], :-1] = 0
        frame_idx = np.where(feat[:, self.left_idx[0], -1]<self.visibility_threshold)[0]
        if frame_idx.size > 1:
            for f in np.squeeze(frame_idx):
                feat[f, self.left_idx[1:], :-1] = 0
        max_y = np.max(feat[:, self.right_idx[0], 1])
        frame_idx = np.where(feat[:, self.right_idx[0], 1] > 0.95*max_y)[0]
        if frame_idx.size > 1:
            for f in np.squeeze(frame_idx):
                feat[f, self.right_idx[1:], :-1] = 0
        frame_idx = np.where(feat[:, self.right_idx[0], -1]<self.visibility_threshold)[0]
        if frame_idx.size > 1:
            for f in np.squeeze(frame_idx):
                feat[f, self.right_idx[1:], :-1] = 0
        return feat
    
class MediapipeDataProcess:
    def __init__(self):
        pass
    
    def __call__(self, data):
        width = data['vid_width']
        height = data['vid_height']
        feat = data['feat']
        feat[:, :, 0] = feat[:, :, 0]*width
        feat[:, :, 1] = feat[:, :, 1]*height
        return feat


class PoseSelect:
    def __init__(self, kp_list, coord_list) -> None:
        self.kp_idx = kp_list
        self.coord_list = coord_list
    
    def __call__(self, data:np.ndarray):
        data = np.take(data, indices=self.kp_idx, axis=1)
        data = np.take(data, indices=self.coord_list, axis = 2)
        return data

class NormalizeKeypoints:
    def __init__(self, origin_idx, anchor_points_idx) -> None:
        self.origin_idx = origin_idx
        self.anchor_points_idx = anchor_points_idx
        assert len(self.anchor_points_idx) == 2, "Anchor Points cannot be more than 2"
    
    def __call__(self, data:np.ndarray):
        data = self.normCS2d_for_vid(data)
        return data
    
    def normCS2d_for_vid(self, vid):
        left_top, edge_dist = self.normCS2d_1(vid)
        vid = (vid - left_top)/edge_dist
        return vid

    def normCS2d_1(self, vid):
        for keypoints in vid:
            if keypoints[self.origin_idx].all() != 0 and keypoints[self.anchor_points_idx[0]].all() != 0 and keypoints[self.anchor_points_idx[1]].all() != 0:
                root = keypoints[self.origin_idx]
                unit_dist = np.linalg.norm(keypoints[self.anchor_points_idx[0]] - keypoints[self.anchor_points_idx[1]])
                left_top = root - 3*unit_dist
                left_top[1] = root[1] - 2*unit_dist
                edge_dist = 6*unit_dist
                break

        return left_top, edge_dist
    
class RandomFlip:
    def __init__(self, feature_type) -> None:
        self.feature_type = feature_type

    def __call__(self, data):
        if random.uniform(0, 1) <= 0.5:
            if self.feature_type == "rgb":
                data = np.flip(data, axis = 2)
            else:
                data[:, :, 0] = -data[:, :, 0] + 1
        return data

class Rectifier3Dto2D:
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        data = data[:, :, :2]
        return data

# Adopted from: https://github.com/AmitMY/pose-format/
class ShearTransform:
    """
    Applies `2D shear <https://en.wikipedia.org/wiki/Shear_matrix>`_ transformation
    
    Args:
        shear_std (float): std to use for shear transformation. Default: 0.2
    """
    def __init__(self, shear_std: float=0.2):
        self.shear_std = shear_std

    def __call__(self, data:np.ndarray):
        """
        Applies shear transformation to the given data.

        Args:
            data (np.ndarray): input data

        Returns:
            np.ndarray: data after shear transformation
        """
        # print("ShearTransform")
        origin = np.clip(np.random.normal(loc=0.5, scale=0.1, size=data.shape[2]), 0, 1)
        x = data - origin
        #assert x.shape[2] == 2, "Only 2 channels inputs supported for ShearTransform"
        shear_matrix = np.eye(2)
        shear_matrix[0][1] = np.random.normal(loc=0, scale=self.shear_std, size=1)[0]
        x[:, :, :2]= np.matmul(x[:, :, :2], shear_matrix)
        data = x + origin
        return data

class ShearTransform3D:
    """
    Applies `3D shear <https://en.wikipedia.org/wiki/Shear_matrix>`_ transformation
    
    Args:
        shear_std (float): std to use for shear transformation. Default: 0.2
    """
    def __init__(self, shear_std: float=0.2):
        self.shear_std = shear_std

    def __call__(self, data:np.ndarray):
        """
        Applies shear transformation to the given data.

        Args:
            data (np.ndarray): input data

        Returns:
            np.ndarray: data after shear transformation
        """
        # print("ShearTransform")
        origin = np.clip(np.random.normal(loc=0.5, scale=0.1, size=data.shape[2]), 0, 1)
        x = data - origin
        #assert x.shape[2] == 2, "Only 2 channels inputs supported for ShearTransform"
        shear_matrix = np.eye(3)
        a, b, c = np.random.normal(loc=0, scale=self.shear_std, size=3)
        a_mat = b_mat = c_mat = np.eye(3)
        a_mat[0, 1] = b        
        a_mat[0, 2] = c        
        b_mat[1, 0] = a
        b_mat[1, 2] = c
        c_mat[2, 0] = a
        c_mat[2, 1] = b
        shear_matrix = a_mat@b_mat@c_mat
        x[:, :, :2]= np.matmul(x[:, :, :3], shear_matrix)
        data = x + origin
        return data

class RotatationTransform:
    """
    Applies `2D rotation <https://en.wikipedia.org/wiki/Rotation_matrix>`_ transformation.
    
    Args:
        rotation_std (float): std to use for rotation transformation. Default: 0.2
    """
    def __init__(self, rotation_std: float=0.2):
        self.rotation_std = rotation_std

    def __call__(self, data):
        """
        Applies rotation transformation to the given data.

        Args:
            data (np.ndarray): input data

        Returns:
            np.ndarray: data after rotation transformation
        """
        # print("RotatationTransform")
        origin = np.clip(np.random.normal(loc=0.5, scale=0.1, size=data.shape[2]), 0, 1)
        x = data - origin
        if x.shape[2] == 2:
            rotation_angle = np.random.normal(loc=0, scale=self.rotation_std, size=1)[0]
            rotation_cos = np.cos(rotation_angle)
            rotation_sin = np.sin(rotation_angle)
            rotation_matrix = [[rotation_cos, -rotation_sin], [rotation_sin, rotation_cos]]
            res = np.matmul(x, rotation_matrix)
        else:
            thetas = np.random.normal(loc=0, scale=self.rotation_std, size=3)*90
            rotation = R.from_euler("xyz", thetas, degrees=True)
            res = np.matmul(x, rotation.as_matrix())
        data = res + origin
        return data

class KeypointMasking:
    """
    Make hand keypoints 0 in some frames according to input probability
    """

    def __init__(self, sampling_prob=0.2, start_kp=9, end_kp=29) -> None:
        self.sampling_prob = sampling_prob
        self.start_kp = start_kp
        self.end_kp = end_kp
    
    def __call__(self, data):
        # print("KeypointMasking")
        x = data #TVC
        sample_num_frames = int(self.sampling_prob*x.shape[0])
        choices = sorted(random.sample(list(range(x.shape[0])), sample_num_frames))
        x[np.array(choices), self.start_kp:self.end_kp] = 0.
        data = x
        return data

class TemporalAugmentation:
    """

    """
    def __init__(self, frame_augmentation=[0.5, 1.5], uniform_sample=True, random_sample=False) -> None:
        self.frame_augmentation = frame_augmentation
        self.uniform_sample = uniform_sample
        self.random_sample = random_sample

    def __call__(self, data):
        # print("TemporalAugmentation")
        x = data
        u = random.uniform(0, 1)
        a = self.frame_augmentation[0]
        b = self.frame_augmentation[1]
        fr_ratio = (b-a)*u + a
        sample_num_frames = int(x.shape[0] * fr_ratio)
        
        sample = random.uniform(0, 1)
        if sample < 0.5 and self.random_sample: # Random Temporal Subsample
            if fr_ratio <= 1:
                choices = sorted(random.sample(list(range(x.shape[0])), sample_num_frames))
            else:
                choices = sorted(random.choices(list(range(x.shape[0])), k=sample_num_frames))
            try:
                x = x[np.array(choices)]
            except:
                print(choices, fr_ratio, x.shape[0])
                return 1
        else:    # Uniform Temporal Subsample
            choices = np.linspace(
                0, x.shape[0]-1, num=sample_num_frames).astype(int)
            x = x[choices]
        data = x
        return data
    
class TemporalSample:
    """

    """
    def __init__(self, max_length=64, random_shift=False) -> None:
        self.max_len = max_length
        self.random_shift = random_shift
        self.scale_std = 0.1
    
    def __call__(self, data):
        # print("TemporalSample")
        x = data
        if x.shape[0] <= self.max_len:  # padding
            
            if not self.random_shift:
                sample = 0.5
            else:
                sample = np.clip(np.random.normal(loc=0.5, scale=self.scale_std, size=1), 0, 1)[0]

            index = int((self.max_len - x.shape[0]) * sample)
            pad1 = x[0]
            pad2 = x[-1]
            # pad1 = pad2 = 0
            blank_feat_fr = np.full(
                (self.max_len//2, x.shape[1], x.shape[2]), pad1, dtype=np.float32)
            blank_feat_bk = np.full(
                (self.max_len-self.max_len//2, x.shape[1], x.shape[2]), pad2, dtype=np.float32) # last frame copying
            vid_feat = np.concatenate([blank_feat_fr, blank_feat_bk], axis=0)
            vid_feat[index:index+x.shape[0], :] = x
            
        else:  # evenly spaced sample
            choices = np.linspace(
                0, x.shape[0]-1, num=self.max_len).astype(int)
            vid_feat = x[choices]
        data = vid_feat
        return data

class HandCorrection:
    """

    """
    def __init__(self, left_slice=[9, 19, 7], right_slice=[19, 29, 8], k_spline=2) -> None:
        self.left_slice = left_slice
        self.right_slice = right_slice
        self.k_spline = k_spline
        
    def vid_hand_correction(self, vid_feat_array, slices):
        dim = vid_feat_array.shape[2]
        if np.sum(vid_feat_array[:, slices[0]:slices[1]]) == 0:
            vid_feat_array[:, slices[0]:slices[1], :] = np.expand_dims(vid_feat_array[:, slices[2], :], axis = 1)
            return vid_feat_array
        start = end = 0
        for frame_idx in range(len(vid_feat_array)):
            if not vid_feat_array[frame_idx, slices[0]:slices[1]].any():
                vid_feat_array[frame_idx, slices[0]:slices[1], :] = np.expand_dims(vid_feat_array[frame_idx, slices[2], :], axis = 0)
            else:
                start = frame_idx
                break
        for frame_idx in reversed(range(len(vid_feat_array))):
            if not vid_feat_array[frame_idx, slices[0]:slices[1]].any():
                vid_feat_array[frame_idx, slices[0]:slices[1], :] = np.expand_dims(vid_feat_array[frame_idx, slices[2], :], axis = 0)
            else:
                end = frame_idx
                break
        not_present_indices = []
        present_indices = []
        for frame_idx in range(start, end+1):
            if vid_feat_array[frame_idx, slices[0]:slices[1]].any():
                present_indices.append(frame_idx)
                # vid_feat_array[frame_idx, slices[0]:slices[1], :] -= np.expand_dims(vid_feat_array[frame_idx, slices[0], :] - vid_feat_array[frame_idx, slices[2], :], axis = 0)
            else:
                not_present_indices.append(frame_idx)

        try:
            for hand_kp in list(range(slices[0], slices[1])):
                x_list = np.take(vid_feat_array, present_indices, 0)[:, hand_kp, 0]
                y_list = np.take(vid_feat_array, present_indices, 0)[:, hand_kp, 1]
                if dim == 3:
                    z_list = np.take(vid_feat_array, present_indices, 0)[:, hand_kp, 2]

                for missing in not_present_indices:
                    tck = interpolate.splrep(present_indices, x_list, k = self.k_spline)
                    x_value = interpolate.splev(missing, tck)
                    # tck = interpolate.interp1d(present_indices, x_list)
                    # x_value = tck(missing)
                    # tck = interpolate.UnivariateSpline(present_indices, x_list)
                    # x_value = tck(missing)
                    # tck = interpolate.Rbf(present_indices, x_list)
                    # x_value = tck(missing)
                    tck = interpolate.splrep(present_indices, y_list, k = self.k_spline)
                    y_value = interpolate.splev(missing, tck)
                    # tck = interpolate.interp1d(present_indices, y_list)
                    # y_value = tck(missing)
                    # tck = interpolate.UnivariateSpline(present_indices, y_list)
                    # y_value = tck(missing)
                    # tck = interpolate.Rbf(present_indices, y_list)
                    # y_value = tck(missing)
                    vid_feat_array[missing, hand_kp, 0] = x_value
                    vid_feat_array[missing, hand_kp, 1] = y_value
                    if dim == 3:
                        tck = interpolate.splrep(present_indices, z_list, k = self.k_spline)
                        z_value = interpolate.splev(missing, tck)
                        vid_feat_array[missing, hand_kp, 2] = z_value
            # print('*')
        except:
            pass
            
        return vid_feat_array

    def __call__(self, data):
        # print("HandCorrection")
        data = self.vid_hand_correction(data, self.left_slice)
        data = self.vid_hand_correction(data, self.right_slice)
        return data
    
class ValidFrames:
    def __init__(self, x_range=[0, 1], y_range=[0, 1]) -> None:
        self.x_range = x_range
        self.y_range = y_range
    
    def __call__(self, data):
        temp_ = np.array([True]*data.shape[1])
        x_ = data[:, :, 0]
        y_ = data[:, :, 1]
        x_0 = x_ >= self.x_range[0]
        x_1 = x_ <= self.x_range[1]
        y_0 = y_ >= self.y_range[0]
        y_1 = y_ <= self.y_range[1]
        x_ = x_0*x_1
        y_ = y_0*y_1
        mask_ = np.logical_not(np.logical_not(x_)@temp_)*np.logical_not(np.logical_not(y_)@temp_)
        if mask_.sum() < 2:
            return data
        return data[mask_]

class WindowCreate:
    def __init__(self, max_len) -> None:
        self.parts_idx = {
            "head": [0, 1, 2],
            "l_arm": [3, 5, 7],
            "l_hand": list(range(9, 19)),
            "r_arm": [4, 6, 8],
            "r_hand": list(range(19, 29)),
        }

        self.window_idx = {
            "window0": self.parts_idx["head"] + self.parts_idx["l_arm"] + self.parts_idx["l_hand"],
            "window1": self.parts_idx["head"] + self.parts_idx["r_arm"] + self.parts_idx["r_hand"],
            "window2": self.parts_idx["head"] + self.parts_idx["l_arm"] + self.parts_idx["r_hand"],
            "window3": self.parts_idx["head"] + self.parts_idx["r_arm"] + self.parts_idx["l_hand"],
        }

        self.max_len = max_len

    def __call__(self, data:np.ndarray):
        vid_feat = data

        vid_feat_stack = np.zeros((self.max_len, 64, vid_feat.shape[-1]))

        vid_feat_stack[:, :16] = vid_feat.take(self.window_idx["window0"], 1)
        vid_feat_stack[:, 16:32] = vid_feat.take(self.window_idx["window1"], 1)
        vid_feat_stack[:, 32:48] = vid_feat.take(self.window_idx["window2"], 1)
        vid_feat_stack[:, 48:64] = vid_feat.take(self.window_idx["window3"], 1)

        return vid_feat_stack
    