import numpy as np
import torch
import importlib
import os
from constants import *
from datetime import datetime
from dataTransform import *

#config file
class CFG:
    def __init__(self):
        self.input_type = 'kp2D'

        self.origin_idx = 0 #nose point
        self.anchor_points = [3, 4] #shoulder points
        self.frame_augmentation = [0.5, 1.5]
        self.sampling_prob = 0.2
        self.shear_std = 0.1
        self.rotation_std = 0.1
        self.origin_shift_std = 0.1
        self.left_slice = [9, 19, 7] #left hand slice
        self.right_slice = [19, 29, 8] #right hand slice
        self.x_range = self.y_range = [0, 1] #tpose hand range
        self.random_shift = True
        self.uniform_sample = True
        self.random_sample = True

        if not os.path.exists('input'):
            os.makedirs('input')

        if not os.path.exists('output'):
            os.makedirs('output')

class dataCFG(CFG):
    def __init__(self, dataset, feature_type, pose_method=None):
        super().__init__()
        self.dataset_name = dataset
        self.feature_type = feature_type
        if pose_method is not None:
            if feature_type == 'keypoints':
                self.data_transform = Compose([MediapipeDataProcess(),
                                            PoseSelect(kp_list[pose_method], coord_list[pose_method+self.input_type])])
            else:
                raise NotImplementedError('rgb not implemented')

        self.class_map_path = f"input/{self.dataset_name}/class_map_{self.dataset_name}.csv"
        self.vid_split_path = f"input/{self.dataset_name}/vid_splits_{self.dataset_name}.pkl"
        self.vid_class_path = f"input/{self.dataset_name}/vid_class_{self.dataset_name}.pkl"
        self.data_map_path = f"input/{self.dataset_name}/data_map_{self.dataset_name}_{self.feature_type}.pkl"


class runCFG(dataCFG):
    def __init__(self, dataset, model_type, name_postfix, feature_type, mode, time, model_weights, cuda_id):
        super().__init__(dataset, feature_type)
        rand_seed = 1001
        np.random.seed(rand_seed)
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)

        self.mode = mode

        self.dataset_name = dataset
        self.src_len = dataset_params[self.dataset_name]['src_len']
        self.feature_type = feature_type
        self.input_dim = input_dim[self.input_type]

        self.model_type = model_type
        self.criterion_type = "smooth_cross_entropy"
        self.optimizer_type = "adamw"
        self.scheduler = "CosineAnnealingLR"
        self.early_stopping = False
        self.early_stopping_step = 400

        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
        
        print("Running on device = ", self.device)

        module = importlib.import_module('models.model_params')
        self.model_params = getattr(module, self.model_type+'Params')(dataset_params[self.dataset_name], 
                                                                      self.input_dim, self.device)

        self.lr = 0.0005  # learning rate
        self.start_epoch = 0
        self.epochs = 200
        self.batch_size = 4
        self.best_val_loss = float('inf')
        self.n_workers = 8

        self.save_interval = 100

        self.train_transform = Compose([
                                        KeypointMasking(self.sampling_prob, self.left_slice[0], self.right_slice[1]),
                                        HandCorrection(self.left_slice, self.right_slice),
                                        NormalizeKeypoints(self.origin_idx, self.anchor_points),
                                        ShearTransform(self.shear_std),
                                        RotatationTransform(self.rotation_std),
                                        TemporalAugmentation(self.frame_augmentation, self.uniform_sample, self.random_sample),
                                        TemporalSample(self.src_len, self.random_shift),
                                        RandomFlip(self.feature_type),
                                        WindowCreate(self.src_len),
                                        ])
        
        self.test_transform = self.val_transform = Compose([HandCorrection(self.left_slice, self.right_slice),
                                                            NormalizeKeypoints(self.origin_idx, self.anchor_points),
                                                            TemporalSample(self.src_len),
                                                            WindowCreate(self.src_len),
                                                            ])

        self.train_pin_memory = True
        self.test_pin_memory = False
        self.val_pin_memory = True
        self.train_shuffle = True
        self.test_shuffle = False
        self.val_shuffle = False

        if time == 'none':
            self.time = datetime.now().strftime("%Y%m%d_%H%M")[2:]
        else:
            self.time = time

        if name_postfix == 'none':
            self.postfix = 'best_loss'
        else:
            self.postfix = name_postfix

        if model_weights != 'none':
            self.model_weights = model_weights
        else:
            self.model_weights = None

        # save models, cm, image paths
        self.save_suffix = f'{self.model_type}_{self.time}'
        self.out_folder = f"output/{self.dataset_name}/{self.save_suffix}"
        self.save_config_path = f"{self.out_folder}/config.pkl"
        self.save_model_path = f"{self.out_folder}/model"
        self.save_cm_path = f"{self.out_folder}/cm_list_w.csv"
        self.save_loss_curve_path = f"{self.out_folder}/loss_curve.png"
        self.save_acc_curve_path = f"{self.out_folder}/acc_curve.png"

        if not os.path.exists(f"{self.out_folder}"):
            os.makedirs(f"{self.out_folder}")
