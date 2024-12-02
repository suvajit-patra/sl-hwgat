# get kplist keys from pose_modules.keypoint_extract_models

dataset_params = {"INCLUDE" : {'num_class' : 262, 'src_len' : 64},
                  "INCLUDE_INTERSECTION" : {'num_class' : 2002, 'src_len' : 192},
                  "FDMSE_INTERSECTION" : {'num_class' : 262, 'src_len' : 64},
                  "FDMSE-ISL" : {'num_class' : 2002, 'src_len' : 192},
                  "FDMSE-ISL400" : {'num_class' : 400, 'src_len' : 192},
                  "FDMSE-ATOMIC": {'num_class' : 1099, 'src_len' : 192},
                  "FDMSE_COMPOSITE": {'num_class' : 1099, 'src_len' : 192},
                  "WLASL" : {'num_class' : 2000, 'src_len' : 64},
                  "AUTSL": {"num_class" : 226, 'src_len' : 64},
                  "MSASL": {"num_class" : 1000, 'src_len' : 64},
                  "LSA64": {"num_class" : 64, 'src_len' : 64}}

feature_type_list = ["rgb", "keypoints"]

input_dim = {"kp2D" : 2, "kp3D" : 3}

kp_list = {"mediapipe": [0, 2, 5, 11, 12, 13, 14, 15, 16] + 
           [0+33+468, 4+33+468, 5+33+468, 8+33+468, 9+33+468, 12+33+468, 13+33+468, 16+33+468, 17+33+468, 20+33+468,
             0+21+33+468, 4+21+33+468, 5+21+33+468, 8+21+33+468, 9+21+33+468, 12+21+33+468, 13+21+33+468, 16+21+33+468,
               17+21+33+468, 20+21+33+468],
            "dwpose": [0, 1, 2, 5, 6, 7, 8, 9, 10] + [91, 95, 96, 99, 100, 103, 104, 107, 108, 111] +
             [91+21, 95+21, 96+21, 99+21, 100+21, 103+21, 104+21, 107+21, 108+21, 111+21]}

coord_list = {"mediapipekp2D": [0, 1], "mediapipekp3D": [0, 1, 2], "dwposekp2D": [0, 1]}
