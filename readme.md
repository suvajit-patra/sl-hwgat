## Setup

Python packages need to be installed:-

```
pickle
torch==1.13.1
numpy==1.23.3
tqdm==4.64.1
opencv-python==4.6.0.66
mediapipe==0.8.11
decord
```

After this, one metadata must be generated to run the deep learning pipeline with the following command.
```
python meta_generators/AUTSL_meta_gen.py --root '/home/sysadm/workspace/datasets/AUTSL/' --dp '/home/sysadm/workspace/datasets/AUTSL/AUTSL/' --meta '/home/sysadm/workspace/datasets/AUTSL/metadata.csv'
```

!!!Warning: value of ```--dp``` must not be followed by '/' like '/home/sysadm/workspace/datasets/data_300'

This should generate a file in '/home/sysadm/workspace/datasets/data_300_meta/metadata.csv'.

## Execute

In three steps the model can be trained on any dataset.

1. Extract mediapipe keypoints and save them using the "pose_feature_extract.py" file by running the following command, where `--root`: root directory of the dataset, `--meta`: dataset\'s metadata.csv, `--out_path`: saving path of the mediapipe outputs (keypoints) (the folder will be created under the root directory).

```
python pose_feature_extract.py --root '/home/sysadm/workspace/datasets/' --meta '/home/sysadm/workspace/datasets/data_300_meta/metadata.csv' --out_path 'mediapipe_out/'
```

2. Next preprocess the computed keypoints so that it can be used to trained the transformer based model using the following command, where `--ds`: dataset name, `--root`: root directory of the dataset, `--meta`: dataset\'s metadata.csv, `--w`: video width, `--h`: video height, `--mout`: mediapipe output relative path or absolute path `--f_type`: feature type that is extracted. it can be 2d kp, 3d kp or distance..

```
python data_preprocess.py --root /data2/datasets/AUTSL/ --ds AUTSL --meta /data2/datasets/AUTSL/AUTSL_meta/metadata.csv -dr mediapipe_out/ -ft keypoints
```

3. Now we can start the training process of the model by running: `python main.py -m train -d AUTSL --model SignAttention_v6 -p mediapipe`

4. Now we can start the testing process of the model by running: `python main.py -m test -d AUTSL --model SignAttention_v6 -p mediapipe -t 240227_1807`

5. Now we can load and train the model by running: `python main.py -m load -d AUTSL --model SignAttention_v6 -p mediapipe -t 240227_1807 -px best_acc`

## Extra

1. "config.py" file can be modified accordingly.
