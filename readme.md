# Hierarchical Windowed Graph Attention Network (HWGAT) for Sign Language Recognition

## Overview

Hierarchical Windowed Graph Attention Network (HWGAT) is a deep learning model specifically designed for sign language recognition. This model leverages hierarchical and windowed attention mechanisms to effectively capture the temporal and spatial dependencies in sign language skeleton data. This repository includes a comprehensive implementation of HWGAT, covering data preprocessing and the full training pipeline.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
- [Examples](#examples)
- [License](#license)

## Installation

To get started with HWGAT for sign language recognition, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/suvajit-patra/sl-hwgat.git
    cd sl-hwgat/hwgat
    ```

2. Create a docker instance with the `Dockerfile` and run the container.

3. Install the required dependencies with `pip install -r requirements.txt`.

## Usage

### Data Preprocessing

The data preprocessing pipeline prepares the raw sign language data for training.

1. **Generate metadata:** Ensure your dataset is structured properly and run the metadata generator scripts with correspoding dataset. After this, one metadata must be generated to run the deep learning pipeline with the following command. If you are using different dataset then make our own mata generator. 
    ```bash
    python meta_generators/FDMSE_meta_gen.py
    ```

    **!!!Note:** Remember to update the paths inside every meta generator script.

    This should generate a file in '/data/datasets/FDMSE/FDMSE_meta/metadata.csv'.

2. **Generate keypoints:** Extract keypoints and save them using the `pose_feature_extract.py` file by running the following command, where `--root`: root directory of the dataset, `--meta`: dataset\'s metadata.csv, `--out_path`: saving path of the outputs (keypoints) (the folder will be created under the root directory).

    ```
    python pose_feature_extract.py --root '/data/datasets/FDMSE' --meta '/data/datasets/FDMSE/FDMSE_meta/metadata.csv' -m mediapipe --out_path 'mediapipe_out/'
    ```

3. **Process keypoints data:** Next preprocess the generated keypoints so that it can be used to trained the transformer based model using the following command, where `--ds`: dataset name, `--root`: root directory of the dataset, `--meta`: dataset\'s metadata.csv, `-dr`: keypoints output relative path from the root `-ft`: feature type that is extracted.

    ```
    python data_preprocess.py --root /data/datasets/FDMSE/ --ds FDMSE --meta /data/datasets/FDMSE/FDMSE_meta/metadata.csv -dr mediapipe_out/ -ft keypoints
    ```

### Model Training

Once the data is preprocessed, you can train the HWGAT model using the training pipeline provided.

1. **Configure the training parameters:** Edit the `configs.py` file to set your training parameters, such as learning rate, batch size, number of epochs, etc.

2. **Training the model:** Start the training process of the model by running
    ```bash
    python main.py -m train -d FDMSE --model SignAttention_v6 -p mediapipe
    ```

3. **Testing the model:** Test the model using
    ```bash
    python main.py -m test -d FDMSE --model SignAttention_v6 -p mediapipe -t 240227_1807 -px best_loss
    ```

4. **Load and train the model:** Load and train the model or finetune on different datasets using
    1. Load and train on same dataset.
        ```bash
        python main.py -m load -d FDMSE --model SignAttention_v6 -p mediapipe -t 240227_1807 -px best_loss
        ```

    1. Finetune on other dataset.
        ```bash
        python main.py -m load -d INCLUDE --model SignAttention_v6 -p mediapipe -mw output/FDMSE/SignAttention_v6_240227_1807/model_best_loss.pt
        ```


## Examples

Go to [this repository](https://github.com/suvajit-patra/sl-hwgat-demo) to get the demo application the HWGAT model for sign language recognition tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for using this repository. For any questions or support, please open an issue in this repository.

---