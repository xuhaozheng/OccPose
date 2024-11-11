# OccPose

This is the official implementation of "Occlusion-Robust Markerless Surgical Instrument Pose Estimation"

## Installation

1. Set up the python environment:
    ```
    conda create -n OccPose
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install yacs
    pip install opencv-python
    pip install pycocotools
    pip install tensorboardX
    conda install mkl==2024.0 -y
    pip install tqdm
    pip install plyfile
    pip install scipy
    pip install imgaug
    ```
2. Compile cuda extensions under `lib/csrc`:
    ```
    ROOT=/path/to/clean-pvnet
    cd $ROOT/lib/csrc
    export CUDA_HOME="/usr/local/cuda-11.3"
    cd ransac_voting
    python setup.py build_ext --inplace
    cd ../nn
    python setup.py build_ext --inplace
    cd ../fps
    python setup.py build_ext --inplace
    
    # If you want to run with a detector
    cd ../dcn_v2
    python setup.py build_ext --inplace
    ```
3. Put the dataset in dataset folder

## Testing
    ```
python run.py --type evaluate --cfg_file configs/hrnet.yaml or
python run.py --type evaluate --cfg_file configs/custom.yaml
    ```

## Training

### Training on Linemod
    ```
    python train_net.py --cfg_file configs/hrnet.yaml train.batch_size 2 or
    python train_net.py --cfg_file configs/custom.yaml train.batch_size 4
    ```