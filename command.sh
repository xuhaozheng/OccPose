### Train
python train_net.py --cfg_file configs/custom.yaml train.batch_size 8
python train_net.py --cfg_file configs/hrnet.yaml train.batch_size 2
### Generate Dataset
python run.py --type custom_multifolder
### Evaluate
python run.py --type evaluate --cfg_file configs/custom.yaml
### Tensorboard
tensorboard --logdir data/record/pvnet
### Test (Generate npy then test)
python run.py --type generate_test --cfg_file configs/custom.yaml
python run.py --type test --cfg_file configs/custom.yaml
### Inference
python run.py --type inference --cfg_file configs/custom.yaml
### Vis
python run.py --type vis_refinement



'''
3090 Env configuration
conda create -n OccPose python=3.9 -y
conda activate OccPose
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install yacs
pip install opencv-python
pip install pycocotools
pip install tensorboardX
conda install mkl==2024.0 -y
pip install tqdm
pip install plyfile
pip install scipy
pip install imgaug
'''
