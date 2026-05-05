# Install

## Create conda env
conda create -n repl python=3.10 
conda activate repl  

OLD REQUIREMENTS
pip install numpy
pip install opencv-python
pip install rosbags
pip install pandas
pip install torch
pip install scipy
pip install mmengine
pip install mmdet3d
pip install -U openmim
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"

NEW 
pip install -U pip setuptools wheel
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" scipy pandas opencv-python rosbags
pip install -U openmim
mim install "mmengine"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmdet3d==1.4.0"
pip install osmnx
pip install pyrender
pip install carla==0.9.15
pip install contextily