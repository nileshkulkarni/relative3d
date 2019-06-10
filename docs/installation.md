# Installation Instructions

Two overall comments:
* Result visualizations depend on blender. We provide a version, but if you have issues where the renderings don't show up or where the script cannot read the result images, you may have to compile blender and provide a bpy.so file that matches your precise system configuration. See [here](https://wiki.blender.org/index.php/User:Ideasman42/BlenderAsPyModule) and [here](https://blender.stackexchange.com/questions/102933/a-working-guidance-for-building-blender-as-bpy-python-module) for information about how to do this. The results by themselves do not depend on blender, and if you just want to compute predictions, you do not need blender.
* You should run each of these commands in the main root directory. 

#### Setup virtualenv.
```
conda env create -f docs/condenv.yml
source activate 3drelNet
pip install -U pip
pip install -r docs/requirements.txt
```

#### Install pytorch.
```
pip install torchvision visdom dominate
```

#### Compile cython modules.
First, we need to compile some cython utilities.
```
cd utils
python setup.py build_ext --inplace
mv relative3d/utils/bbox_utils.so ./
rm -rf build/ # remove redundant folders
rm -rf relative3d/ # remove redundant folders
sh make.sh
cd ..
```

#### Download pre-trained models.
```
# Download pre-trained Resnet18 Model.
wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O ~/.torch/models/resnet18-5c106cde.pth

# Download our models.
wget https://www.dropbox.com/s/r7xw5fw6awj5h4v/cachedir.tar.gz && tar -xf cachedir.tar.gz
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/factored3d/blender.tar.gz && tar -xf blender.tar.gz && mv blender renderer/. 
```

#### Setup external dependencies.
```
mkdir external; cd external;
# Python interface for binvox
git clone https://github.com/dimatura/binvox-rw-py ./binvox

# Piotr Dollar's toolbox
git clone https://github.com/pdollar/toolbox ./toolbox

# Edgeboxes code
git clone https://github.com/pdollar/edges ./edges

# SSC-Net code (used for computing voxelization for the baseline)
git clone https://github.com/shurans/sscnet ./sscnet
cd ..
```
