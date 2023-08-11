# Installation guide

1\) Environment requirements

* Python >= 3.7
* Pytorch >= 1.9.0
* CUDA >= 10.2

The following installation guide supposes ``python=3.7``, ``pytorch=1.12.1``, ``cuda=11.3``, ``torch-scatter=2.0.9``, and ``spconv-cu113==2.1.25``. You may change them according to your system.

Create a conda virtual environment and activate it.
```
conda create -n gapro python=3.7
conda activate gapro
```

2\) Install the dependencies
```
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip3 install spconv-cu113==2.1.25
pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip3 install -r requirements.txt
```

3\) Install [Segmentator](https://github.com/Karbo123/segmentator)

```
git clone https://github.com/Karbo123/segmentator.git

cd segmentator/csrc
mkdir build && cd build

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

make && make install
```

4\) Install build requirement

```
sudo apt-get install libsparsehash-dev
```

5\) Setup [ISBNet](https://github.com/VinAIResearch/ISBNet)

```
cd ./ISBNet/isbnet/pointnet2
python3 setup.py bdist_wheel
cd ./dist
pip3 install <.whl>

cd ../../../
python3 setup.py build_ext develop
```

6\) Setup [SPFormer](https://github.com/sunjiahao1999/SPFormer)

```
cd ./SPformer/spformer/lib/
python setup.py develop
cd ../../
python setup.py develop
```