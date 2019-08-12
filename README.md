# 3D-RelNet: Joint Object and Relation Network for 3D prediction
Nilesh Kulkarni, Ishan Misra, Shubham Tulsiani, Abhinav Gupta.

[Project Page](https://nileshkulkarni.github.io/relative3d/)

![Teaser Image](https://nileshkulkarni.github.io/relative3d/resources/images/teaser.png)

## Demo and Pre-trained Models

Please check out the [interactive notebook suncg](demo/demo_suncg.ipynb), [interactive notebook nyu](demo/demo_nyu.ipynb) which shows reconstructions using the learned models. To run this, you'll first need to follow the [installation instructions](docs/installation.md) to download trained models and some pre-requisites.

## Training and Evaluating
To train or evaluate the (trained/downloaded) models, it is first required to [download the SUNCG dataset](https://github.com/shubhtuls/factored3d/blob/master/docs/suncg_data.md) and [preprocess the data](https://github.com/shubhtuls/factored3d/blob/master/docs/preprocessing.md) and download the splits [here](https://www.dropbox.com/s/tomlyczen5ktyva/suncg_splits.tar.gz?dl=0). Please see the detailed README files for [Training](docs/training.md) or [Evaluation](docs/evaluation.md) of models for subsequent instructions.

To train or evaluate on the NYUv2 dataset the (trained/downloaded) models, it is first required to [download the NYU dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [preprocess the data](docs/preprocess_nyu.md) and download the splits [here](https://www.dropbox.com/s/mhvu39z1rhqmfox/nyu_splits.tar.gz?dl=0). Please see the detailed README files for [Training](docs/training.md) or [Evaluation](docs/evaluation.md) of models for subsequent instructions.
### Citation
If you use this code for your research, please consider citing:
```
@article{kulkarni20193d,
  title={3D-RelNet: Joint Object and Relational Network for 3D Prediction},
  author={Nilesh Kulkarni
  and Ishan Misra
  and Shubham Tulsiani
  and Abhinav Gupta},
  journal={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
