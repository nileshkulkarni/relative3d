# 3D-RelNet: Joint Object and Relation Network for 3D prediction
Nilesh Kulkarni, Ishan Misra, Shubham Tulsiani, Abhinav Gupta.

[Project Page](https://nileshkulkarni.github.io/relative3d/)

![Teaser Image](https://nileshkulkarni.github.io/relative3d/resources/images/teaser.png)

## Demo and Pre-trained Models

Please check out the [interactive notebook suncg](demo/demo_suncg.ipynb), [interactive notebook nyu](demo/demo_nyu.ipynb) which shows reconstructions using the learned models. To run this, you'll first need to follow the [installation instructions](docs/installation.md) to download trained models and some pre-requisites.

## Training and Evaluating
To train or evaluate the (trained/downloaded) models, it is first required to [download the SUNCG dataset](https://github.com/shubhtuls/factored3d/blob/master/docs/suncg_data.md) and [preprocess the data](https://github.com/shubhtuls/factored3d/blob/master/docs/preprocessing.md) and download the splits [here](https://cmu.box.com/s/zd0mkzghishx7qa82yz0ubh1orts6bx4). Please see the detailed README files for [Training](docs/training.md) or [Evaluation](docs/evaluation.md) of models for subsequent instructions.

To train or evaluate on the NYUv2 dataset the (trained/downloaded) models, it is first required to [download the NYU dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [preprocess the data](docs/preprocess_nyu.md) and download the splits [here](https://cmu.box.com/s/3igy99ghe0mxrdpmtynt209f4mpvaiac). Please see the detailed README files for [Training](docs/training.md) or [Evaluation](docs/evaluation.md) of models for subsequent instructions.
### Citation
If you use this code for your research, please consider citing:
```

@article{kulkarni20193d,
  title={3D-RelNet: Joint Object and Relational Network for 3D Prediction},
  author={Kulkarni, Nilesh
  and Misra, Ishan 
  and Tulsiani, Shubham
  and Gupta, Abhinav},
  journal={International Conference on Computer Vision (ICCV)}
  year={2019}
}

```
