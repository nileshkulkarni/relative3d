
Please get the nyuv2 images from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

Download the proposals and object data for nyu from here

```
wget https://www.dropbox.com/s/a0bqx3nxu1iaory/nyuv2.tar.gz && tar -xf nyuv2.tar.gz nyuv2
```

```
cd nyuv2
wget https://www.dropbox.com/s/mhvu39z1rhqmfox/nyu_splits.tar.gz && tar -xf nyu_splits.tar.gz splits
```

Move NYU images to nyuv2 dir.




Please consider citing if you use this for your reserach

```
@inproceedings{gupta2014learning,
  title={Learning rich features from RGB-D images for object detection and segmentation},
  author={Gupta, Saurabh and Girshick, Ross and Arbel{\'a}ez, Pablo and Malik, Jitendra},
  booktitle={European Conference on Computer Vision},
  pages={345--360},
  year={2014},
  organization={Springer}
}
```
