
Please get the nyuv2 images from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

Download the [proposals](https://cmu.box.com/s/vs9mxylim1v09xeld8gk1p249pdqqlqk) and [object data](https://cmu.box.com/s/4gb25203vltk3maykd9p7mell0gpfjh9) for nyu

```
tar -xf nyuv2.tar.gz nyuv2
cd nyuv2
```
Download the [splits](https://cmu.box.com/s/mz64rgzheifglkxjytv8llvoupqfk0t3). Please make sure the splits are in the nyuv2 dir
```
tar -xf nyu_splits.tar.gz splits
```

Move NYU images to nyuv2 dir.


Preprocess the NYU data. Create the mat files using the script

```
python voxelize_objects.py
```




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
