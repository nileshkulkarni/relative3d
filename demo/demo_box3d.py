
code_root='/home/nileshk/Research3/3dRelnet/relative3d'
import sys
import numpy as np
import os.path as osp
import scipy.misc
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
sys.path.append(osp.join(code_root, '..'))
import pdb
from absl import flags
from relative3d.demo import demo_utils

flags.FLAGS(['demo'])
opts =  flags.FLAGS
opts.batch_size = 1
opts.num_train_epoch = 8
opts.name = 'box3d_base_spatial_mask_common_upsample_t2'
opts.classify_rot = True
opts.classify_dir = True
opts.pred_voxels = False
opts.use_context = True

opts.upsample_mask=True
opts.pred_relative=True
opts.use_mask_in_common=True
opts.use_spatial_map=True

opts.pretrained_shape_decoder=True


if opts.classify_rot:
    opts.nz_rot = 24
else:
    opts.nz_rot = 4



## Load the trained models
tester = demo_utils.DemoTester(opts)
tester.init_testing()
pdb.set_trace()

# renderer = demo_utils.DemoRenderer(opts)
## Load input data
dataset = 'suncg'

img = scipy.misc.imread('./data/{}_img.png'.format(dataset))

img_fine = scipy.misc.imresize(img, (opts.img_height_fine, opts.img_width_fine))
img_fine = np.transpose(img_fine, (2,0,1))

img_coarse = scipy.misc.imresize(img, (opts.img_height, opts.img_width))
img_coarse = np.transpose(img_coarse, (2,0,1))

pdb.set_trace()
proposals = sio.loadmat('./data/{}_proposals.mat'.format(dataset))['proposals'][:, 0:4]

inputs = {}
inputs['img'] = torch.from_numpy(img_coarse/255.0).unsqueeze(0)
inputs['img_fine'] = torch.from_numpy(img_fine/255.0).unsqueeze(0)
inputs['bboxes'] = [torch.from_numpy(proposals)]

inputs['empty'] = False
tester.set_input(inputs)

pdb.set_trace()
objects = tester.predict_box3d()
# img_factored_cam, img_factored_novel = renderer.render_factored3d(objects)


f, axarr = plt.subplots(2, 4, figsize=(20, 8))


axarr[0, 1].imshow(img_factored_cam)
axarr[0, 1].axis('off')
axarr[1, 1].imshow(img_factored_novel)
axarr[1, 1].axis('off')

plt.show()