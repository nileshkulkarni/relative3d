from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import collections

import scipy.misc
import scipy.linalg
import scipy.io as sio
import scipy.ndimage.interpolation
from absl import flags
import cPickle as pkl
import torch
import multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Manager
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pdb
from datetime import datetime
import sys
from ..utils import nyu_parse
from ..utils import suncg_parse

from ..renderer import utils as render_utils

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_string('nyu_dir', '/data0/shubhtuls/datasets/suncg_pbrs_release', 'Suncg Data Directory')
flags.DEFINE_boolean('filter_objects', True, 'Restrict object classes to main semantic classes.')
flags.DEFINE_integer('max_views_per_house', 0, '0->use all views. Else we randomly select upto the specified number.')

flags.DEFINE_boolean('suncg_dl_out_codes', True, 'Should the data loader load codes')
flags.DEFINE_boolean('suncg_dl_out_paths', False, 'Should the data loader load  paths')
flags.DEFINE_boolean('suncg_dl_out_layout', False, 'Should the data loader load layout')
flags.DEFINE_boolean('suncg_dl_out_depth', False, 'Should the data loader load modal depth')
flags.DEFINE_boolean('suncg_dl_out_fine_img', True, 'We should output fine images')
flags.DEFINE_boolean('suncg_dl_out_voxels', False, 'We should output scene voxels')
flags.DEFINE_boolean('suncg_dl_out_proposals', False, 'We should edgebox proposals for training')
flags.DEFINE_boolean('suncg_dl_out_only_pos_proposals', True, 'We should only output +ve edgebox proposals for training')
flags.DEFINE_boolean('suncg_dl_out_test_proposals', False, 'We should edgebox proposals for testing')
flags.DEFINE_integer('suncg_dl_max_proposals', 40, 'Max number of proposals per image')

flags.DEFINE_integer('img_height', 128, 'image height')
flags.DEFINE_integer('img_width', 256, 'image width')

flags.DEFINE_integer('img_height_fine', 480, 'image height')
flags.DEFINE_integer('img_width_fine', 640, 'image width')
flags.DEFINE_integer('nyu_img_height_fine', 480, 'image height')
flags.DEFINE_integer('nyu_img_width_fine', 640, 'image width')

flags.DEFINE_integer('layout_height', 64, 'amodal depth height : should be half image height')
flags.DEFINE_integer('layout_width', 128, 'amodal depth width : should be half image width')

flags.DEFINE_integer('max_object_classes', 10, 'maximum object classes')

flags.DEFINE_integer('voxels_height', 32, 'scene voxels height. Should be half of width and depth.')
flags.DEFINE_integer('voxels_width', 64, 'scene voxels width')
flags.DEFINE_integer('voxels_depth', 64, 'scene voxels depth')
flags.DEFINE_boolean('suncg_dl_debug_mode', False, 'Just running for debugging, should not preload ojects')

flags.DEFINE_boolean('use_trans_scale', False, 'scale trans to 0-1')
flags.DEFINE_boolean('relative_trj', True, 'use relative trjs')

#------------- Dataset ------------#
#----------------------------------#
class NYUDataset(Dataset):
    '''SUNCG data loader'''
    def __init__(self, image_names, opts):
        self._nyu_dir = opts.nyu_dir
        self._image_names = image_names
        self.img_size = (opts.img_height, opts.img_width)
        self.output_fine_img = opts.suncg_dl_out_fine_img
        if self.output_fine_img:
            self.img_size_fine = (opts.img_height_fine, opts.img_width_fine)
        self.output_codes = opts.suncg_dl_out_codes
        self.output_layout = opts.suncg_dl_out_layout
        self.output_modal_depth = opts.suncg_dl_out_depth
        self.output_voxels = opts.suncg_dl_out_voxels
        self.output_proposals = opts.suncg_dl_out_proposals
        self.output_test_proposals = opts.suncg_dl_out_test_proposals
        self.output_paths = opts.suncg_dl_out_paths
        self.relative_trj = opts.relative_trj
        self.use_trans_scale = opts.use_trans_scale
        self.max_object_classes = opts.max_object_classes
        self.only_pos_proposals = opts.suncg_dl_out_only_pos_proposals
        self.Adict = {}
        self.lmbda = 1
        self.opts  = opts
        
        if self.output_proposals:
            self.max_proposals = opts.suncg_dl_max_proposals
        if self.output_codes:
            self.max_rois = opts.max_rois
            
            self._obj_loader = nyu_parse.ObjectLoader(osp.join(opts.nyu_dir, 'object_obj'))
            if not opts.suncg_dl_debug_mode:
                self._obj_loader.preload()
            if opts.filter_objects:
                self._meta_loader = nyu_parse.MetaLoader()
            else:
                self._meta_loader = None

        data_tuples = []
        for ix, image in enumerate(image_names):
            if (ix % 500) == 0:
                print('Reading image names {}/{}'.format(ix, len(image_names)))
            data_tuples.append(image.replace(".png",""))

        self._data_tuples = data_tuples
        self.n_imgs = len(data_tuples)

        print('Using object classes {}'.format(nyu_parse.valid_object_classes))



    def forward_img(self, index):
        image_name = self._data_tuples[index]
        img = scipy.misc.imread(osp.join(self._nyu_dir, 'images', image_name + ".png"))
        img_size = img.shape
        # pdb.set_trace()
        bbox_rescaling = np.array(self.img_size_fine)/np.array([img_size[0], img_size[1]], dtype=np.float32)
        if self.output_fine_img:
            img_fine = scipy.misc.imresize(img, self.img_size_fine)
            img_fine = np.transpose(img_fine, (2,0,1))

        img = scipy.misc.imresize(img, self.img_size)
        img = np.transpose(img, (2,0,1))
        if self.output_fine_img:
            return img/255, img_fine/255, image_name, bbox_rescaling
        else:
            return img/255, image_name, img_size, bbox_rescaling

    def forward_codes(self, image_name):
        opts = self.opts
        data_file = osp.join(self._nyu_dir, 'img_data', image_name + '.pkl')
        if osp.exists(data_file):
            with open(osp.join(self._nyu_dir, 'img_data', image_name + '.pkl'), 'rb') as f:
                image_data = pkl.load(f)
            object_data  = nyu_parse.select_ids(image_data['objects'], metaloader=self._meta_loader, min_pixels=500)
            objects_codes, objects_bboxes = nyu_parse.codify_room_data(object_data, self._obj_loader)
            if len(objects_bboxes) > 0:
                objects_bboxes -= 1 #0 indexing to 1 indexing
                if len(objects_codes) > self.max_rois:
                    select_inds = np.random.permutation(len(objects_codes))[0:self.max_rois]
                    objects_bboxes = objects_bboxes[select_inds, :].copy()
                    objects_codes = [objects_codes[ix] for ix in select_inds]
            return objects_codes, objects_bboxes
        else:
            return [], []

    def forward_proposals(self, image_name, codes_gt, bboxes_gt, bbox_rescaling):
        proposals_data = sio.loadmat(
            osp.join(self._nyu_dir, 'fast_rcnn_proposals',  image_name + '_propsals.mat'))
        bboxes_proposals = proposals_data['boxes'][:,0:4]
        bboxes_scores = proposals_data['score']
        bboxes_proposals -= 1 #zero indexed
        bboxes_proposals = self.rescale_bboxes(bboxes_proposals, bbox_rescaling)
        codes, bboxes, labels = suncg_parse.extract_proposal_codes(
            codes_gt, bboxes_gt, bboxes_proposals, self.max_proposals,
            only_pos_proposals = self.only_pos_proposals)
        return codes, bboxes, labels

    def forward_test_proposals(self, image_name, bbox_rescaling):
        proposals_data = sio.loadmat(
            osp.join(self._nyu_dir, 'fast_rcnn_proposals',  image_name + '_propsals.mat'))
        bboxes_proposals = proposals_data['boxes'][:,0:4]
        bboxes_scores = proposals_data['score']
        bboxes_proposals -= 1 #zero indexed
        bboxes_proposals = self.rescale_bboxes(bboxes_proposals, bbox_rescaling)
        return bboxes_proposals, bboxes_scores

    def rescale_bboxes(self, bboxes, scaling):
        bboxes[:,0] = bboxes[:,0] * scaling[0]
        bboxes[:,2] = bboxes[:,2] * scaling[0]
        bboxes[:,1] = bboxes[:,1] * scaling[1]
        bboxes[:,3] = bboxes[:,3] * scaling[1]
        return bboxes

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, index):
        # pdb.set_trace()
        # image_name = 'img_6416'
        # for ix, (inx) in enumerate(self._data_tuples):
        #     if inx ==image_name:
        #         index = ix
        #         print('Using a fixed house')
        #         break
       
        if self.output_fine_img:
            img, img_fine, image_name, bbox_rescaling = self.forward_img(index)
        else:
            img, image_name, bbox_rescaling = self.forward_img(index)


        # bbox_rescaling -- y, x
        bbox_rescaling[0], bbox_rescaling[1] = bbox_rescaling[1], bbox_rescaling[0] 

        # print('Starting {} {}_{}, {}'.format(str(datetime.now()), house_name, view_id, multiprocessing.current_process()))
        # sys.stdout.flush()
        elem = {
            'img': img,
            'image_name': image_name,
        }


        if self.output_codes:
            valid = True
            codes_gt, bboxes_gt = self.forward_codes(image_name)
            # pdb.set_trace()
            if len(bboxes_gt) > 0:
                bboxes_gt[:,0] = bboxes_gt[:,0] * bbox_rescaling[0]
                bboxes_gt[:,2] = bboxes_gt[:,2] * bbox_rescaling[0]
                bboxes_gt[:,1] = bboxes_gt[:,1] * bbox_rescaling[1]
                bboxes_gt[:,3] = bboxes_gt[:,3] * bbox_rescaling[1]
            elem['codes'] = codes_gt
            elem['bboxes'] = bboxes_gt
            if len(elem['bboxes']) == 0: # Ensures that every images has some-information to be help in the loss.
                elem['bboxes'] = []
                valid = False

        if self.output_proposals:
            valid = True
            codes_proposals, bboxes_proposals, labels_proposals = self.forward_proposals(
               image_name, codes_gt, bboxes_gt, bbox_rescaling)
            if labels_proposals.size == 0:
                # print('No proposal found: ', house_name, view_id, labels_proposals, bboxes_proposals)
                bboxes_proposals = []
                labels_proposals = []
                valid = False
            elem['codes_proposals'] = codes_proposals
            elem['bboxes_proposals'] = bboxes_proposals
            elem['labels_proposals'] = labels_proposals
            

            # elem['codes_proposals'] = codes_gt
            # elem['bboxes_proposals'] = bboxes_gt
            # elem['labels_proposals'] = np.zeros((len(bboxes_gt),)) + 1
            # if len(elem['bboxes_proposals']) == 0: # Ensures that every images has some-information to be help in the loss.
            #     elem['bboxes_proposals'] = []
            #     elem['labels_proposals'] = []
            #     valid = False

        if self.output_test_proposals:
            bboxes_proposals, scores = self.forward_test_proposals(image_name, bbox_rescaling)
            if bboxes_proposals.size == 0:
                print('No proposal found: ', image_name)
                bboxes_proposals = []
                valid = False

            elem['bboxes_test_proposals'] = bboxes_proposals
            elem['scores'] = np.array(scores.reshape(-1), copy=True)
            
        if self.output_fine_img:
            elem['img_fine'] = img_fine
        return (valid, elem)


#-------- Collate Function --------#
#----------------------------------#    
def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    elif elem is None:
        return elem
    else:
        return elem

def collate_fn(batch):
    '''NYU data collater.
    
    Assumes each instance is a dict.
    Applies different collation rules for each field.

    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {'empty' : True}
    # iterate over keys
    new_batch = []
    for valid,t in batch:
        if valid:
            new_batch.append(t)
        else:
            'Print, found a empty in the batch'

    # batch = [t for t in batch if t is not None]
    # pdb.set_trace()
    batch = new_batch
    if len(batch) > 0:
        for key in batch[0]:
            if key =='codes' or key=='bboxes' or key=='codes_proposals' or key=='bboxes_proposals'\
                    or key=='bboxes_test_proposals' or key=='trajectory_masks' or key=='trajectories'\
                    or key=='extra_codes' or key == 'trajectory_offsets' or key=='pwd':
                collated_batch[key] = [recursive_convert_to_torch(elem[key]) for elem in batch]
                # pdb.set_trace()
            elif key == 'labels_proposals':
                collated_batch[key] = torch.cat([default_collate(elem[key]) for elem in batch if elem[key].size > 0])
            else:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
        collated_batch['empty']  = False
    return collated_batch

#----------- Data Loader ----------#
#----------------------------------#
def nyu_data_loader(image_names, opts):
    dset = NYUDataset(image_names, opts)
    return DataLoader(
        dset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.n_data_workers,
        collate_fn=collate_fn, pin_memory=True)


def nyu_data_loader_benchmark(image_names, opts):
    dset = NYUDataset(image_names, opts)
    return DataLoader(
        dset, batch_size=opts.batch_size,
        shuffle=False, num_workers=opts.n_data_workers,
        collate_fn=collate_fn)


def define_spatial_image(img_height, img_width, spatial_scale):
        img_height = int(img_height * spatial_scale)
        img_width = int(img_width * spatial_scale)
        spatial_h = torch.arange(0, img_height).unsqueeze(1).expand(torch.Size([img_height, img_width]))
        spatial_w = torch.arange(0, img_width).unsqueeze(0).expand(torch.Size([img_height, img_width]))
        spatial_h /= img_height
        spatial_w /= img_width
        spatial_image = torch.stack([spatial_h, spatial_w])
        return spatial_image
