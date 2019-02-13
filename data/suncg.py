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
from ..utils import suncg_parse

from ..renderer import utils as render_utils

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_string('suncg_dir', '/data0/shubhtuls/datasets/suncg_pbrs_release', 'Suncg Data Directory')
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
class SuncgDataset(Dataset):
    '''SUNCG data loader'''
    def __init__(self, house_names, opts):
        self._suncg_dir = opts.suncg_dir
        self._house_names = house_names
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
        if self.output_layout or self.output_modal_depth:
            self.layout_size = (opts.layout_height, opts.layout_width)
        if self.output_voxels:
            self.voxels_size = (opts.voxels_width, opts.voxels_height, opts.voxels_depth)

        if self.output_proposals:
            self.max_proposals = opts.suncg_dl_max_proposals
        if self.output_codes:
            self.max_rois = opts.max_rois
            self._obj_loader = suncg_parse.ObjectLoader(osp.join(opts.suncg_dir, 'object'))
            if not opts.suncg_dl_debug_mode:
                self._obj_loader.preload()
            if opts.filter_objects:
                self._meta_loader = suncg_parse.MetaLoader(osp.join(opts.suncg_dir, 'ModelCategoryMappingEdited.csv'))
            else:
                self._meta_loader = None

        data_tuples = []
        for hx, house in enumerate(house_names):
            if (hx % 1000) == 0:
                print('Reading image names from house {}/{}'.format(hx, len(house_names)))
            imgs_dir = osp.join(opts.suncg_dir, 'renderings_ldr', house)
            view_ids = [f[0:6] for f in os.listdir(imgs_dir)]
            np.random.seed(0)
            view_ids.sort()
            rng = np.random.RandomState([ord(c) for c in house])
            rng.shuffle(view_ids)
            # if house == 'ffdbe78368fcf4488e9f930efb82f0e0':
            #     pdb.set_trace()

            if (opts.max_views_per_house > 0) and (opts.max_views_per_house < len(view_ids)):
                view_ids = view_ids[0:opts.max_views_per_house]
            for view_id in view_ids:
                data_tuples.append((house, view_id))
        self.n_imgs = len(data_tuples)
        self._data_tuples = data_tuples
        self._preload_cameras(house_names)
        print('Using object classes {}'.format(suncg_parse.valid_object_classes))



    def forward_img(self, index):
        house, view_id = self._data_tuples[index]
        try:
            img = scipy.misc.imread(osp.join(self._suncg_dir, 'renderings_ldr', house, view_id + '_mlt.jpg'))
        except:
            img = scipy.misc.imread(osp.join(self._suncg_dir, 'renderings_ldr', house, view_id + '_mlt.png'))
        if len(img.shape) == 2:
          ## Image is corrupted and it does not have third channel.
          ### Repeat the sample image 3 times.
          house, view_id = self._data_tuples[index]
          print("Corrupted Image Type 1 {} , {}".format(house, view_id))
          img = np.repeat(np.expand_dims(img,2),3, axis=2)

        if img.shape[2] == 2:
          house, view_id = self._data_tuples[index]
          print("Corrupted Image Type 2 {} , {}".format(house, view_id))
          img = np.concatenate((img, img[:, :, 0:1]), axis=2)


        if self.output_fine_img:
            img_fine = scipy.misc.imresize(img, self.img_size_fine)
            img_fine = np.transpose(img_fine, (2,0,1))

        img = scipy.misc.imresize(img, self.img_size)
        img = np.transpose(img, (2,0,1))
        if self.output_fine_img:
            return img/255, img_fine/255, house, view_id
        else:
            return img/255, house, view_id

    def _preload_cameras(self, house_names):
        self._house_cameras = {}
        for hx, house in enumerate(house_names):
            if (hx % 200) == 0:
                print('Pre-loading cameras from house {}/{}'.format(hx, len(house_names)))
            cam_file = osp.join(self._suncg_dir, 'camera', house, 'room_camera.txt')
            camera_poses = suncg_parse.read_camera_pose(cam_file)
            self._house_cameras[house] = camera_poses

    def forward_trajectories(self, house_name, view_id):
        trajectories = None
        trj_dir = 'trajectory_centers'
        # trj_dir = 'trajectory_diagnoals'
        # trj_dir = 'trajectory_a_path
        # pdb.set_trace()
        if osp.exists(osp.join(self._suncg_dir, trj_dir ,
                     house_name, '{}.json'.format(view_id))):
            trajectories = suncg_parse.load_json(
                osp.join(self._suncg_dir, trj_dir,
                         house_name, '{}.json'.format(view_id)))
            trajectories['nodes'] = {int(key): int(value) for key, value in trajectories['nodes'].items()}
            trajectories['real_object_locations'] = {int(key): value for key, value in trajectories['real_object_locations'].items()}
        return  trajectories


    def forward_codes(self, house_name, view_id):
        campose = self._house_cameras[house_name][int(view_id)]
        cam2world = suncg_parse.campose_to_extrinsic(campose).astype(np.float32)
        world2cam = scipy.linalg.inv(cam2world).astype(np.float32)

        trajectories = None
        house_data = suncg_parse.load_json(
            osp.join(self._suncg_dir, 'house', house_name, 'house.json'))
        bbox_data = sio.loadmat(
            osp.join(self._suncg_dir, 'bboxes_node', house_name, view_id + '_bboxes.mat'))

        objects_data, objects_bboxes, trajectory_data = suncg_parse.select_ids(
            house_data, bbox_data, trajectories, meta_loader=self._meta_loader, min_pixels=500)
        
        objects_codes, _ = suncg_parse.codify_room_data(
            objects_data, world2cam, self._obj_loader,
            max_object_classes = self.max_object_classes)

        # objects_paths = suncg_parse.codify_path_data(world2cam, trajectory_data, objects_codes, relative_trj=self.relative_trj)
        objects_bboxes -= 1 #0 indexing to 1 indexing
        if len(objects_codes) > self.max_rois:
            select_inds = np.random.permutation(len(objects_codes))[0:self.max_rois]
            objects_bboxes = objects_bboxes[select_inds, :].copy()
            objects_codes = [objects_codes[ix] for ix in select_inds]
            # extra_codes = (extra_codes[0], extra_codes[1],)  + tuple([[extra_codes[i][temp_i] for temp_i in select_inds] for i in range(2,8)])

        # return objects_codes, objects_bboxes, extra_codes
        return objects_codes, objects_bboxes

    def forward_proposals(self, house_name, view_id, codes_gt, bboxes_gt):
        proposals_data = sio.loadmat(
            osp.join(self._suncg_dir, 'edgebox_proposals', house_name, view_id + '_proposals.mat'))
        bboxes_proposals = proposals_data['proposals'][:,0:4]
        bboxes_proposals -= 1 #zero indexed
        codes, bboxes, labels = suncg_parse.extract_proposal_codes(
            codes_gt, bboxes_gt, bboxes_proposals, self.max_proposals,
            only_pos_proposals = self.only_pos_proposals)
        return codes, bboxes, labels
    
    def forward_test_proposals(self, house_name, view_id):
        proposals_data = sio.loadmat(
            osp.join(self._suncg_dir, 'edgebox_proposals', house_name, view_id + '_proposals.mat'))
        bboxes_proposals = proposals_data['proposals'][:,0:4]
        bboxes_proposals -= 1 #zero indexed
        return bboxes_proposals


    def forward_layout(self, house_name, view_id, bg_depth=1e4):
        depth_im = scipy.misc.imread(osp.join(
            self._suncg_dir, 'renderings_layout', house_name, view_id + '_depth.png'))
        depth_im =  depth_im.astype(np.float)/1000.0  # depth was saved in mm
        depth_im += bg_depth*np.equal(depth_im,0).astype(np.float)
        disp_im = 1./depth_im
        amodal_depth = scipy.ndimage.interpolation.zoom(
            disp_im, (self.layout_size[0]/disp_im.shape[0], self.layout_size[1]/disp_im.shape[1]), order=0)
        amodal_depth = np.reshape(amodal_depth, (1, self.layout_size[0], self.layout_size[1]))
        return amodal_depth

    def forward_depth(self, house_name, view_id, bg_depth=1e4):
        depth_im = scipy.misc.imread(osp.join(
            self._suncg_dir, 'renderings_depth', house_name, view_id + '_depth.png'))
        depth_im =  depth_im.astype(np.float)/1000.0  # depth was saved in mm
        depth_im += bg_depth*np.equal(depth_im,0).astype(np.float)
        disp_im = 1./depth_im
        modal_depth = scipy.ndimage.interpolation.zoom(
            disp_im, (self.layout_size[0]/disp_im.shape[0], self.layout_size[1]/disp_im.shape[1]), order=0)
        modal_depth = np.r
        eshape(modal_depth, (1, self.layout_size[0], self.layout_size[1]))
        return modal_depth

    def forward_max_depth(self, house_name, view_id, bg_depth=1e4):
        depth_im = scipy.misc.imread(osp.join(
            self._suncg_dir, 'renderings_layout', house_name,
            view_id + '_depth.png'))
        depth_im = depth_im.astype(np.float) / 1000.0  # depth was saved in mm
        max_depth = np.max(depth_im)
        if max_depth < 1E-3:
            max_depth = np.max(depth_im) + 1000
        return max_depth
    
    def forward_voxels(self, house_name, view_id):
        scene_voxels = sio.loadmat(osp.join(
            self._suncg_dir, 'scene_voxels', house_name, view_id + '_voxels.mat'))
        scene_voxels = render_utils.downsample(
            scene_voxels['sceneVox'].astype(np.float32),
            64//self.voxels_size[1], use_max=True)
        return scene_voxels


    def __len__(self):
        return self.n_imgs

    def __getitem__(self, index):
        # pdb.set_trace()
        #ffdbe78368fcf4488e9f930efb82f0e0_000003
        # 5fcc3f555c3172c80bcc0de4f59b64a1_000001
        # cf869fbab53178098abda4bd6a6e3b19_000001
        # e442c931d7e5b8c56b9c0523a619c21e_000019
        # house_name = "6ba5f2a6027488989b361b8a50922f93"
        # view_id = "000017"
        # for ix, (hnx, vix) in enumerate(self._data_tuples):
        #     if hnx ==house_name and vix == view_id:
        #         index = ix
        #         print('Using a fixed house')
        #         break

        if self.output_fine_img:
            img, img_fine, house_name, view_id = self.forward_img(index)
        else:
            img, house_name, view_id = self.forward_img(index)

        # print('Starting {} {}_{}, {}'.format(str(datetime.now()), house_name, view_id, multiprocessing.current_process()))
        # sys.stdout.flush()
        elem = {
            'img': img,
            'house_name': house_name,
            'view_id': view_id,
        }

        if self.output_layout:
            layout = self.forward_layout(house_name, view_id)
            elem['layout'] = layout

        if self.output_voxels:
            voxels = self.forward_voxels(house_name, view_id)
            elem['voxels'] = voxels

        if self.output_modal_depth:
            depth = self.forward_depth(house_name, view_id)
            elem['depth'] = depth

        if self.output_codes:
            valid = True
            # codes_gt, bboxes_gt, extra_codes = self.forward_codes(house_name, view_id)
            codes_gt, bboxes_gt = self.forward_codes(house_name, view_id)
            elem['codes'] = codes_gt
            elem['bboxes'] = bboxes_gt
            # elem['extra_codes'] = extra_codes
            
            # valid = len(elem['bboxes']) > 0
            if len(elem['bboxes']) == 0: # Ensures that every images has some-information to be help in the loss.
                elem['bboxes'] = []
                valid = False


        if self.output_proposals:
            valid = True
            codes_proposals, bboxes_proposals, labels_proposals = self.forward_proposals(
                house_name, view_id, codes_gt, bboxes_gt)
            if labels_proposals.size == 0:
                # print('No proposal found: ', house_name, view_id, labels_proposals, bboxes_proposals)
                bboxes_proposals = []
                labels_proposals = []
                valid = False
            elem['codes_proposals'] = codes_proposals
            elem['bboxes_proposals'] = bboxes_proposals
            elem['labels_proposals'] = labels_proposals
            # pdb.set_trace()

            # elem['codes_proposals'] = codes_gt
            # elem['bboxes_proposals'] = bboxes_gt
            # elem['labels_proposals'] = np.zeros((len(bboxes_gt),)) + 1
            # if len(elem['bboxes_proposals']) == 0: # Ensures that every images has some-information to be help in the loss.
            #     elem['bboxes_proposals'] = []
            #     elem['labels_proposals'] = []
            #     valid = False

        if self.output_test_proposals:
            bboxes_proposals = self.forward_test_proposals(house_name, view_id)
            if bboxes_proposals.size == 0:
                print('No proposal found: ', house_name, view_id)
                bboxes_proposals = []
                valid = False

            elem['bboxes_test_proposals'] = bboxes_proposals

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
    '''SUNCG data collater.
    
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
def suncg_data_loader(house_names, opts):
    dset = SuncgDataset(house_names, opts)
    return DataLoader(
        dset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.n_data_workers,
        collate_fn=collate_fn, pin_memory=True)


def suncg_data_loader_benchmark(house_names, opts):
    dset = SuncgDataset(house_names, opts)
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
