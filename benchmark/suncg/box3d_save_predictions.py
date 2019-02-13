
"""Script for dwr prediction benchmarking.
"""
# Sample usage:
# (shape_ft) : python -m factored3d.benchmark.suncg.dwr --num_train_epoch=1 --name=dwr_shape_ft --classify_rot --pred_voxels=True --use_context  --save_visuals --visuals_freq=50 --eval_set=val  --suncg_dl_debug_mode  --max_eval_iter=20
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import time
import scipy.misc
import pdb
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
from ...data import suncg as suncg_data
from . import evaluate_detection
from ...utils import bbox_utils
from ...utils import suncg_parse
from ...nnutils import test_utils
from ...nnutils import net_blocks
from ...nnutils import loss_utils
from ...nnutils import oc_net
from ...nnutils import disp_net
from ...utils import metrics
from ...utils import visutil
from ...renderer import utils as render_utils
from ...utils import quatUtils
import cv2
from ...utils import transformations
from collections import Counter
from six.moves import cPickle as pickle
import collections

q2e = quatUtils.convert_quat_to_euler

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'),
                    'Directory where intermittent renderings are saved')

flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')
flags.DEFINE_float('voxel_eval_thresh', 0.25, 'Voxel evaluation threshold')
flags.DEFINE_string('id', 'default', 'Plot string')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')

flags.DEFINE_integer('max_rois', 100, 'If we have more objects than this per image, we will subsample.')
flags.DEFINE_integer('max_total_rois', 100, 'If we have more objects than this per batch, we will reject the batch.')
flags.DEFINE_integer('num_visuals', 200, 'Number of renderings')
flags.DEFINE_boolean('preload_stats', False, 'Reload the stats for the experiment')
flags.DEFINE_string('layout_name', 'layout_pred', 'Experiment name for layout predictor')
flags.DEFINE_integer('layout_train_epoch', 8, 'Experiment name for layout predictor')
flags.DEFINE_boolean('use_gt_voxels', True, 'Use gt_voxels_for_prediction')
flags.DEFINE_string('ovis_ids_filename', None, 'Ids to visualize output file')
flags.DEFINE_string('ivis_ids_filename', None, 'Ids to visualize output file')
flags.DEFINE_string('results_name', None, 'results_name')
flags.DEFINE_boolean('gt_updates', False, 'Use gt_relative updates')
flags.DEFINE_boolean('do_updates', True, 'Do opt updates')
flags.DEFINE_string('index_file', None, 'file containing house names and view ids')
flags.DEFINE_string('log_csv', None, 'file containing relative acc data')
flags.DEFINE_boolean('draw_vis', False, 'Do not evaluate only draw visualization')
flags.DEFINE_boolean('load_predictions_from_disk', False, 'Load pkl files')
flags.DEFINE_boolean('save_predictions_to_disk', True, 'Save pkl files')
flags.DEFINE_float('lambda_weight', 1.0, 'lambda for rotation')
flags.DEFINE_float('split_size', 1.0, 'Split size of the train set')
flags.DEFINE_boolean('only_pairs', True, 'Train with only more than 2 examples per ')
flags.DEFINE_boolean('dwr_model', False, 'Load a dwr mode ')
flags.DEFINE_boolean('pretrained_shape_decoder', True, 'Load pretrained shape decoder model, use only when you are using the detector trained on GT boxes')


FLAGS = flags.FLAGS

EP_box_iou_thresh = [0.5, 0.5, 0.5, 0.5, 0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ]
EP_rot_delta_thresh = [30., 30., 400., 30., 30., 30., 400., 30., 400., 400., 400., 30, ]
EP_trans_delta_thresh = [1., 1., 1., 1000., 1, 1., 1000., 1000., 1.0, 1000., 1000., 1000., ]
EP_shape_iou_thresh = [0.25, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0.25, 0, 0.25, ]
EP_scale_delta_thresh = [0.5, 0.5, 0.5, 0.5, 0.5, 100., 100., 100, 100, 100, 0.5, 100, ]
EP_ap_str = ['all', '-shape', '-rot', '-trans', '-box2d', '-scale', 'box2d',
    'box2d+rot', 'box2d+trans', 'box2d+shape', 'box2d+scale', 'box2d+rot+shape', ]


def my_print(tensor):
    try:
        print(np.round(tensor.numpy(),2))
    except:
        print(np.round(tensor, 2))
    return

class DWRTester(test_utils.Tester):

    def define_model(self):
        '''
        Define the pytorch net 'model' whose weights will be updated during training.
        '''
        self.eval_shape_iou = False
        opts = self.opts
        self.object_class2index = {'bed' : 1, 'sofa' :2, 'table' :3, 
            'chair':4 , 'desk':5, 'television':6,
        }

        self.index2object_class = {1: 'bed', 2 :'sofa', 3 : 'table', 
            4 :'chair', 5 : 'desk', 6 : 'television',
        }

        self.voxel_encoder, nc_enc_voxel = net_blocks.encoder3d(
            opts.n_voxel_layers, nc_max=opts.voxel_nc_max, nc_l1=opts.voxel_nc_l1, nz_shape=opts.nz_shape)

        self.voxel_decoder = net_blocks.decoder3d(
            opts.n_voxel_layers, opts.nz_shape, nc_enc_voxel, nc_min=opts.voxel_nc_l1)

        self.model = oc_net.OCNet(
            (opts.img_height, opts.img_width), opts=self.opts,
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=False, nz_shape=opts.nz_shape, pred_labels=True, pred_graph=opts.pred_graph,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot, n_g_layers=opts.n_g_layers,)
        #

        if opts.pred_voxels and opts.dwr_model:
            self.model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))

        if opts.dwr_model:
            # self.opts.num_train_epoch=1
            self.model.add_label_predictor()
            self.eval_shape_iou = True
            opts.use_gt_voxels = False

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        
        if not opts.dwr_model:
            self.model.add_label_predictor()
        
        if opts.pretrained_shape_decoder:
                self.model.code_predictor.shape_predictor.add_voxel_decoder(
                    copy.deepcopy(self.voxel_decoder))
                network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
                print('Loading shape decoder pretrained')
                self.load_network(
                    self.model.code_predictor.shape_predictor.decoder,
                    'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)

        self.model.eval()
        self.model = self.model.cuda()
        # self.model = self.model.cuda(device=self.opts.gpu_id)

        if opts.pred_voxels and (not opts.dwr_model):
             self.voxel_decoder = copy.deepcopy(self.model.code_predictor.shape_predictor.decoder)

        self.layout_model = disp_net.dispnet()
        network_dir = osp.join(opts.cache_dir, 'snapshots', opts.layout_name)
        self.load_network(
            self.layout_model, 'pred', opts.layout_train_epoch, network_dir=network_dir)
        # self.layout_model.eval()
        # self.layout_model = self.layout_model.cuda(device=self.opts.gpu_id)

        return

    def init_dataset(self):
        opts = self.opts
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        # houses_splits = self.split[opts.eval_set]
        if opts.eval_set == 'train':
            rng = np.random.RandomState(10)
            rng.shuffle(self.split[opts.eval_set])
            len_splitset = int(len(self.split[opts.eval_set])*opts.split_size)
            self.split[opts.eval_set] = self.split[opts.eval_set][0:len_splitset]
            # print(self.split[opts.eval_set])

        self.dataloader = suncg_data.suncg_data_loader_benchmark(self.split[opts.eval_set], opts)

        if opts.voxel_size < 64:
            self.downsample_voxels = True
            self.downsampler = render_utils.Downsample(
                64 // opts.voxel_size, use_max=True, batch_mode=True
            ).cuda()
        else:
            self.downsampler = None

        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)

        if not opts.pred_voxels:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.voxel_decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.voxel_decoder.eval()
            self.voxel_decoder = self.voxel_decoder.cuda()

        self.spatial_image = Variable(suncg_data.define_spatial_image(opts.img_height_fine, opts.img_width_fine, 1.0/16).unsqueeze(0).cuda()) ## (1, 2, 30, 40)
        
        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)
            if opts.nz_rot == 48:
                self.quat_medoids = torch.from_numpy(
                    scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids_48.mat'))['medoids']).type(torch.FloatTensor)

            nz_rel_rot = opts.nz_rel_rot
            self.quat_medoids_relative = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids_relative_{}_new.mat'.format(nz_rel_rot)))['medoids']).type(torch.FloatTensor)
            assert len(self.quat_medoids_relative) == opts.nz_rel_rot, ' Relative rotation architecture does not match'
            # self.quat_medoids_relative = torch.from_numpy(
            #     scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids_relative.mat'))['medoids']).type(torch.FloatTensor)
            self.quat_medoids_var = None

            # define the nearest bin metric?
            # n_absoulte_bins = len(self.quat_medoids)
            # quatsA = self.quat_medoids.unsqueeze(1).expand(torch.Size(
            #     [n_absoulte_bins, n_absoulte_bins, 4])).contiguous().view(-1, 4)
            # quatsB = self.quat_medoids.unsqueeze(0).expand(torch.Size(
            #     [n_absoulte_bins, n_absoulte_bins, 4])).contiguous().view(-1, 4)
            # quatsA_conjugate = quatUtils.quat_conjugate(quatsA)
            # relative_quats = quatUtils.rotate_quat(quatsB, quatsA_conjugate)
            # self.relative_quats_binids = suncg_parse.quats_to_bininds(relative_quats, self.quat_medoids_relative)
            # self.relative_quats_binids = self.relative_quats_binids.view(n_absoulte_bins, n_absoulte_bins)
        if opts.classify_dir:
            self.direction_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'direction_medoids_relative_{}_new.mat'.format(opts.nz_rel_dir)))['medoids']).type(torch.FloatTensor)
            self.direction_medoids = torch.nn.functional.normalize(self.direction_medoids)

        self.data_vis = []
        self.stored_quat_relative_gt_classes = []
        self.stored_quat_relative_pred_classes = []
        self.rotation_bins = []
        self.translation = []
        self.pred_translation = []
        self.pred_rotation = []
        self.pred_relative_directions = []
        self.relative_directions =[]
        return

    def decode_shape(self, pred_shape):
        opts = self.opts
        if opts.use_gt_voxels:
            # assert pred_shape.size() == self.codes_gt[0].size(), 'predict size from gt incorrect'
            return self.codes_gt['shape'].clone()

        pred_shape = torch.nn.functional.sigmoid(self.voxel_decoder.forward(pred_shape))
        return pred_shape

    def decode_rotation(self, pred_rot):
        opts = self.opts
        if opts.classify_rot:
            _, bin_inds = torch.max(pred_rot.data.cpu(), 1)
            pred_rot = Variable(suncg_parse.bininds_to_quats(
                bin_inds, self.quat_medoids), requires_grad=False)
        return pred_rot

    def decode_rotation_topk(self, pred_rot):
        opts = self.opts
        if opts.classify_rot:
            _, bin_inds = torch.topk(pred_rot.data.cpu(), k=2, dim=1)
            bin_inds = bin_inds.view(-1, 1)
            pred_rot = Variable(suncg_parse.bininds_to_quats(
                bin_inds, self.quat_medoids), requires_grad=False)
            pred_rot = pred_rot.view(-1, 2, 4)
        return pred_rot

    def get_class_indices(self, pred_rot):
        opts = self.opts
        _, bin_inds = torch.max(pred_rot.data.cpu(), 1)
        return bin_inds

    def decode_rotation_relative(self, pred_rot):
        opts = self.opts
        if opts.classify_rot:
            _, bin_inds = torch.max(pred_rot.data.cpu(), 1)
            pred_rot = Variable(suncg_parse.bininds_to_quats(
                bin_inds, self.quat_medoids_relative), requires_grad=False)
        return pred_rot

    def decode_class(self, pred_class):
        opts = self.opts
        # pdb.set_trace()
        _, bin_inds = torch.max(pred_class.data.cpu(), 1)
        return bin_inds

    def count_number_pairs(self, rois):
        counts = Counter(rois[:,0].numpy().tolist())
        pairs = sum([v*v for (k,v) in counts.items() if v > 1])
        return pairs

    def set_input(self, batch):
        opts = self.opts
        if batch is None or not batch:
            self.invalid_batch = True
            self.invalid_rois = None
            return

        if batch['empty']:
            self.invalid_rois = None
            self.invalid_batch = True
            return

        bboxes_gt = suncg_parse.bboxes_to_rois(batch['bboxes'])
        bboxes_proposals = suncg_parse.bboxes_to_rois(batch['bboxes_test_proposals'])
        bboxes_proposals = bboxes_gt
        rois = bboxes_proposals
        if rois.numel() <= 0 or bboxes_gt.numel() <= 0:  # some proposals and gt objects should be there
            self.invalid_batch = True
            self.invalid_rois = None
            return
        else:
            if bboxes_gt.numel() == 5 and self.opts.only_pairs: 
                self.invalid_rois = None
                self.invalid_batch = True
                return
            # if bboxes_gt.numel() > 8 * 5:
            #     self.invalid_batch = True
            #     return
            pairs = self.count_number_pairs(rois)
            # if pairs <= 1:
            #     self.invalid_batch = True
            #     self.invalid_rois = None
            #     return
            self.invalid_batch = False

        self.house_names = batch['house_name']
        self.view_ids = batch['view_id']
        # Inputs for prediction
        if self.opts.load_predictions_from_disk:
            return



        input_imgs_fine = batch['img_fine'].type(torch.FloatTensor)
        input_imgs = batch['img'].type(torch.FloatTensor)

        self.input_imgs_layout = Variable(
            input_imgs.cuda(), requires_grad=False)

        for b in range(input_imgs_fine.size(0)):
            input_imgs_fine[b] = self.resnet_transform(input_imgs_fine[b])
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        self.input_imgs = Variable(
            input_imgs.cuda(), requires_grad=False)

        self.input_imgs_fine = Variable(
            input_imgs_fine.cuda(), requires_grad=False)

        self.rois = Variable(
            rois.type(torch.FloatTensor).cuda(), requires_grad=False)


        code_tensors = suncg_parse.collate_codes(batch['codes'])
        code_tensors_quats = code_tensors['quat']
        self.amodal_bboxes =  code_tensors['amodal_bbox']
        object_classes = code_tensors['class'].type(torch.LongTensor)
        self.class_gt = self.object_classes = Variable(object_classes.cuda(), requires_grad=False)
       
        self.object_locations = suncg_parse.batchify(code_tensors['trans'], self.rois[:, 0].data.cpu())
        code_tensors['shape'] = code_tensors['shape'].unsqueeze(1)  # unsqueeze voxels
       
        vox2cams = code_tensors['transform_vox2cam']
        self.vox2cams = vox2cams =  suncg_parse.batchify(vox2cams, self.rois[:,0].data.cpu())
       
        cam2voxs = code_tensors['transform_cam2vox']
        cam2voxs = suncg_parse.batchify(cam2voxs, self.rois[:,0].data.cpu())
       
        relative_direction_rotation = []
        relative_dir_mask = []
        for bx in range(len(cam2voxs)):
            nx = len(cam2voxs[bx])
            mask = torch.ones([nx*nx])
            for ix in range(nx):
                for jx in range(nx):
                    if ix == jx:
                        mask[ix*nx + jx] = 0
                    directions = []
                    for cam2vox in cam2voxs[bx][ix]:
                        dt_trj_cam_frame = self.object_locations[bx][jx]
                        direction = self.homogenize_coordinates(dt_trj_cam_frame.unsqueeze(0))
                        direction = torch.matmul(cam2vox, direction.t()).t()[:,0:3]
                        direction = direction/(torch.norm(direction,p=2, dim=1, keepdim=True) + 1E-5)
                        directions.append(direction)
                    directions = torch.cat(directions)
                    relative_direction_rotation.append(directions)
            relative_dir_mask.append(mask)
        self.relative_dir_mask = Variable(torch.cat(relative_dir_mask,dim=0).byte()).cuda()

        assert self.opts.batch_size == 1, 'batch size > 1 not supported'

        if opts.classify_dir and not opts.gmm_dir:
            self.relative_direction_rotation = relative_direction_rotation
            relative_direction_rotation_binned  = [suncg_parse.directions_to_bininds(t, self.direction_medoids) for t in relative_direction_rotation]
            # relative_direction_rotation_directions = [suncg_parse.bininds_to_directions(t, self.direction_medoids) for t in relative_direction_rotation_binned]
            # ## compute some average error?
            # error = torch.cat([(1 - t1*t2.sum(1)).mean() for (t1, t2) in zip(relative_direction_rotation, relative_direction_rotation_directions)]).mean()
            # # pdb.set_trace()
            self.relative_direction_rotation_binned = [Variable(t).cuda() for t in relative_direction_rotation_binned]

        
        self.object_locations = [Variable(t_.cuda(), requires_grad=False) for t_ in
                                 self.object_locations]

        self.object_scales = suncg_parse.batchify(code_tensors['scale'] + 1E-10, self.rois[:, 0].data.cpu())
        self.object_scales = [Variable(t_.cuda(), requires_grad=False) for t_ in
                                 self.object_scales]

        self.relative_trans_gt = []
        self.relative_scale_gt = []
        for bx in range(len(self.object_locations)):
            relative_locations = self.object_locations[bx].unsqueeze(0) - self.object_locations[bx].unsqueeze(1)
            relative_locations = relative_locations.view(-1, 3)
            self.relative_trans_gt.append(relative_locations)
            # this is in log scale.
            relative_scales = self.object_scales[bx].unsqueeze(0).log() - self.object_scales[bx].unsqueeze(1).log()
            relative_scales = relative_scales.view(-1, 3)
            self.relative_scale_gt.append(relative_scales)

        self.relative_scale_gt = torch.cat(self.relative_scale_gt, dim=0)  # this is in log scale
        self.relative_trans_gt = torch.cat(self.relative_trans_gt, dim=0)
        self.relative_gt = {'relative_trans' : self.relative_trans_gt,
                            'relative_scale' : self.relative_scale_gt,
                            'relative_dir' : self.relative_direction_rotation,
                            'relative_mask' : self.relative_dir_mask,
                            }
    
        self.layout_gt=Variable(
            batch['layout'].cuda(), requires_grad=False)

        self.codes_gt_quats = [
            Variable(t.cuda(), requires_grad=False) for t in code_tensors_quats]
        codes_gt_keys = ['shape', 'scale', 'trans']
        self.codes_gt  ={key : Variable(code_tensors[key].cuda(), requires_grad=False) 
                            for key in codes_gt_keys}
        self.codes_gt['quat'] = self.codes_gt_quats

        self.rois_gt=Variable(
            bboxes_gt.type(torch.FloatTensor).cuda(), requires_grad=False)
        if self.downsample_voxels:
            self.codes_gt['shape']=self.downsampler.forward(self.codes_gt['shape'])
        return

    def convert_multiple_bins_to_probabilites(self, bins_ids, num_medoids, no_noise=1.0):
       
        bins = [torch.LongTensor([random.choice(c.data)]) for c in bins_ids]
        noise_values = torch.bernoulli(torch.FloatTensor(len(bins)).zero_() + no_noise)
        bins = [c if n > 0.5 else torch.LongTensor([np.random.randint(num_medoids)]) for c, n in zip(bins, noise_values)]
        bins = torch.cat(bins)
        probs = torch.FloatTensor(len(bins), num_medoids).zero_()
        probs.scatter_(1, bins.unsqueeze(1), 1-0.001*num_medoids)
        probs = probs + 0.001
        return probs

    '''
    args
        relative_directions : list N^2, torch.Tensor K x 3
        vox2cams : list N , K x 4,4
        img_size : (H, W)
    returns:
        relative_directions in image plane N^2 K x 2 x 2
    '''
    def convert_relative_vectors_to_image_plane(self, relative_directions, vox2cams, img_size):
        def convert_vector_to_image_plane(vector, vox2cam, cam_intrinsic, img_size):
            vector_cam_frame = suncg_parse.transform_coordinates(vox2cam, vector.reshape(1, -1))
            img_frame = suncg_parse.transform_to_image_coordinates(vector_cam_frame, cam_intrinsic)
            img_frame = np.clip(img_frame, a_min=np.array([[0,0]]), a_max=np.array([[img_size[1], img_size[0]]]))
            return img_frame

        cam_intrinsic = suncg_parse.cam_intrinsic()
        img_vectors = []
        n_objects = len(vox2cams)
        for ix, rel_dir in enumerate(relative_directions):
            rel_dir = rel_dir[0]
            vox2cam = vox2cams[ix//n_objects][0]
            src_vector = convert_vector_to_image_plane(np.array([0,0,0]), vox2cam.numpy(), cam_intrinsic, img_size)
            trj_vector = convert_vector_to_image_plane(rel_dir.numpy(), vox2cam.numpy(), cam_intrinsic, img_size)
            img_vectors.append(np.concatenate([src_vector, trj_vector], axis=0))

        index = [ix*n_objects + ix for ix in range(n_objects)]
        img_vectors = [(img_vectors[ix], ix//n_objects, ix % n_objects) for ix in range(n_objects*n_objects) if ix not in index]
        return img_vectors

    # def save_current_stats(self, bench):
    #     imgs_dir=osp.join(self.opts.results_vis_dir, 'vis_iter_{}'.format(self.vis_iter))
    #     if not os.path.exists(imgs_dir):
    #         os.makedirs(imgs_dir)
    #     json_file=os.path.join(imgs_dir, 'bench_iter_{}.json'.format(0))
    #     # print(json_file)
    #     with open(json_file, 'w') as f:
    #         json.dump({'bench': bench}, f)

    def save_layout_mesh(self, mesh_dir, layout, prefix='layout'):
        opts=self.opts
        layout_vis=layout.data[0].cpu().numpy().transpose((1, 2, 0))
        mesh_file=osp.join(mesh_dir, prefix + '.obj')
        vs, fs=render_utils.dispmap_to_mesh(
            layout_vis,
            suncg_parse.cam_intrinsic(),
            scale_x=self.opts.layout_width / 640,
            scale_y=self.opts.layout_height / 480
        )
        fout=open(mesh_file, 'w')
        mesh_file=osp.join(mesh_dir, prefix + '.obj')
        fout=open(mesh_file, 'w')
        render_utils.append_obj(fout, vs, fs)
        fout.close()

    def save_codes_mesh(self, mesh_dir, code_vars, prefix='codes'):
        opts=self.opts
        n_rois=code_vars['shape'].size()[0]
        code_list=suncg_parse.uncollate_codes(code_vars, self.input_imgs.data.size(0), torch.Tensor(n_rois).fill_(0))

        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        mesh_file=osp.join(mesh_dir, prefix + '.obj')
        new_codes_list = suncg_parse.convert_codes_list_to_old_format(code_list[0])
        render_utils.save_parse(mesh_file, new_codes_list, save_objectwise=False, thresh=0.1)


    def render_visuals(self, mesh_dir, obj_name=None):
        png_dir=osp.join(mesh_dir, 'rendering')
        if obj_name is not None:
            render_utils.render_mesh(osp.join(mesh_dir, obj_name + '.obj'), png_dir)
            im_view1=scipy.misc.imread(osp.join(png_dir, '{}_render_000.png'.format(obj_name)))
            # im_view2=scipy.misc.imread(osp.join(png_dir, '{}_render_003.png'.format(obj_name)))
        else:
            render_utils.render_directory(mesh_dir, png_dir)
            im_view1=scipy.misc.imread(osp.join(png_dir, 'render_000.png'))
            # im_view2=scipy.misc.imread(osp.join(png_dir, 'render_003.png'))

        # return im_view1, im_view2
        return im_view1



    def get_current_visuals(self):
        visuals={}
        opts=self.opts
        visuals['img']=visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))
        rois=self.rois.data
        visuals['img_roi']=render_utils.vis_detections(visuals['img'], rois[:, 1:])

        # img_rel_vectors_pred = self.convert_relative_vectors_to_image_plane([[x] for x in self.relative_direction_prediction_3d], 
        #     self.vox2cams[0], (self.opts.img_height_fine, self.opts.img_width_fine))
        # img_rel_vectors_gt = self.convert_relative_vectors_to_image_plane(self.relative_direction_rotation,
        #     self.vox2cams[0], (self.opts.img_height_fine, self.opts.img_width_fine))
        # visuals['img_rel_dir_pred']=render_utils.vis_relative_dirs(visuals['img_roi'], img_rel_vectors_pred)
        # visuals['img_rel_dir_gt']=render_utils.vis_relative_dirs(visuals['img_roi'], img_rel_vectors_gt)

        
        # mesh_dir=osp.join(opts.rendering_dir, opts.name)
        # return visuals

        mesh_dir=osp.join(opts.rendering_dir)
        # vis_codes=[self.codes_pred_vis, self.codes_gt]
        vis_codes=[self.codes_pred_eval, self.codes_gt]
        # vis_layouts = [self.layout_pred, self.layout_gt]
        vis_names=['b_pred', 'c_gt']
        for vx, v_name in enumerate(vis_names):
            os.system('rm {}/*.obj'.format(mesh_dir))
            self.save_codes_mesh(mesh_dir, vis_codes[vx])
            # self.save_layout_mesh(mesh_dir, vis_layouts[vx])

            # visuals['{}_layout_cam_view'.format(v_name)], visuals['{}_layout_novel_view'.format(v_name)] = self.render_visuals(
            #     mesh_dir, obj_name='layout')
            # visuals['{}_objects_cam_view'.format(v_name)], visuals['{}_objects_novel_view'.format(v_name)]=self.render_visuals(
            #     mesh_dir, obj_name='codes')
            # visuals['{}_scene_cam_view'.format(v_name)], visuals['{}_scene_novel_view'.format(v_name)]=self.render_visuals(
            #     mesh_dir)
            visuals['{}_objects_cam_view'.format(v_name)] =self.render_visuals(mesh_dir, obj_name='codes')
            # visuals['{}_scene_cam_view'.format(v_name)] =self.render_visuals(mesh_dir)
        return visuals


    def filter_pos(self, codes, pos_inds):
        pos_inds=torch.from_numpy(np.array(pos_inds)).squeeze()
        t = torch.LongTensor

        if type(codes) == dict:
            key = 'shape'
            if isinstance(codes[key], torch.autograd.Variable):
                if isinstance(codes[key].data, torch.cuda.FloatTensor):
                    t = torch.cuda.LongTensor
            elif isinstance(codes[key], torch.cuda.FloatTensor):
                t = torch.cuda.LongTensor


            pos_inds=torch.autograd.Variable(
                    pos_inds.type(t), requires_grad=False)
            filtered_codes= {k : torch.index_select(code, 0, pos_inds) for k, code in codes.items()}

        else:
            if isinstance(codes[0], torch.autograd.Variable):
                if isinstance(codes[0].data, torch.cuda.FloatTensor):
                    t = torch.cuda.LongTensor
            elif isinstance(codes[0], torch.cuda.FloatTensor):
                t = torch.cuda.LongTensor

            pos_inds =torch.autograd.Variable(
                    pos_inds.type(t), requires_grad=False)
            filtered_codes = [torch.index_select(code, 0, pos_inds) for code in codes]
        return filtered_codes


    def compute_entropy(self, log_probs):
        return np.sum(-1*np.exp(log_probs)*log_probs, axis=-1)

    def update_locations(self, trans_location, relative_locations):
        n_objects=trans_location.size(0)
        # lmbda = min(n_objects*n_objects, 5)
        # lmbda = max(n_objects, 5)
        lmbda=1.0
        relative_locations=relative_locations.numpy()
        trans_location=trans_location.numpy()
        A=np.zeros((n_objects * n_objects, n_objects))
        b=np.zeros((n_objects * n_objects, 3))
        index=0
        for i in range(n_objects):
            for j in range(n_objects):
                if i == j:
                    continue
                # don't add the constraint if it is farther than a particular distance
                dist=np.linalg.norm(relative_locations[i * n_objects + j])
                if dist < 10:
                    A[index][i]=-1
                    A[index][j]=1
                    b[index]=relative_locations[i * n_objects + j]
                    index += 1
        for i in range(n_objects):
            A[index][i]=lmbda * 1
            b[index]=lmbda * trans_location[i]
            index += 1
        A=A[0:index]
        b=b[0:index]
        # pdb.set_trace()
        new_location=np.linalg.lstsq(A, b)
        # source_matrix =  np.cat([np.zeros(n_objects-1, n_objects)
        return torch.from_numpy(new_location[0]), np.linalg.norm(new_location[0] - trans_location, axis=1).tolist()
        # return torch.from_numpy(trans_location)


    def save_predictions_to_pkl(self, dict_of_outputs):
        pkl_file_name = osp.join(self.opts.results_eval_dir, "{}_{}.pkl".format(self.house_names[0], self.view_ids[0]))
        
        def recursive_convert_to_numpy(elem):
            if isinstance(elem, collections.Mapping):
                return {key: recursive_convert_to_numpy(elem[key]) for key in elem}
            elif isinstance(elem, str):
                return elem
            elif isinstance(elem, collections.Sequence):
                return [recursive_convert_to_numpy(samples) for samples in elem]
            elif isinstance(elem, torch.FloatTensor):
                return elem.numpy()
            elif isinstance(elem, torch.cuda.FloatTensor):
                return elem.cpu().numpy()
            elif isinstance(elem, torch.LongTensor):
                return elem.numpy()
            elif isinstance(elem, torch.cuda.LongTensor):
                return elem.cpu().numpy()
            elif isinstance(elem, torch.autograd.Variable):
                return recursive_convert_to_numpy(elem.data)
            else:
                return elem

        new_dict = recursive_convert_to_numpy(dict_of_outputs)
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(new_dict, f)

    def convert_pkl_to_predictions(self, ):
        pkl_file_name = osp.join(self.opts.results_eval_dir, "{}_{}.pkl".format(self.house_names[0], self.view_ids[0]))
        def recursive_convert_to_torch(elem):
            if isinstance(elem, collections.Mapping):
                return {key: recursive_convert_to_torch(elem[key]) for key in elem}
            elif isinstance(elem, str):
                return elem
            elif isinstance(elem, collections.Sequence):
                return [recursive_convert_to_torch(samples) for samples in elem]
            elif isinstance(elem, np.ndarray):
                if elem.dtype == np.int32:
                    torch.from_numpy(elem).long()
                else:
                    return torch.from_numpy(elem).float()
            else:
                return elem
        with open(pkl_file_name, 'rb') as f:
            predictions = pickle.load(f)
        predictions = recursive_convert_to_torch(predictions)
        predictions['gt_codes'] = [Variable(k) for k in predictions['gt_codes']]
        predictions['pred_codes'] = [Variable(k) for k in predictions['pred_codes']]
        predictions['object_class_gt'] = Variable(predictions['object_class_gt']).long()
        predictions['rois'] = Variable(predictions['rois'])
        predictions['amodal_bboxes'] = predictions['amodal_bboxes']
        predictions['codes_gt_quats'] = [Variable(t) for t in predictions['codes_gt_quats']]

        
        try:
            predictions['relative_trans'] = Variable(predictions['relative_trans'])
            predictions['relative_scale'] = Variable(predictions['relative_scale'])
            predictions['relative_dir'] = Variable(predictions['relative_dir'])
            predictions['relative_gt'] = [Variable(k) for k in predictions['relative_gt']]
            predictions['trans_dependent_rotation'] = [k for  k in predictions['trans_dependent_rotation']]
            predictions['trans_dependent_rotation_binned'] = [Variable(k).long() for k in predictions['trans_dependent_rotation_binned']]
            predictions['relative_quat_tensors_angles_gt'] = predictions['relative_quat_tensors_angles_gt']
        
        except KeyError as e:
            assert self.opts.pred_relative == False, 'relative outputs required'
            predictions['relative_trans'] = predictions['relative_scale'] = predictions['relative_quat'] = None
            predictions['relative_dir'] = predictions['relative_gt'] =  predictions['trans_dependent_rotation'] = None
            predictions['trans_dependent_rotation_binned'] = predictions['relative_quat_tensors_angles_gt'] = None

        try:
            predictions['codes_quat_var'] = Variable(predictions['codes_quat_var'])
        except KeyError as e:
            assert self.opts.var_gmm_rot == False, 'var gmm rots given'
            predictions['codes_quat_var'] = None

        return predictions


    def predict(self):
        
        if not self.opts.load_predictions_from_disk:
            feed_dict = {}
            feed_dict['imgs_inp_fine'] = self.input_imgs_fine
            feed_dict['imgs_inp_coarse'] = self.input_imgs
            feed_dict['rois_inp'] = self.rois
            feed_dict['location_inp'] = self.object_locations
            feed_dict['class_inp'] = self.object_classes
            feed_dict['spatial_image'] = self.spatial_image

            model_pred , _ =self.model.forward(feed_dict)

            codes_pred_all = model_pred['codes_pred']
            if self.opts.gmm_rot and self.opts.var_gmm_rot:
                self.codes_quat_var = codes_pred_all['quat'][1].data.cpu().numpy()
                codes_pred_all['quat'] = torch.nn.functional.log_softmax(codes_pred_all['quat'][0])
            else:
                codes_pred_all['quat'] = torch.nn.functional.log_softmax(codes_pred_all['quat'])
            
            stuff_to_save = {'gt_codes' : self.codes_gt,
                             'pred_codes' : codes_pred_all, 
                             'object_class_gt' : self.object_classes,
                             'rois' : self.rois, 
                             'index2object' : self.index2object_class,
                             'amodal_bboxes' : self.amodal_bboxes,
                             'codes_gt_quats' : self.codes_gt_quats,}

            if self.opts.gmm_rot and self.opts.var_gmm_rot:
                stuff_to_save['codes_quat_var'] = self.codes_quat_var


            if self.opts.pred_relative:
                self.relative_predictions = model_pred['codes_relative']
                self.relative_trans_predictions = self.relative_predictions['relative_trans']
                self.relative_scale_predictions = self.relative_predictions['relative_scale']
                self.relative_direction_prediction = self.relative_predictions['relative_dir']

                if self.opts.gmm_dir:
                    self.relative_direction_prediction = self.relative_direction_prediction[0]

                self.relative_direction_prediction = torch.nn.functional.log_softmax(self.relative_direction_prediction)
                stuff_to_save['relative_trans' ] = self.relative_trans_predictions
                stuff_to_save['relative_scale' ] = self.relative_scale_predictions
                stuff_to_save['relative_dir' ] = self.relative_direction_prediction
                stuff_to_save['relative_gt' ] = self.relative_gt
                stuff_to_save['trans_dependent_rotation' ] = self.relative_direction_rotation
                stuff_to_save['trans_dependent_rotation_binned' ] = self.relative_direction_rotation_binned
            if self.opts.pred_class:
                self.class_pred=model_pred['class_pred']
                # self.class_pred=model_pred['class_pred']
            labels_pred=model_pred['labels_pred']
            
            if self.opts.save_predictions_to_disk:
                self.save_predictions_to_pkl(stuff_to_save)
                assert osp.exists(osp.join(self.opts.results_eval_dir, "{}_{}.pkl".format(self.house_names[0], self.view_ids[0]))), 'pkl file does not exist'
        else:

            predictions = self.convert_pkl_to_predictions()
            self.codes_gt = tuple(predictions['gt_codes'])
            codes_pred_all = tuple(predictions['pred_codes'])

            self.relative_trans_predictions = predictions['relative_trans']
            self.relative_scale_predictions = predictions['relative_scale']
            self.relative_quat_predictions =  predictions['relative_quat' ]
            self.relative_direction_prediction = predictions['relative_dir'] 
            self.relative_predictions = [self.relative_trans_predictions, self.relative_scale_predictions, self.relative_quat_predictions, self.relative_direction_prediction]
            self.relative_gt =  predictions['relative_gt']
            self.relative_direction_rotation  = predictions['trans_dependent_rotation']
            self.relative_direction_rotation_binned  = predictions['trans_dependent_rotation_binned']
            self.object_classes  = predictions['object_class_gt']
            self.rois = predictions['rois']
            self.index2object_class = predictions['index2object']
            self.amodal_bboxes = predictions['amodal_bboxes']
            self.relative_quat_tensors_angles_gt = predictions['relative_quat_tensors_angles_gt']
            self.codes_gt_quats = predictions['codes_gt_quats']
            self.codes_quat_var = predictions['codes_quat_var']

        n = codes_pred_all['shape'].size(0)
        labels_pred = Variable(torch.zeros(n, 1).cuda())
        scores_pred = labels_pred.cpu().data.numpy() * 0 + 1
        bboxes_pred = self.rois.data.cpu().numpy()[:, 1:]
        min_score_eval=np.minimum(0.05, np.max(scores_pred))
        # pos_inds_eval = metrics.nms(
        #     np.concatenate((bboxes_pred, scores_pred), axis=1),
        #     0.3, min_score=min_score_eval)
        
        pos_inds_eval=[i for i in range(n)]

        self.codes_pred_eval=self.filter_pos(codes_pred_all, pos_inds_eval)
        # pdb.set_trace()

        opts=self.opts
        # updates for translation
        if opts.pred_relative:
            for i in range(1):
                if not opts.gt_updates:
                    new_trans, self.update_norm=self.update_locations(
                        self.codes_pred_eval['trans'].data.cpu(), self.relative_trans_predictions.data.cpu())
                else:
                    new_trans, self.update_norm=self.update_locations(
                        self.codes_pred_eval['trans'].data.cpu(), self.relative_gt['relative_trans'].data.cpu())
                self.new_trans=new_trans
                if opts.do_updates:
                    self.codes_pred_eval['trans']=Variable(new_trans)
        else:
            self.update_norm=torch.mean(self.codes_pred_eval['trans'] * 0, dim=1).data.cpu().numpy().tolist()

        if opts.pred_relative:
            for i in range(1):
                if not opts.gt_updates:
                    new_scale, _ =self.update_locations(
                        self.codes_pred_eval['scale'].data.cpu().log(), self.relative_predictions['relative_scale'].data.cpu())
                else:
                    new_scale, _ =self.update_locations(
                        self.codes_pred_eval['scale'].data.cpu().log(), self.relative_gt['relative_scale'].data.cpu())
                
                new_scale=new_scale.exp()
                self.new_scale=new_scale
                if opts.do_updates:
                    self.codes_pred_eval['scale']=Variable(new_scale)

        quats_gt_binned = [suncg_parse.quats_to_bininds(q.data.cpu(), self.quat_medoids) for q in self.codes_gt['quat']]
        quats_gt_binned = [Variable(q) for q in quats_gt_binned]
        quats_gt_binned_probs = Variable(self.convert_multiple_bins_to_probabilites(quats_gt_binned, opts.nz_rot, no_noise=1.0)).log()
        
        self.codes_pred_quat_before = Variable(self.codes_pred_eval['quat'].data.clone())
        self.entropy_before_optim = (-1 * self.codes_pred_eval['quat']  * self.codes_pred_eval['quat'].exp()).sum(1).data.cpu().numpy()

        if opts.pred_relative and opts.classify_dir and opts.do_updates:
            if opts.gt_updates:
                relative_direction_prediction = self.convert_multiple_bins_to_probabilites(self.relative_direction_rotation_binned, opts.nz_rel_dir).log().numpy()
                self.relative_direction_prediction = Variable(torch.from_numpy(relative_direction_prediction).cuda())
            else:
                relative_direction_prediction = self.relative_direction_prediction.data.cpu().numpy()
            self.relative_direction_prediction_3d = relative_direction_prediction

            absolute_locations = self.codes_pred_eval['trans'].data.cpu().numpy()
            # absolute_locations = self.codes_gt[3].data.cpu().numpy()
            absolute_log_probabilites = self.codes_pred_eval['quat'].data.cpu().numpy()
            n_objects = len(absolute_log_probabilites)
            n_absoulte_bins = absolute_log_probabilites.shape[1]
            relative_direction_prediction = relative_direction_prediction.reshape(n_objects, n_objects, -1)
            n_relative_bins = relative_direction_prediction.shape[2]
            bin_scores = np.zeros((n_objects, n_objects, n_absoulte_bins))
            quat_medoids = self.quat_medoids.numpy()
            direction_medoids = self.direction_medoids.numpy()
            new_probability = absolute_log_probabilites
            # lambda_weight = opts.lambda_weight 
            # lambda_weight = opts.lambda_weight * 1./np.sqrt(n_objects)
            lambda_weight = opts.lambda_weight * 1./n_objects
            adaptive_weight = np.ones(n_objects)
            # pdb.set_trace()

            for nx in range(n_objects):
                src_c = self.index2object_class[self.object_classes.data[nx, 0]]
                ignore_bin_scores = False
                # if src_c == 'table':
                #     ignore_bin_scores = True
                #     continue
                for mx in range(n_objects):
                    if mx == nx:
                        continue
                    expected_direction = absolute_locations[mx] - absolute_locations[nx] ## make it unit norm
                    dist = (1E-5 + np.linalg.norm(expected_direction))
                    if dist > 4:
                        continue

                    expected_direction = expected_direction/ (1E-5 + np.linalg.norm(expected_direction))
                    expected_direction = expected_direction.reshape(1, -1)
                    alignment_scores = []
                    indices = []
                    entropy = -1*np.sum(np.exp(relative_direction_prediction[nx, mx]) * relative_direction_prediction[nx, mx])
                    # if entropy > 2:
                    #     continue
                    # adaptive_weight[nx] += 1
                    # pdb.set_trace()
                    for  abinx in range(n_absoulte_bins):
                        prob_bin = absolute_log_probabilites[nx][abinx]
                        quaternion_abinx = quat_medoids[abinx]
                        rotation = transformations.quaternion_matrix(quaternion_abinx)
                        transform = rotation.copy()
                        transform[0:3, 3] = np.array(absolute_locations[nx], copy=True)
                        
                        # translation = suncg_parse.trans_transform(absolute_locations[nx])
                        # transform = np.matmul(rotation, translation)
                        
                        relative_direction = direction_medoids
                        predicted_direction = suncg_parse.transform_coordinates(transform, relative_direction) - absolute_locations[nx].reshape(1, -1)
                        # # log_alignment_score = (1 - np.matmul(expected_direction, predicted_direction.transpose()).squeeze()) #* relative_direction_prediction[nx, mx]
                        # alignment_score = np.matmul(expected_direction, predicted_direction.transpose()).squeeze()
                        # alignment_score = (alignment_score > 0.95) * alignment_score
                        # alignment_score = alignment_score * np.exp(relative_direction_prediction[nx, mx])
                        # alignment_score = np.log(np.sum(alignment_score) + 1E-5)
                        # pdb.set_trace()

                        alignment_score = (1 - np.matmul(expected_direction, predicted_direction.transpose()).squeeze())
                        index = np.argmin(alignment_score, axis=0)
                        alignment_score = np.min(alignment_score, axis=0) + relative_direction_prediction[nx, mx, index]# absolute_log_probabilites[nx][abinx]
                        alignment_score = np.min(relative_direction_prediction[nx, mx, index])
                        alignment_scores.append(alignment_score)
                        # indices.append(index)
                        

                    temp = np.array([metrics.quat_dist(quat_medoids[0], quat_medoids[k]) for k in range(0,24)]).round(2)
                    alignment_scores = np.exp(np.array(alignment_scores))
                    alignment_scores = np.log(alignment_scores/np.sum(alignment_scores) + 1E-10)
                    bin_scores[nx,mx,:] = alignment_scores
            bin_scores = np.sum(bin_scores, axis=1)
            bin_scores = np.exp(bin_scores)
            bin_scores = np.log(1E-10 + bin_scores/np.sum(bin_scores, 1, keepdims=True))
            if ignore_bin_scores == True:
                bin_scores = bin_scores * 0
            
            # pdb.set_trace()
            abs_ent = self.compute_entropy(new_probability).reshape(-1, 1)
            rel_ent = self.compute_entropy(bin_scores).reshape(-1, 1)
            # pdb.set_trace()
            new_probability = 1.0 * new_probability  + np.minimum(lambda_weight, 1.0)*(1*bin_scores) + 0.00 * quats_gt_binned_probs.data.cpu().numpy()
            new_probability = torch.from_numpy(new_probability).float()
            new_probability = torch.nn.functional.normalize(new_probability.exp(),1)
            self.codes_pred_eval['quat'] = Variable(new_probability.cuda())
        self.entropy_after_optim = (-1 * self.codes_pred_eval['quat']  * (self.codes_pred_eval['quat'] + 1E-10).log()).sum(1).data.cpu().numpy()




        self.rois_pos_eval=self.filter_pos([self.rois], pos_inds_eval)[0]     # b x 5, 1:5 is box (x1 y1 x2 y2)
        self.codes_pred_eval['shape']=self.decode_shape(self.codes_pred_eval['shape'])    # b x 1 x 32 x 32 x 32

        # if self.opts.gmm_rot:
        #     self.codes_pred_eval[2] = self.decode_rotation_slerp(self.codes_pred_eval[2], 
        # else:
        self.codes_pred_eval['quat']=self.decode_rotation(self.codes_pred_eval['quat'])  # b x 4
        self.codes_pred_quat_before = self.decode_rotation(self.codes_pred_quat_before)
        # self.codes_pred_eval[2]=self.decode_rotation_topk(self.codes_pred_eval[2])  # b x 4
        
        # self.codes_pred_eval[2] = suncg_parse.quats_to_bininds(self.codes_gt[2].data.cpu(), self.quat_medoids)
        # self.codes_pred_eval[2] = Variable(suncg_parse.bininds_to_quats(self.codes_pred_eval[2], self.quat_medoids))
        
        self.codes_pred_eval['scale']  # Probably scale b x 3
        self.codes_pred_eval['trans']  # Probably trans b x 3

        self.scores_pred_eval=scores_pred[pos_inds_eval, :] * 1.
        if opts.pred_class:
            self.class_pred=self.decode_class(self.class_pred)
        # pdb.set_trace()
        min_score_vis=np.minimum(0.7, np.max(scores_pred))
        # pos_inds_vis = metrics.nms(
        #     np.concatenate((bboxes_pred, scores_pred), axis=1),
        #     0.3, min_score=min_score_vis)

        pos_inds_vis=[i for i in range(n)]
        self.codes_pred_vis=self.filter_pos(codes_pred_all, pos_inds_vis)
        self.rois_pos_vis=self.filter_pos([self.rois], pos_inds_vis)[0]
        self.codes_pred_vis['shape']=self.decode_shape(self.codes_pred_vis['shape'])
        # self.codes_pred_vis[2]=self.decode_rotation(self.codes_pred_vis[2])
        self.codes_pred_vis['quat']=self.codes_pred_eval['quat']

        # self.layout_pred = self.layout_model.forward(self.input_imgs_layout)

    def clamp_to_image(self, amodal_bboxes, img_size):
        return torch.stack([torch.clamp(amodal_bboxes[:,0], 0, img_size[1]),
                            torch.clamp(amodal_bboxes[:,1], 0, img_size[0]),
                            torch.clamp(amodal_bboxes[:,2], 0, img_size[1]),
                            torch.clamp(amodal_bboxes[:,3], 0, img_size[0])], dim=1)

    def compute_object_presence_parameters(self, amodal_bboxes, roi_bboxes, img_size):
        ## Compute % visible in the image
        ## % Overlap with other objects not visble
        ammodal_bboxes_clip_to_image = self.clamp_to_image(amodal_bboxes, img_size)
        size_box = amodal_bboxes[:,2:4] - amodal_bboxes[:,0:2]
        area_box = size_box[:,0]*size_box[:,1]

        size_image_box = ammodal_bboxes_clip_to_image[:,2:4] - ammodal_bboxes_clip_to_image[:,0:2]
        area_image_box = size_image_box[:,0]*size_image_box[:,1]

        size_roi_box = roi_bboxes[:,2:4] - roi_bboxes[:,0:2]
        area_roi_box = size_roi_box[:,0]*size_roi_box[:,1]

        return area_image_box/(area_box + 1), area_roi_box/(area_image_box + 1)

    def evaluate(self):
        # rois as numpy array
        # Get Predictions.
        # pdb.set_trace()
        opts = self.opts
        shapes = self.codes_pred_eval['shape'] 
        scales = self.codes_pred_eval['scale']
        rots = self.codes_pred_eval['quat']
        trans = self.codes_pred_eval['trans']
        rots_before = self.codes_pred_quat_before
        trans=trans
        scores=self.scores_pred_eval
        boxes=self.rois_pos_eval.cpu().data.numpy()[:, 1:]
        # Get Ground Truth.
        # pdb.set_trace()
        gt_shapes = self.codes_gt['shape']
        gt_scales = self.codes_gt['scale']
        gt_rots = self.codes_gt['quat']
        gt_trans = self.codes_gt['trans']


        gt_boxes=self.rois.cpu().data.numpy()[:, 1:]
        iou_box=bbox_utils.bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
        trans_, gt_trans_=trans.cpu().data.numpy(), gt_trans.cpu().data.numpy()
        err_trans=np.linalg.norm(np.expand_dims(trans_, 1) - np.expand_dims(gt_trans_, 0), axis=2)
        err_pwd=np.zeros([len(err_trans)])


        err_rel_quat = (0*err_pwd).tolist()
        acc_rel = (0*err_pwd).tolist()
        object_presence, object_visibility  = self.compute_object_presence_parameters(self.amodal_bboxes, self.rois[:, 1:].data.cpu(),
                                                               [opts.img_height_fine, opts.img_width_fine])

        n_objects=len(gt_rots)

        acc_rel_dir_conditions = []
        acc_rel_dir = []
        err_rel_dir = []

        if opts.pred_relative:
            indices=[i + i * n_objects for i in range(n_objects)]

            if self.opts.classify_dir:
                relative_direction_predictions_classes = self.get_class_indices(self.relative_direction_prediction)
                entropy = -1*(self.relative_direction_prediction * self.relative_direction_prediction.exp()).sum(1).data.cpu().numpy()
                relative_direction_prediction = suncg_parse.bininds_to_directions(relative_direction_predictions_classes, self.direction_medoids)
            else:
                relative_direction_prediction = self.relative_direction_prediction.data.cpu()

            # pdb.set_trace()
            # pdb.set_trace()
            for i,  (pred_dir, gt_dirs) in enumerate(zip(relative_direction_prediction, self.relative_direction_rotation)):
            # for i,  (pred_dir, gt_dirs) in enumerate(zip(self.relative_direction_rotation, self.relative_direction_rotation)):
                if i in indices:
                    continue

                src_i = i //n_objects
                src_c = self.index2object_class[self.object_classes.data[src_i, 0]]
                # if src_c != 'desk':
                #     continue

                min_err  = 180
                for gt_dir in gt_dirs:
                    min_err = min(min_err, metrics.direction_dist(pred_dir.numpy(), gt_dir.numpy()))
                err_rel_dir.append(min_err)
            state = -1
            err_angle_iter = iter(err_rel_dir)
            if opts.classify_dir:
                for i, (pred, gt_bins, pred_dir, gt_dirs) in enumerate(zip(relative_direction_predictions_classes, self.relative_direction_rotation_binned,
                 relative_direction_prediction, self.relative_direction_rotation)):
                    if i in indices:
                        continue
                    if pred in gt_bins.data.cpu():
                        acc_rel_dir.append(1)
                        state = 1
                    else:
                        acc_rel_dir.append(0)
                        state = 0
                    src_i = i//n_objects
                    trj_i = i % n_objects
                    src_c = self.index2object_class[self.object_classes.data[src_i, 0]]
                    trj_c = self.index2object_class[self.object_classes.data[trj_i, 0]]
                    t = ("{}_{}".format(self.house_names[0], self.view_ids[0]), src_i, trj_i, pred, gt_bins.data.cpu().numpy(),
                         np.linalg.norm(gt_trans_[src_i]- gt_trans_[trj_i]), state, 
                         next(err_angle_iter),
                         src_c, trj_c,
                         object_visibility[src_i], object_visibility[trj_i],
                         object_presence[src_i], object_presence[trj_i], entropy[i])
                    acc_rel_dir_conditions.append(t)
            else:
                acc_rel_dir.append(0)


        scales_, gt_scales_=scales.cpu().data.numpy(), gt_scales.cpu().data.numpy()
        err_scales=np.mean(np.abs(np.expand_dims(np.log(scales_), 1) - np.expand_dims(np.log(gt_scales_), 0)), axis=2)
        err_scales /= np.log(2.0)

        gt_quats =  [t.data.cpu() for t in self.codes_gt_quats]


        ndt, ngt=err_scales.shape
        err_shapes=err_scales * 0.
        err_rots=err_scales * 0.
        err_rots_before = err_scales * 0
        # pdb.set_trace()
        for i in range(ndt):
          for j in range(ngt):
            err_shapes[i, j]=metrics.volume_iou(shapes[i, 0].data, gt_shapes[
                                                j, 0].data, thresh=self.opts.voxel_eval_thresh)
            if len(rots[i]) == 4:
                # err_rots[i, j]=metrics.quat_dist(rots[i].data.cpu(), gt_rots[j].data.cpu())
                q_errs = []
                for quat in gt_quats[j]:
                    q_errs.append(metrics.quat_dist(rots[i].data.cpu(), quat))
                err_rots[i, j] = min(q_errs)
            else:
                m1 = metrics.quat_dist(rots[i][0].data.cpu(), gt_rots[j].data.cpu())
                m2 = metrics.quat_dist(rots[i][1].data.cpu(), gt_rots[j].data.cpu())
                err_rots[i, j] = min(m1, m2)

        for i in range(ndt):
          for j in range(ngt):
            err_shapes[i, j]=metrics.volume_iou(shapes[i, 0].data, gt_shapes[
                                                j, 0].data, thresh=self.opts.voxel_eval_thresh)
            if len(rots_before[i]) == 4:
                # err_rots[i, j]=metrics.quat_dist(rots[i].data.cpu(), gt_rots[j].data.cpu())
                q_errs = []
                for quat in gt_quats[j]:
                    q_errs.append(metrics.quat_dist(rots_before[i].data.cpu(), quat))
                err_rots_before[i, j] = min(q_errs)
            else:
                m1 = metrics.quat_dist(rots_before[i][0].data.cpu(), gt_rots[j].data.cpu())
                m2 = metrics.quat_dist(rots_before[i][1].data.cpu(), gt_rots[j].data.cpu())
                err_rots_before[i, j] = min(m1, m2)

        err_rots=np.diag(err_rots).tolist()
        acc_rots = [1 if err < 30 else 0 for err in err_rots]
        err_rots_before = np.diag(err_rots_before).tolist()
        acc_rots_before = [1 if err < 30 else 0 for err in err_rots_before]
        err_trans=np.diag(err_trans).tolist()
        err_scales=np.diag(err_scales).tolist()
        err_pwd=err_pwd.tolist()
        err_shapes = np.diag(err_shapes).tolist()

        house_name_view_id = "{}_{}".format(self.house_names[0], self.view_ids[0])
        absolute_rot_conditions = []
        # pdb.set_trace()
        if self.opts.var_gmm_rot:
            self.codes_quat_var = np.sqrt(np.exp(self.codes_quat_var))*180/np.pi
        else:
            self.codes_quat_var = np.zeros([len(err_rots), self.opts.nz_rot])
        for ox in range(len(err_rots)):
            bf_class = suncg_parse.quats_to_bininds(rots_before[ox].data.unsqueeze(0), self.quat_medoids)[0]
            af_class = suncg_parse.quats_to_bininds(rots[ox].data.unsqueeze(0), self.quat_medoids)[0]
            t = (house_name_view_id, ox,  self.index2object_class[self.object_classes.data[ox, 0]], err_rots_before[ox], err_rots[ox],
                 self.entropy_before_optim[ox], self.entropy_after_optim[ox], bf_class, af_class,
                 metrics.quat_dist(rots_before[ox].data, rots[ox].data), self.codes_quat_var[ox][bf_class])
            # pdb.set_trace()
            absolute_rot_conditions.append(t)


        # absolute_rot_conditions = [(err_b, err, entp_be, entp_af, metrics.quat_dist(rot_b.data, rot_af.data)) for err_b, err, entp_be, entp_af, rot_b, rot_af in zip(err_rots_before,
        #  err_rots, self.entropy_before_optim, self.entropy_after_optim, rots_before, rots)]

        # for i in range(len(err_rots_before)):
        #     if self.entropy_before_optim[i] < self.entropy_after_optim[i] - 1.0:
        #         # pdb.set_trace()
        #         err_rots[i] = err_rots_before[i]



        stats={'trans': err_trans, 'scales': err_scales,'shape': err_shapes, 'rot': err_rots, 'rot_b' : err_rots_before, 'acc_rots' : acc_rots, 'acc_rots_bef' : acc_rots_before,
        'pwd': err_pwd, 'trans_updates': self.update_norm, 'acc_rot' : acc_rots, 'acc_rot_before' : acc_rots_before,
         # 'pwr': err_rel_quat, 'acc_rel_quat' : acc_rel_quat, 
         'acc_rel_dir' : acc_rel_dir, 'rel_dir': err_rel_dir
         }
        # print(stats)
        # pdb.set_trace()

        if opts.pred_class:
            correct=torch.sum(self.class_pred == self.object_classes.squeeze(1).data.cpu())
            total=len(self.class_pred)
            stats['correct']=correct
            stats['total']=total
        else:
            stats['correct']=0
            stats['total']=0
        
        if len(err_trans) == 1:
            stats = {}
        return stats, acc_rel_dir_conditions, absolute_rot_conditions

    def save_current_visuals(self, house_name, view_id):
        imgs_dir=osp.join(self.opts.results_quality_dir, '{}_{}'.format(house_name, view_id))
        img_file = osp.join(imgs_dir, 'c_gt_objects_cam_view.png')
        if osp.exists(imgs_dir) and osp.exists(img_file):
            return
        else:
            visuals=self.get_current_visuals()
            if not os.path.exists(imgs_dir) :
                os.makedirs(imgs_dir)
            for k in visuals:
                img_path=osp.join(imgs_dir, k + '.png')
                scipy.misc.imsave(img_path, visuals[k])

    def save_current_stats(self, bench, house_name, view_id):
        imgs_dir=osp.join(self.opts.results_quality_dir, '{}_{}'.format(house_name, view_id))
        json_file=os.path.join(imgs_dir, 'bench_iter_{}.json'.format(0))
        # print(json_file)
        # if house_name == 'd49bb0b4b52cceffbe6086dfa1976e51':
        #     pdb.set_trace()
        with open(json_file, 'w') as f:
            json.dump({'bench': bench}, f)

    def test_draw(self):
        base_dir=osp.join(self.opts.results_quality_dir)
        opts=self.opts
        house_name_view_ids=[]
        index_filename=opts.index_file
        if index_filename is not None:
            with open(index_filename) as f:
                for line in f:
                    line=line.strip()
                    house_name_view_ids.append('_'.join(line.split('_')))

        rng = np.random.RandomState(0)
        indices = rng.choice(len(self.dataloader), 200)
        if len(house_name_view_ids) == 0:
            dump_to_file = True
            dump_file_fd = open(osp.join(base_dir,'random_house_views.txt'),'w') 

        # pdb.set_trace()
        # read the files for which you want to create visualizations?
        for i, batch in enumerate(self.dataloader):
            self.set_input(batch)
            self.vis_iter=i
            # print(i)
            if self.invalid_batch:
                continue
            house_name=batch['house_name'][0]
            view_id=batch['view_id'][0]
            example_id='{}_{}'.format(house_name, view_id)
            if example_id in house_name_view_ids or (i in indices and len(house_name_view_ids) == 0):
                self.predict()
                bench_image_stats,_,_, = self.evaluate()
                self.save_current_visuals(house_name, view_id)
                self.save_current_stats(bench_image_stats, house_name, view_id)
                print("Generating {}".format(i))
                if dump_to_file:
                    dump_file_fd.write("{}_{}\n".format(house_name, view_id))
            # if i > 10 and len(house_name_view_ids) == 0:
            #     break
        if dump_to_file:
            dump_file_fd.close()


    def test(self):
        # Choose 30 random examples and save it.
        opts=self.opts
        if not opts.preload_stats:
            invalid_rois=0
            bench_stats=[]
            acc_rel_conditions_all = []
            head = ['house_name_view_id', 'src_index', 'trj_index', 'PC', 'GT', 'D', 'Acc', 'Err', 'SC', 'TC', 'VS', 'VT', 'PS', 'PT', 'Entropy', 'Var']
            acc_rel_conditions_all.append(head)
            acc_rel_dir_conditions_all = [head]
            absolute_rot_conditions_all = [['house_name_view_id', 'obj_index', 'obj_class', 'err_before', 'err', 'entropy_before', 'entropy_after', 'rot_b', 'rot_af', 'diff_bw_bf_af']]

            # codes are (shapes, scales, quats, trans)
            n_iter=len(self.dataloader)
            for i, batch in enumerate(self.dataloader):
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                self.set_input(batch)
                if not self.invalid_batch:
                    self.predict()
                    house_name = batch['house_name'][0]
                    view_id = batch['view_id'][0]
                    bench_image_stats, acc_rel_dir_conditions, absolute_rot_conditions =self.evaluate()
                    acc_rel_dir_conditions_all.extend(acc_rel_dir_conditions)
                    absolute_rot_conditions_all.extend(absolute_rot_conditions)
                    json_file=osp.join(opts.results_eval_dir, 'eval_result_{}_{}.json'.format(house_name, view_id))
                    bench_image_stats['house_name']=batch['house_name'][0]
                    bench_image_stats['view_id']=batch['view_id'][0]
                    # pdb.set_trace()
                    # with open(json_file, 'w') as f:
                    #     json.dump({'bench': bench_image_stats}, f)

                    bench_stats.append(bench_image_stats)

                    # if opts.save_visuals and (i % opts.visuals_freq == 0):
                    #     self.save_current_visuals()
                else:
                    if self.invalid_rois is not None:
                        print("Total rois {}".format(self.invalid_rois.numel() / 5))
                    invalid_rois += 1
                # if i > 10:
                #     break
                # break

            print("% of RoI invalid {}".format(invalid_rois * 100.0 / n_iter))


            # Accumalate stats and print
            acc_stats={'trans': [], 'scales': [], 'shape' : [], 'rot_b': [], 'rot': [],  'trans_updates': [],
             # 'pwr': [], 'acc_rel_quat' : [], 
             'acc_rel_dir' : [], 'rel_dir' : [], 'acc_rot' : [], 'acc_rot_before' : []}
            class_stats={'correct': [], 'total': []}
            for bench in bench_stats:
                for key in acc_stats.keys():
                    if key in bench:
                        acc_stats[key].extend(bench[key])
                for key in class_stats.keys():
                    if key in bench:
                        class_stats[key].append(bench[key])

            # acc_threshold = {'shape' : 0.25 , 'trans' : 1, 'rot_b' : 30, 'rot' : 30, 'scales':0.5}
            acc_threshold = {'shape' : 0.25 , 'trans' : 0.5, 'rot_b' : 30, 'rot' : 30, 'scales':0.2}
            for key, thres in acc_threshold.items():
                acc_stats["{}_acc".format(key)] = [1 if v < thres  else 0 for v in acc_stats[key]]

            # pdb.set_trace()
            json_file=os.path.join(FLAGS.results_eval_dir, 'eval_set_{}_{}_{}.json'.format(opts.id, opts.eval_set, 0))

            print('Writing results to file: {:s}'.format(json_file))
            with open(json_file, 'w') as f:
                json.dump(acc_stats, f)
        else:
            json_file=os.path.join(FLAGS.results_eval_dir, 'eval_set_{}_{}_{}.json'.format(opts.id, opts.eval_set, 0))
            with open(json_file) as f:
                acc_stats=json.load(f)

         # Print mean error and median error
        metrics={'mean': np.mean, 'median': np.median}
        criterias={'trans', 'scales', 'rot','rot_b', 'trans_updates', 'shape',
         # 'pwr', 'acc_rel_quat',
         'acc_rel_dir', 'rel_dir', 'acc_rot', 'acc_rot_before',
         'trans_acc', 'rot_b_acc', 'rot_acc', 'scales_acc', 'shape_acc'}

        for key in criterias:
            for mkey in metrics.keys():
                print('{} {} : {:0.3f}'.format(mkey, key, metrics[mkey](np.array(acc_stats[key]))))

        for key in acc_stats.keys():
            acc_stats[key]=np.array(acc_stats[key])

        # keys=['trans', 'scales', 'rot', 'pwd', 'trans_updates', 'pwr']
        key_clip={'shape' : 1.0, 'trans': 3.0, 'pwd': 5.0, 'scales': 1.5, 'rot_b': 180, 'rot': 180,'trans_updates': 4, 'pwr': 180 , 'rel_dir': 180}
        for key in criterias:
            err=acc_stats[key]
            if 'acc' in key:
                clip_max = 2
                continue
            else:
                clip_max=key_clip[key]
            values, base=np.histogram(np.clip(np.array(err), 0, clip_max), 40)
            cumulative=np.cumsum(values)
            cumulative=cumulative / len(err)
            plt.plot(cumulative, base[:-1], c='blue')
            plt.plot([0.0, 1.0], [np.mean(err), np.mean(err)], c='red')
            plt.title('Error {} vs data-fraction'.format(key))
            plt.savefig(os.path.join(FLAGS.results_eval_dir, 'eval_set_{}_{}_{}.png'.format(opts.id, opts.eval_set, key)))
            plt.close()

            with open(os.path.join(FLAGS.results_eval_dir, 'eval_set_{}_{}_{}.pkl'.format(opts.id, opts.eval_set, key)) , 'wb') as f:
                pickle.dump({'err' : acc_stats[key], 'freq_values' : cumulative, 'bin_values': base[:-1]}, f)


        if self.opts.pred_class:
            correct=sum(class_stats['correct'])
            total=sum(class_stats['total'])
            print('{}, {}/{}  {}'.format('class accuracy', correct, total, correct * 1.0 / total))


        if opts.log_csv is not None:
            with open("{}_rel_dir.csv".format(opts.log_csv), 'w') as f:
                for acc_cond in acc_rel_dir_conditions_all:
                    for val in list(acc_cond):
                        if type(val) == np.ndarray:
                            t = list(set([str(x) for x in val]))
                            t.sort()
                            f.write('{},'.format(';'.join(t)))
                        else:
                            if type(val) == str:
                                f.write('{},'.format(val))
                            else:
                                f.write('{},'.format(str(np.round(val, 3))))
                    f.write('\n')

        if opts.log_csv is not None:
            with open("{}_abs_rot.csv".format(opts.log_csv), 'w') as f:
                for entropy_cond in absolute_rot_conditions_all:
                    for val in list(entropy_cond):
                        if type(val) == np.ndarray:
                            t = list(set([str(x) for x in val]))
                            t.sort()
                            f.write('{},'.format(';'.join(t)))
                        else:
                            if type(val) == str:
                                f.write('{},'.format(val))
                            else:
                                f.write('{},'.format(str(np.round(val, 3))))
                    f.write('\n')


def main(_):
    FLAGS.suncg_dl_out_codes=True
    FLAGS.suncg_dl_out_fine_img=True
    FLAGS.suncg_dl_out_test_proposals=True
    FLAGS.suncg_dl_out_voxels=False
    FLAGS.suncg_dl_out_layout=True
    FLAGS.suncg_dl_out_depth=False
    # FLAGS.n_data_workers=4
    FLAGS.max_views_per_house=2
    

    FLAGS.batch_size=1
    assert(FLAGS.batch_size == 1)

    if FLAGS.results_name is None:
        FLAGS.results_name=FLAGS.name

    FLAGS.results_vis_dir=osp.join(FLAGS.results_vis_dir, 'box3d_base', FLAGS.eval_set, FLAGS.results_name)
    FLAGS.results_quality_dir=osp.join(FLAGS.results_quality_dir, 'box3d_base', FLAGS.eval_set, FLAGS.results_name)
    FLAGS.results_eval_dir=osp.join(FLAGS.results_eval_dir, 'box3d_base', FLAGS.eval_set, FLAGS.results_name)
    FLAGS.rendering_dir = osp.join(FLAGS.rendering_dir, FLAGS.results_name)
    if not os.path.exists(FLAGS.results_eval_dir):
        os.makedirs(FLAGS.results_eval_dir)
    if not os.path.exists(FLAGS.results_vis_dir):
        os.makedirs(FLAGS.results_vis_dir)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if not FLAGS.classify_rot:
        FLAGS.nz_rot=4


    if not FLAGS.classify_dir:
        FLAGS.nz_rel_dir=3

    tester=DWRTester(FLAGS)
    tester.init_testing()
    if not FLAGS.draw_vis:
        tester.test()
    else:
        tester.test_draw()

    # pred_clases = torch.cat(tester.stored_quat_relative_pred_classes).numpy()
    # gt_clases = torch.cat(tester.stored_quat_relative_gt_classes).numpy()

    # with open(osp.join(FLAGS.results_eval_dir, 'pred_relative_classes.npy'),'w') as f:
    #     np.save(f, pred_clases)
    # with open(osp.join(FLAGS.results_eval_dir, 'gt_relative_classes.npy'),'w') as f:
    #     np.save(f, gt_clases)
    # # pdb.set_trace()


if __name__ == '__main__':
    app.run(main)
