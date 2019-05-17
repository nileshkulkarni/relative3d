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
import matplotlib.pyplot as plt
import time

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
from ...utils import transformations


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'), 'Directory where intermittent renderings are saved')

flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')
flags.DEFINE_float('voxel_eval_thresh', 0.25, 'Voxel evaluation threshold')
flags.DEFINE_float('relative_update_threshold', 0.3, 'Voxel evaluation threshold')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')

flags.DEFINE_integer('max_rois', 100, 'If we have more objects than this per image, we will subsample.')
flags.DEFINE_integer('max_total_rois', 100, 'If we have more objects than this per batch, we will reject the batch.')

flags.DEFINE_string('results_name', None, 'results_name')
flags.DEFINE_string('layout_name', 'layout_pred', 'Experiment name for layout predictor')
flags.DEFINE_integer('layout_train_epoch', 8, 'Experiment name for layout predictor')
flags.DEFINE_boolean('do_updates', True, 'Do opt updates')
flags.DEFINE_boolean('draw_vis', False, 'Do opt updates')
flags.DEFINE_float('split_size', 1.0, 'Split size of the train set')
flags.DEFINE_float('lambda_weight', 5.0, 'Lambda used for rotation')
flags.DEFINE_boolean('box3d_model', False, 'Load pretrained box3d model, use only when you are using the detector trained on GT boxes')
flags.DEFINE_boolean('pos_label_model', False, 'Load pos label model, use only when you are using the detector trained on GT boxes')
flags.DEFINE_boolean('update_proposal_scores', False, 'Update scores for 2d proposals')
flags.DEFINE_boolean('pretrained_shape_decoder', True, 'Load pretrained shape decoder model, use only when you are using the detector trained on GT boxes')
flags.DEFINE_string('index_file', None, 'file containing house names and view ids')
flags.DEFINE_boolean('pretrained_detections', True, 'Load pretrained detection model to score proposals')

FLAGS = flags.FLAGS

EP_box_iou_thresh =     [0.5   , 0.5      , 0.5    , 0.5      , 0.       , 0.5      , 0.5     , 0.5         , 0.5           , 0.5           , 0.5           , 0.5               , ]
EP_rot_delta_thresh =   [30.   , 30.      , 400.   , 30.      , 30.      , 30.      , 400.    , 30.         , 400.          , 400.          , 400.          , 30                , ]
EP_trans_delta_thresh = [1.    , 1.       , 1.     , 1000.    , 1        , 1.       , 1000.   , 1000.       , 1.            , 1000.         , 1000.         , 1000.             , ]
EP_trans_delta_thresh = [x*0.5 for x in EP_trans_delta_thresh]
EP_shape_iou_thresh =   [0.25  , 0        , 0.25   , 0.25     , 0.25     , 0.25     , 0       , 0           , 0             , 0.25          , 0             , 0.25              , ]
EP_scale_delta_thresh = [0.2   , 0.2      , 0.2    , 0.2      , 0.2      , 100.     , 100.    , 100         , 100           , 100           , 0.2           , 100               , ]
# EP_scale_delta_thresh = [0.5   , 0.5      , 0.5    , 0.5      , 0.5      , 100.     , 100.    , 100         , 100           , 100           , 0.5           , 100               , ]
EP_ap_str =             ['all' , '-shape' , '-rot' , '-trans' , '-box2d' , '-scale' , 'box2d' , 'box2d+rot' , 'box2d+trans' , 'box2d+shape' , 'box2d+scale' , 'box2d+rot+shape' , ]

class DWRTester(test_utils.Tester):


    def preload_detection_pretrained_model(self):
        opts = self.opts
        detection_model = oc_net.OCNet(
            (opts.img_height, opts.img_width), opts=opts,
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=False, nz_shape=opts.nz_shape, pred_labels=True,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot,
            cython_roi= True, use_basic=True)
        detection_model.add_label_predictor()
        detection_model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))
        network_dir = osp.join(opts.cache_dir, 'snapshots', 'pretrained_dwr_shape_ft')
        self.load_network(detection_model, 'pred', 1, network_dir = network_dir)
        detection_model.eval()
        detection_model.cuda()
        self.detection_model = detection_model
        return


    def define_model(self):
        '''
        Define the pytorch net 'model' whose weights will be updated during training.
        '''
        opts = self.opts

        self.voxel_encoder, nc_enc_voxel = net_blocks.encoder3d(
            opts.n_voxel_layers, nc_max=opts.voxel_nc_max, nc_l1=opts.voxel_nc_l1, nz_shape=opts.nz_shape)

        self.voxel_decoder = net_blocks.decoder3d(
            opts.n_voxel_layers, opts.nz_shape, nc_enc_voxel, nc_min=opts.voxel_nc_l1)
        
        self.model = oc_net.OCNet(
            (opts.img_height, opts.img_width), opts=self.opts,
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=False, nz_shape=opts.nz_shape, pred_labels=True,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot, cython_roi=opts.cython_roi)
        
        if opts.box3d_model:
            self.load_network(self.model, 'pred', self.opts.num_train_epoch)

        self.model.add_label_predictor()

        if opts.pos_label_model:
            self.load_network(self.model, 'pred', self.opts.num_train_epoch)

        if opts.pred_voxels:
            self.model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))
        
        # if not (opts.box3d_model or opts.pos_label_model) :
        #     self.load_network(self.model, 'pred', self.opts.num_train_epoch)

        print('Loading detection model')
        self.preload_detection_pretrained_model()

     
        if opts.pretrained_shape_decoder:
            self.model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            print('Loading shape decoder pretrained')
            self.load_network(
                self.model.code_predictor.shape_predictor.decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)

        if opts.pred_voxels:
             self.voxel_decoder = copy.deepcopy(self.model.code_predictor.shape_predictor.decoder)
             self.voxel_decoder.cuda()
             self.voxel_decoder.eval()

        self.layout_model = disp_net.dispnet()
        network_dir = osp.join(opts.cache_dir, 'snapshots', opts.layout_name)
        self.load_network(
            self.layout_model, 'pred', opts.layout_train_epoch, network_dir=network_dir)
        self.layout_model.eval()
        self.layout_model = self.layout_model.cuda(device=self.opts.gpu_id)
        if not opts.pretrained_detections:
            print('Using the current model to evaluate detections')
            self.detection_model = self.model

        self.model.eval()
        self.model = self.model.cuda(device=self.opts.gpu_id)

        return

    def init_dataset(self):
        opts = self.opts
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        if self.opts.split_size < 1.0:
            rng = np.random.RandomState(10)
            rng.shuffle(self.split[opts.eval_set])
            len_splitset = int(len(self.split[opts.eval_set])*opts.split_size)
            self.split[opts.eval_set] = self.split[opts.eval_set][0:len_splitset]
        
        self.dataloader = suncg_data.suncg_data_loader_benchmark(self.split[opts.eval_set], opts)

        if opts.voxel_size < 64:
            self.downsample_voxels = True
            self.downsampler = render_utils.Downsample(
                64//opts.voxel_size, use_max=True, batch_mode=True
            ).cuda(device=self.opts.gpu_id)
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
            self.voxel_decoder = self.voxel_decoder.cuda(device=self.opts.gpu_id)

        self.spatial_image = Variable(suncg_data.define_spatial_image(opts.img_height_fine, opts.img_width_fine, 1.0/16).unsqueeze(0).cuda()) ## (1, 2, 30, 40)
        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)
            if opts.nz_rot == 48:
                self.quat_medoids = torch.from_numpy(
                    scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids_48.mat'))['medoids']).type(torch.FloatTensor)

            # self.quat_medoids_relative = torch.from_numpy(
            #     scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids_relative.mat'))['medoids']).type(torch.FloatTensor)
            self.quat_medoids_var = None

        self.direction_medoids = torch.from_numpy(
            scipy.io.loadmat(osp.join(opts.cache_dir, 'direction_medoids_relative_{}_new.mat'.format(opts.nz_rel_dir)))['medoids']).type(torch.FloatTensor)
        self.direction_medoids = torch.nn.functional.normalize(self.direction_medoids)


        return

    def decode_shape(self, pred_shape):
        opts = self.opts
        pred_shape = torch.nn.functional.sigmoid(
            self.voxel_decoder.forward(pred_shape)
        )
        return pred_shape

    def decode_rotation(self, pred_rot):
        opts = self.opts
        if opts.classify_rot:
            _, bin_inds = torch.max(pred_rot.data.cpu(), 1)
            pred_rot = Variable(suncg_parse.bininds_to_quats(
                bin_inds, self.quat_medoids), requires_grad=False)
        return pred_rot

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
        rois = bboxes_proposals
        if rois.numel() <= 0 or bboxes_gt.numel() <= 0: #some proposals and gt objects should be there
            self.invalid_batch = True
            self.invalid_rois = None
            return
        else:
            self.invalid_batch = False

        self.house_names = batch['house_name']
        self.view_ids = batch['view_id']
        # Inputs for prediction
        input_imgs_fine = batch['img_fine'].type(torch.FloatTensor)
        input_imgs = batch['img'].type(torch.FloatTensor)

        self.input_imgs_layout = Variable(
            input_imgs.cuda(device=opts.gpu_id), requires_grad=False)

        for b in range(input_imgs_fine.size(0)):
            input_imgs_fine[b] = self.resnet_transform(input_imgs_fine[b])
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        self.input_imgs = Variable(
            input_imgs.cuda(device=opts.gpu_id), requires_grad=False)

        self.input_imgs_fine = Variable(
            input_imgs_fine.cuda(device=opts.gpu_id), requires_grad=False)

        self.rois = Variable(
            rois.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)

        # Useful for evaluation
        
        self.layout_gt = Variable(
            batch['layout'].cuda(device=opts.gpu_id), requires_grad=False)

        code_tensors = suncg_parse.collate_codes(batch['codes'])
        code_tensors['shape'] = code_tensors['shape'].unsqueeze(1) #unsqueeze voxels
        code_tensors_quats = code_tensors['quat']


        self.codes_gt_quats = [
            Variable(t.cuda(), requires_grad=False) for t in code_tensors_quats]
        codes_gt_keys = ['shape', 'scale', 'trans']
        self.codes_gt  ={key : Variable(code_tensors[key].cuda(), requires_grad=False) 
                            for key in codes_gt_keys}
        self.codes_gt['quat'] = self.codes_gt_quats



        self.rois_gt = Variable(
            bboxes_gt.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)
        
        if self.downsample_voxels:
            self.codes_gt['shape'] = self.downsampler.forward(self.codes_gt['shape'])

        object_classes = code_tensors['class'].type(torch.LongTensor)
        self.class_gt = self.object_classes = Variable(object_classes.cuda(), requires_grad=False)
       
        self.object_locations = suncg_parse.batchify(code_tensors['trans'], self.rois_gt[:, 0].data.cpu())
        code_tensors['shape'] = code_tensors['shape'].unsqueeze(1)  # unsqueeze voxels
       
        vox2cams = code_tensors['transform_vox2cam']
        self.vox2cams = vox2cams =  suncg_parse.batchify(vox2cams, self.rois_gt[:,0].data.cpu())
       
        cam2voxs = code_tensors['transform_cam2vox']
        cam2voxs = suncg_parse.batchify(cam2voxs, self.rois_gt[:,0].data.cpu())

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

        if opts.pred_relative:
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

        return



    def save_layout_mesh(self, mesh_dir, layout, prefix='layout'):
        opts = self.opts
        layout_vis = layout.data[0].cpu().numpy().transpose((1,2,0))
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        vs, fs = render_utils.dispmap_to_mesh(
            layout_vis,
            suncg_parse.cam_intrinsic(),
            scale_x=self.opts.layout_width/640,
            scale_y=self.opts.layout_height/480
        )
        fout = open(mesh_file, 'w')
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        fout = open(mesh_file, 'w')
        render_utils.append_obj(fout, vs, fs)
        fout.close()

    # def save_codes_mesh(self, mesh_dir, code_vars, prefix='codes'):
    #     opts = self.opts

    #     n_rois = code_vars[0].size()[0]
    #     code_list = suncg_parse.uncollate_codes(code_vars, self.input_imgs.data.size(0), torch.Tensor(n_rois).fill_(0))

    #     if not os.path.exists(mesh_dir):
    #         os.makedirs(mesh_dir)
    #     mesh_file = osp.join(mesh_dir, prefix + '.obj')
    #     render_utils.save_parse(mesh_file, code_list[0], save_objectwise=False, thresh=0.1)

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
        png_dir = osp.join(mesh_dir, 'rendering')
        if obj_name is not None:
            render_utils.render_mesh(osp.join(mesh_dir, obj_name + '.obj'), png_dir)
            im_view1 = scipy.misc.imread(osp.join(png_dir, '{}_render_000.png'.format(obj_name)))
            im_view2 = scipy.misc.imread(osp.join(png_dir, '{}_render_003.png'.format(obj_name)))
        else:
            render_utils.render_directory(mesh_dir, png_dir)
            im_view1 = scipy.misc.imread(osp.join(png_dir, 'render_000.png'))
            im_view2 = scipy.misc.imread(osp.join(png_dir, 'render_003.png'))

        return im_view1

    def filter_pos(self, codes, pos_inds):
        pos_inds=torch.from_numpy(np.array(pos_inds)).squeeze()
        t = torch.LongTensor

        if type(codes) == dict:
            key = codes.keys()[0]
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



    def optimize_unaries(self, unary_, relative_, valid_indices, do_updates, lambda_weight=1.0, dist_threshold = 10.0):
        '''
        do_updates : Booleam
        unary_ : np array N x 3
        relative_  : np array N*N x 2
        valid_indices  : list 
        lambda_weight : 1
        '''
        lambda_weight = 1.0
        new_unary = unary_
        n_objects = len(unary_)
        relative_ = relative_.reshape(n_objects, n_objects, 3)


        if do_updates:
            new_unary = [[] for _ in range(n_objects)]
            A=np.zeros((n_objects * (1 + n_objects), n_objects))
            b=np.zeros((n_objects * (1 + n_objects), 3))
            index=0
            for i in range(n_objects):
                index = 0
                A=np.zeros((n_objects * (1 + n_objects), n_objects))
                b=np.zeros((n_objects * (1 + n_objects), 3))
                ctr = 0
                for j in range(n_objects):
                    if i == j:
                        continue
                    if j not in valid_indices:
                        continue
                    rel_dist = np.linalg.norm(relative_[i, j , :])
                    unary_dist = np.linalg.norm(unary_[i] - unary_[j])
                    
                    if rel_dist > 10 or unary_dist > 10:
                        continue

                    A[index][i] = -1
                    A[index][j] =  1
                    b[index] = relative_[i][j]
                    index += 1
                    
                    A[index][i]  = 1
                    A[index][j] = -1
                    b[index]  = relative_[j][i]
                    index += 1

                    A[index][j] = lambda_weight
                    b[index]  = lambda_weight * unary_[j]
                    index +=1
                    ctr += 2
                A[index][i] = lambda_weight  * 1
                b[index]  = lambda_weight  * unary_[i]
                index += 1
                A=A[0:index]
                b=b[0:index]
                soln = np.linalg.lstsq(A, b)[0]
                new_unary[i] = soln[i]
            new_unary = np.stack(new_unary)

        return new_unary


    def optimize_unaries_strat2(self, unary_, relative_, valid_indices, do_updates, lambda_weight=1.0, dist_threshold = 10.0):
        '''
        do_updates : Booleam
        unary_ : np array N x 3
        relative_  : np array N*N x 2
        valid_indices  : list 
        lambda_weight : 1
        '''
        lambda_weight = 1.0
        new_unary = unary_
        n_objects = len(unary_)
        relative_ = relative_.reshape(n_objects, n_objects, 3)

        if do_updates:
            baseA=np.zeros((n_objects * (n_objects + 1), n_objects))
            baseb=np.zeros((n_objects * (n_objects + 1), 3))
            index=0
            for i in range(n_objects):
                if i not in valid_indices:
                    continue
                for j in range(n_objects):
                    if i == j :
                        continue
                    if j not in valid_indices:
                        continue
                    rel_dist = np.linalg.norm(relative_[i, j , :])
                    unary_dist = np.linalg.norm(unary_[i] - unary_[j])
                    
                    if rel_dist > 10 or unary_dist > 10:
                        continue

                    baseA[index][i] = -1
                    baseA[index][j] = 1
                    baseb[index] = relative_[i][j]
                    index +=1

                baseA[index][i] = lambda_weight  * 1
                baseb[index] = lambda_weight * unary_[i]
                index += 1

            if index > 0:
                baseA = baseA[0:index, :]
                baseb = baseb[0:index, :]
                baseIndex = index
                new_unary = np.linalg.lstsq(baseA, baseb, rcond=None)[0]
                
                ## Now do updates for the objects that are not very relevant in the scene.
                for i in range(n_objects):
                    if i in valid_indices:
                        continue
                    A = np.zeros((n_objects* (n_objects + 1), n_objects))
                    b = np.zeros((n_objects * (n_objects + 1), 3))
                    A[0:baseIndex] = baseA[0:baseIndex]
                    baseb[0:baseIndex] = baseb[0:baseIndex]
                    index = baseIndex 
                    for j in range(n_objects):
                        rel_dist = np.linalg.norm(relative_[i, j , :])
                        unary_dist = np.linalg.norm(unary_[i] - unary_[j])
                        
                        if rel_dist > 10 or unary_dist > 10:
                            continue
                        A[index][i] = -1
                        A[index][j] =  1
                        b[index] = relative_[i][j]
                        index += 1

                        A[index][i] = 1
                        A[index][j] =  -1
                        b[index] = relative_[j][i]
                        index += 1
                    A[index] = lambda_weight * 1
                    b[index] = lambda_weight * unary_[i]
                    index +=1 
                    A = A[0:index]
                    b = b[0:index]
                    try:
                        soln = np.linalg.lstsq(A, b, rcond=None)[0]
                        new_unary[i] = soln[i]
                    except np.linalg.linalg.LinAlgError:
                        new_unary[i] = unary_[i]
                    

        return new_unary

    def optimize_rotation_unaries(self, unary_rotation, unary_translation, relative_direction, unary_rotation_medoids_,
     relative_direction_medoids_, valid_indices, lambda_weight_scalar=5.0, dist_threshold=4.0):


        absolute_locations = unary_translation
        absolute_log_probabilites = unary_rotation
        n_objects = len(absolute_log_probabilites)
        n_absoulte_bins = absolute_log_probabilites.shape[1]
        relative_direction_prediction = relative_direction.reshape(n_objects, n_objects, -1)
        n_relative_bins = relative_direction_prediction.shape[2]
        bin_scores = np.zeros((n_objects, n_objects, n_absoulte_bins))
        quat_medoids = unary_rotation_medoids_.numpy()
        direction_medoids = relative_direction_medoids_.numpy()
        new_probability = absolute_log_probabilites
        # lambda_weight = opts.lambda_weight * 1./n_objects
        lambda_weight = np.ones((n_objects,))

        for nx in range(n_objects):
            ignore_bin_scores = False
            for mx in range(n_objects):
                if mx == nx:
                    continue
                if mx not in valid_indices:
                    continue

                expected_direction = absolute_locations[mx] - absolute_locations[nx] ## make it unit norm
                dist = (1E-5 + np.linalg.norm(expected_direction))
                if dist > 4 or dist < 1E-3: ## Either the objects are too close or they are the same. We have duplicates coming from proposals.
                    continue

                lambda_weight[nx] += 1

                expected_direction = expected_direction/ (1E-5 + np.linalg.norm(expected_direction))
                expected_direction = expected_direction.reshape(1, -1)
                alignment_scores = []
                indices = []
                for  abinx in range(n_absoulte_bins):
                    prob_bin = absolute_log_probabilites[nx][abinx]
                    quaternion_abinx = quat_medoids[abinx]
                    rotation = transformations.quaternion_matrix(quaternion_abinx)
                    transform = rotation.copy()
                    transform[0:3, 3] = np.array(absolute_locations[nx], copy=True)

                    relative_direction = direction_medoids
                    predicted_direction = suncg_parse.transform_coordinates(transform,
                                         relative_direction) -absolute_locations[nx].reshape(1, -1)
        

                    alignment_score = (1 - np.matmul(expected_direction, predicted_direction.transpose()).squeeze())
                    index = np.argmin(alignment_score, axis=0)
                    alignment_score = np.min(alignment_score, axis=0) + relative_direction_prediction[nx, mx, index]# absolute_log_probabilites[nx][abinx]
                    alignment_score = np.min(relative_direction_prediction[nx, mx, index])
                    alignment_scores.append(alignment_score)

                temp = np.array([metrics.quat_dist(quat_medoids[0], quat_medoids[k]) for k in range(0,24)]).round(2)
                alignment_scores = np.exp(np.array(alignment_scores))
                alignment_scores = np.log(alignment_scores/np.sum(alignment_scores) + 1E-10)
                bin_scores[nx,mx,:] = alignment_scores

        bin_scores = np.sum(bin_scores, axis=1)
        bin_scores = np.exp(bin_scores)
        bin_scores = np.log(1E-10 + bin_scores/np.sum(bin_scores, 1, keepdims=True))
        lambda_weight = np.clip(lambda_weight_scalar * 1.0/lambda_weight, a_max=1, a_min=0)
        lambda_weight = lambda_weight.reshape(-1, 1)
        new_probability = 1.0 * new_probability  + lambda_weight * bin_scores
        new_probability = torch.from_numpy(new_probability).float()
        new_probability = torch.nn.functional.normalize(new_probability.exp(),1).log()
        return new_probability.numpy()

    def rescore_detections(self, valid_indices, scores, unary_trans, unary_scale, unary_rotation, relative_trans, relative_scale, relative_rotation):
        n_objects = len(unary_trans)
        new_scores = np.array(scores, copy=True).squeeze()
        relative_trans = relative_trans.reshape(n_objects, n_objects, -1)
        relative_scale = relative_scale.reshape(n_objects, n_objects, -1)
        relative_rotation = relative_rotation.reshape(n_objects, n_objects, -1)
        relative_trans_errors = [[0] for _ in range(n_objects)]
        relative_scale_errors = [[0] for _ in range(n_objects)]
        relative_rotation_errors = np.zeros((n_objects, 1))

        for ix in range(n_objects):
            for jx in range(n_objects):
                if (ix == jx) or (jx not in valid_indices):
                    continue;
                    # We don't use this to improve the confidence.
                err = np.linalg.norm((unary_trans[jx] - unary_trans[ix]) - relative_trans[ix][jx]) + np.linalg.norm((unary_trans[ix] - unary_trans[jx]) - relative_trans[jx][ix])
                relative_trans_errors[ix].append(err / 2)

                err = np.linalg.norm((unary_scale[jx] - unary_scale[ix]) - relative_scale[ix][jx]) + np.linalg.norm((unary_scale[ix] - unary_scale[jx]) - relative_scale[jx][ix])
                relative_scale_errors[ix].append(err / 2)

        relative_trans_errors = np.clip(np.array([np.median(x) for x in relative_trans_errors]).squeeze(), a_min=0.25, a_max=10)
        relative_scale_errors = np.array([np.median(x) for x in relative_scale_errors]).squeeze()
        # pdb.set_trace()
        new_scores = new_scores - 0.01 * relative_trans_errors - 0.0* relative_scale_errors
        new_scores = new_scores.reshape(scores.shape)
        return new_scores





    def predict(self):
        opts = self.opts
        feed_dict = {}
        feed_dict['imgs_inp_fine'] = self.input_imgs_fine
        feed_dict['imgs_inp_coarse'] = self.input_imgs
        feed_dict['rois_inp'] = self.rois
        feed_dict['location_inp'] = self.object_locations
        feed_dict['class_inp'] = self.object_classes
        feed_dict['spatial_image'] = self.spatial_image
        
        min_threshold_eval = 0.05
        relative_update_threshold = opts.relative_update_threshold
        max_proposals = 80
       
        # labels_pred = self.model.forward_labels(feed_dict)
        labels_pred = self.detection_model.forward_labels(feed_dict)
        scores_pred = labels_pred.cpu().data.numpy()
        bboxes_pred = self.rois.data.cpu().numpy()[:, 1:]
        min_score_eval = np.minimum(min_threshold_eval, np.max(scores_pred))

        pos_inds_eval = metrics.nms(
            np.concatenate((bboxes_pred, scores_pred), axis=1),
            0.3, min_score=min_score_eval)
        if len(pos_inds_eval) > max_proposals:
            pos_inds_eval = pos_inds_eval[0:max_proposals]

        labels_pred = self.filter_pos([labels_pred], pos_inds_eval)[0]
        scores_pred = labels_pred.cpu().data.numpy()
        self.rois_pos_eval = self.filter_pos([self.rois], pos_inds_eval)[0]
        self.rois = self.rois_pos_eval
        feed_dict['rois_inp'] = self.rois
        model_pred, _ = self.model.forward(feed_dict)

        # labels_pred = model_pred['labels_pred']
        
        bboxes_pred = self.rois.data.cpu().numpy()[:, 1:]
        min_score_eval = np.minimum(min_threshold_eval, np.max(scores_pred))
        pos_inds_eval = metrics.nms(
            np.concatenate((bboxes_pred, scores_pred), axis=1),
            0.3, min_score=0.0)
        codes_pred_all = model_pred['codes_pred']
        codes_pred_all['quat'] = torch.nn.functional.log_softmax(codes_pred_all['quat'])

        self.codes_pred_eval = self.filter_pos(codes_pred_all, pos_inds_eval)
        self.rois_pos_eval = self.filter_pos([self.rois], pos_inds_eval)[0]     # b x 5, 1:5 is box (x1 y1 x2 y2)

        

        valid_indices_relative = np.where(scores_pred  > relative_update_threshold)[0]
        if opts.do_updates and opts.pred_relative:
            unary_trans = self.codes_pred_eval['trans'].data.cpu().numpy()
            relative_trans = model_pred['codes_relative']['relative_trans'].data.cpu().numpy()
            new_trans = self.optimize_unaries_strat2(unary_trans, relative_trans,
             valid_indices_relative, do_updates=opts.do_updates, lambda_weight=1.0)

            unary_scale = self.codes_pred_eval['scale'].data.cpu().log().numpy()
            relative_scale = model_pred['codes_relative']['relative_scale'].data.cpu().numpy()
            new_scale = self.optimize_unaries_strat2(unary_scale, relative_scale,
             valid_indices_relative, do_updates=opts.do_updates, lambda_weight=1.0)
            

            unary_rotation = self.codes_pred_eval['quat'].data.cpu().numpy() ## log prob
            relative_direction = model_pred['codes_relative']['relative_dir'].data.cpu().numpy()
            unary_rotation_medoids = self.quat_medoids
            relative_direction_medoids = self.direction_medoids
            new_rotation = self.optimize_rotation_unaries(unary_rotation, unary_trans, 
                                            relative_direction, unary_rotation_medoids,
                                            relative_direction_medoids, valid_indices_relative,
                                            lambda_weight_scalar=opts.lambda_weight, dist_threshold=4.0)

            self.codes_pred_eval['quat'] = Variable(torch.from_numpy(new_rotation).exp().cuda())
            self.codes_pred_eval['trans'] = Variable(torch.from_numpy(new_trans).cuda())
            self.codes_pred_eval['scale'] = Variable(torch.from_numpy(new_scale).exp().cuda())

        min_score_vis = np.minimum(0.7, np.max(scores_pred))
        pos_inds_vis = metrics.nms(
            np.concatenate((bboxes_pred, scores_pred), axis=1),
            0.3, min_score=min_score_vis)
        self.codes_pred_vis = self.filter_pos(self.codes_pred_eval, pos_inds_vis)

        self.codes_pred_eval['shape'] = self.decode_shape(self.codes_pred_eval['shape'])    # b x 1 x 32 x 32 x 32
        self.codes_pred_eval['quat'] = self.decode_rotation(self.codes_pred_eval['quat']) # b x 4
        self.codes_pred_eval['scale'] # Probably scale b x 3
        self.codes_pred_eval['trans'] # Probably trans b x 3
        self.scores_pred_eval = scores_pred[pos_inds_eval,:]*1.

        self.rois_pos_vis = self.filter_pos([self.rois], pos_inds_vis)[0]
        self.codes_pred_vis['shape'] = self.decode_shape(self.codes_pred_vis['shape'])
        self.codes_pred_vis['quat'] = self.decode_rotation(self.codes_pred_vis['quat'])
        self.layout_pred = self.layout_model.forward(self.input_imgs_layout)


    def evaluate(self):
        # rois as numpy array
        # Get Predictions.
        shapes = self.codes_pred_eval['shape'] 
        scales = self.codes_pred_eval['scale']
        rots = self.codes_pred_eval['quat']
        trans = self.codes_pred_eval['trans']
        # shapes, scales, rots, trans = self.codes_pred_eval
        scores = self.scores_pred_eval
        boxes = self.rois_pos_eval.cpu().data.numpy()[:,1:]
        # Get Ground Truth.

        gt_shapes = self.codes_gt['shape']
        gt_scales = self.codes_gt['scale']
        gt_rots = self.codes_gt['quat']
        gt_trans = self.codes_gt['trans']
        # gt_shapes, gt_scales, gt_rots, gt_trans = self.codes_gt
        gt_boxes = self.rois_gt.cpu().data.numpy()[:,1:]

        iou_box = bbox_utils.bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

        trans_, gt_trans_ = trans.cpu().data.numpy(), gt_trans.cpu().data.numpy()
        err_trans = np.linalg.norm(np.expand_dims(trans_, 1) - np.expand_dims(gt_trans_, 0), axis=2)

        scales_, gt_scales_ = scales.cpu().data.numpy() + 1E-10, gt_scales.cpu().data.numpy() + 1E-10
        err_scales = np.mean(np.abs(np.expand_dims(np.log(scales_), 1) - np.expand_dims(np.log(gt_scales_), 0)), axis=2)
        err_scales /= np.log(2.0)

        ndt, ngt = err_scales.shape
        err_shapes = err_scales*0.
        err_rots = err_scales*0.

        for i in range(ndt):
          for j in range(ngt):
            err_shapes[i,j] = metrics.volume_iou(shapes[i,0].data, gt_shapes[j,0].data, thresh=self.opts.voxel_eval_thresh)
            q_errs = []
            for gt_quat in gt_rots[j]:
                q_errs.append( metrics.quat_dist(rots[i].data.cpu(), gt_quat.data.cpu()))
            err_rots[i,j] = min(q_errs)

        # Run the benchmarking code here.
        threshs = [EP_box_iou_thresh, EP_rot_delta_thresh, EP_trans_delta_thresh, EP_shape_iou_thresh, EP_scale_delta_thresh]
        fn = [np.greater_equal, np.less_equal, np.less_equal, np.greater_equal, np.less_equal]
        overlaps = [iou_box, err_rots, err_trans, err_shapes, err_scales]

        _dt = {'sc': scores}
        _gt = {'diff': np.zeros((ngt,1), dtype=np.bool)}
        _bopts = {'minoverlap': 0.5}
        stats = []
        for i in range(len(EP_ap_str)):
          # Compute a single overlap that ands all the thresholds.
            ov = []
            for j in range(len(overlaps)):
                ov.append(fn[j](overlaps[j], threshs[j][i]))
            _ov = np.all(np.array(ov), 0).astype(np.float32)
            # Benchmark for this setting.
            tp, fp, sc, num_inst, dup_det, inst_id, ov = evaluate_detection.inst_bench_image(_dt, _gt, _bopts, _ov)
            stats.append([tp, fp, sc, num_inst, dup_det, inst_id, ov])
        return stats

    def get_current_visuals(self):
        visuals={}
        opts=self.opts
        visuals['img']=visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))
        rois=self.rois.data
        visuals['img_roi']=render_utils.vis_detections(visuals['img'], rois[:, 1:])

        mesh_dir=osp.join(opts.rendering_dir)
        # vis_codes=[self.codes_pred_vis, self.codes_gt]
        vis_codes=[self.codes_pred_vis, self.codes_gt]
        # vis_layouts = [self.layout_pred, self.layout_gt]
        vis_names=['b_pred', 'c_gt']
        for vx, v_name in enumerate(vis_names):
            os.system('rm {}/*.obj'.format(mesh_dir))
            self.save_codes_mesh(mesh_dir, vis_codes[vx])
          
            visuals['{}_objects_cam_view'.format(v_name)] =self.render_visuals(mesh_dir, obj_name='codes')
            # visuals['{}_scene_cam_view'.format(v_name)] =self.render_visuals(mesh_dir)
        return visuals

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


    def test_draw(self):
        base_dir=osp.join(self.opts.results_quality_dir)
        if not osp.exists(base_dir):
            os.makedirs(base_dir)
        opts=self.opts
        house_name_view_ids=[]
        index_filename=opts.index_file
        if index_filename is not None:
            with open(index_filename) as f:
                for line in f:
                    line=line.strip()
                    house_name_view_ids.append('_'.join(line.split('_')))

        rng = np.random.RandomState(1)
        indices = rng.choice(len(self.dataloader), 200)
        if len(house_name_view_ids) == 0:
            dump_to_file = True
            dump_file_fd = open(osp.join(base_dir,'random_house_views.txt'),'w') 

       
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
                bench_image_stats = self.evaluate()
                self.save_current_visuals(house_name, view_id)
                # self.save_current_stats(bench_image_stats, house_name, view_id)
                print("Generating {}".format(i))
                if dump_to_file:
                    dump_file_fd.write("{}_{}\n".format(house_name, view_id))

        if dump_to_file:
            dump_file_fd.close()




    def test(self):
        opts = self.opts
        bench_stats = []
        # codes are (shapes, scales, quats, trans)
        n_iter = len(self.dataloader)
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
                bench_image_stats = self.evaluate()
                bench_stats.append(bench_image_stats)
                if opts.save_visuals and (i % opts.visuals_freq == 0):
                    self.save_current_visuals(house_name, view_id)

        # Accumulate stats, write to FLAGS.results_eval_dir.
        bb = zip(*bench_stats)
        bench_summarys = []
        eval_params = {}
        for i in range(len(EP_ap_str)):
            tp, fp, sc, num_inst, dup_det, inst_id, ov = zip(*bb[i])
            ap, rec, prec, npos, details = evaluate_detection.inst_bench(None, None, None, tp, fp, sc, num_inst)
            bench_summary = {'prec': prec.tolist(), 'rec': rec.tolist(), 'ap': ap[0], 'npos': npos}
            print('{:>20s}: {:5.3f}'.format(EP_ap_str[i], ap[0]*100.))
            bench_summarys.append(bench_summary)
        eval_params['ap_str'] = EP_ap_str
        eval_params['set'] = opts.eval_set

        json_file = os.path.join(FLAGS.results_eval_dir, 'eval_set{}_{}_{}.json'.format(opts.eval_set, 0, opts.do_updates))
        print('Writing results to file: {:s}'.format(json_file))
        with open(json_file, 'w') as f:
            json.dump({'eval_params': eval_params, 'bench_summary': bench_summarys}, f)

def main(_):
    FLAGS.suncg_dl_out_codes = True
    FLAGS.suncg_dl_out_fine_img = True
    FLAGS.suncg_dl_out_test_proposals = True
    FLAGS.suncg_dl_out_voxels = False
    FLAGS.suncg_dl_out_layout = True
    FLAGS.suncg_dl_out_depth = False
    # FLAGS.n_data_workers = 0
    FLAGS.max_views_per_house = 2
    FLAGS.batch_size = 1
    assert(FLAGS.batch_size == 1)

    results_name=FLAGS.name
    FLAGS.results_vis_dir = osp.join(FLAGS.results_vis_dir, 'dwr', FLAGS.eval_set, FLAGS.name)
    FLAGS.results_quality_dir=osp.join(FLAGS.results_quality_dir, 'dwr', FLAGS.eval_set, results_name)
    FLAGS.results_eval_dir = osp.join(FLAGS.results_eval_dir, 'dwr', FLAGS.eval_set, FLAGS.name)
    FLAGS.rendering_dir = osp.join(FLAGS.rendering_dir,'dwr', results_name)
    if not os.path.exists(FLAGS.results_eval_dir):
        os.makedirs(FLAGS.results_eval_dir)
    if not os.path.exists(FLAGS.results_vis_dir):
        os.makedirs(FLAGS.results_vis_dir)
    torch.manual_seed(0)

    if FLAGS.classify_rot:
        FLAGS.nz_rot = 24
    else:
        FLAGS.nz_rot = 4


    tester = DWRTester(FLAGS)
    tester.init_testing()
    if FLAGS.draw_vis:
        tester.test_draw()
    else:
        tester.test()
    pdb.set_trace()


if __name__ == '__main__':
    app.run(main)
