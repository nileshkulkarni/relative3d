"""
Testing class for the demo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import scipy.misc
import pdb
import copy
import scipy.io as sio

from ..nnutils import test_utils
from ..nnutils import net_blocks
from ..nnutils import voxel_net
from ..nnutils import oc_net
from ..nnutils import disp_net

from ..utils import visutil
from ..utils import suncg_parse
from ..utils import metrics
from ..utils import transformations
from ..renderer import utils as render_utils


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'), 'Directory where intermittent renderings are saved')

flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')
flags.DEFINE_float('voxel_eval_thresh', 0.25, 'Voxel evaluation threshold')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')

flags.DEFINE_string('layout_name', 'layout_pred', 'Experiment name for layout predictor')
flags.DEFINE_integer('layout_train_epoch', 8, 'Experiment name for layout predictor')

flags.DEFINE_string('depth_name', 'depth_baseline', 'Experiment name for layout predictor')
flags.DEFINE_integer('depth_train_epoch', 8, 'Experiment name for layout predictor')

flags.DEFINE_string('scene_voxels_name', 'voxels_baseline', 'Experiment name for layout predictor')
flags.DEFINE_integer('scene_voxels_train_epoch', 8, 'Experiment name for layout predictor')
flags.DEFINE_float('scene_voxels_thresh', 0.25, 'Threshold for scene voxels prediction')

flags.DEFINE_integer('img_height', 128, 'image height')
flags.DEFINE_integer('img_width', 256, 'image width')
flags.DEFINE_integer('max_object_classes', 10, 'maximum object classes')

flags.DEFINE_boolean('dwr_model', False, 'Load a dwr mode ')
flags.DEFINE_boolean('use_gt_voxels', True, 'Load a gt voxel ')
flags.DEFINE_boolean('pred_labels', True, ' Pred labels ')
flags.DEFINE_integer('img_height_fine', 480, 'image height')
flags.DEFINE_integer('img_width_fine', 640, 'image width')

flags.DEFINE_integer('layout_height', 64, 'amodal depth height : should be half image height')
flags.DEFINE_integer('layout_width', 128, 'amodal depth width : should be half image width')

flags.DEFINE_integer('voxels_height', 32, 'scene voxels height. Should be half of width and depth.')
flags.DEFINE_integer('voxels_width', 64, 'scene voxels width')
flags.DEFINE_integer('voxels_depth', 64, 'scene voxels depth')
flags.DEFINE_boolean('pretrained_shape_decoder', True, 'Load pretrained shape decoder model, use only when you are using the detector trained on GT boxes')
flags.DEFINE_boolean('do_updates', True, 'Do relative updates')
flags.DEFINE_float('relative_update_threshold', 0.3, 'Prediction score to use in relative update')
flags.DEFINE_float('lambda_weight', 5.0, 'lambda weight ')


class DemoTester(test_utils.Tester):
    
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



    def load_dwr_model(self, ):
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
            roi_size=opts.roi_size, use_context=opts.use_context,
            nz_feat=opts.nz_feat, pred_voxels=False, nz_shape=opts.nz_shape,
            pred_labels=opts.pred_labels, classify_rot=opts.classify_rot, nz_rot=opts.nz_rot,)

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

        if opts.pred_voxels and (not opts.dwr_model):
             self.voxel_decoder = copy.deepcopy(self.model.code_predictor.shape_predictor.decoder)

        self.layout_model = disp_net.dispnet()
        network_dir = osp.join(opts.cache_dir, 'snapshots', opts.layout_name)
        self.load_network(self.layout_model, 'pred', opts.layout_train_epoch, network_dir=network_dir)

        return
    
    def define_model(self,):
        self.load_dwr_model()
        self.preload_detection_pretrained_model()
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
    
    def init_dataset(self,):
        opts = self.opts
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
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

        self.spatial_image = Variable(suncg_parse.define_spatial_image(opts.img_height_fine, opts.img_width_fine, 1.0/16).unsqueeze(0).cuda()) ## (1, 2, 30, 40)
        
        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)
            self.quat_medoids_var = None

        if opts.classify_dir:
            self.direction_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'direction_medoids_relative_{}_new.mat'.format(opts.nz_rel_dir)))['medoids']).type(torch.FloatTensor)
            self.direction_medoids = torch.nn.functional.normalize(self.direction_medoids)
        
        return

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
        rois = bboxes_gt 
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

        self.rois = Variable(rois.type(torch.FloatTensor).cuda(), requires_grad=False)
        
        if 'scores' in batch.keys():
            self.bboxes_proposal_scores = torch.cat(batch['scores']).float().cuda()
        return
    
    
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


    def predict_box3d(self, ):
        opts = self.opts
        feed_dict = {}
        feed_dict['imgs_inp_fine'] = self.input_imgs_fine
        feed_dict['imgs_inp_coarse'] = self.input_imgs
        feed_dict['rois_inp'] = self.rois
        feed_dict['class_inp'] = [None]
        feed_dict['spatial_image'] = self.spatial_image
        
        min_threshold_eval = 0.05
        relative_update_threshold = opts.relative_update_threshold
        max_proposals = 80
        if hasattr(self, 'bboxes_proposal_scores'):
            labels_pred = self.bboxes_proposal_scores.view(-1,1)
        else:
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

        codes_pred_all['quat'] = torch.nn.functional.log_softmax(codes_pred_all['quat'], dim=1)

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
        # self.layout_pred = self.layout_model.forward(self.input_imgs_layout)


        return self.codes_pred_eval
    


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
        render_utils.render_directory(mesh_dir, png_dir)
        im_view1=scipy.misc.imread(osp.join(png_dir, 'render_000.png'))
        im_view2=scipy.misc.imread(osp.join(png_dir, 'render_003.png'))
        return im_view1, im_view2
        

    def update_locations(self, trans_location, relative_locations):
        n_objects=trans_location.size(0)
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
        new_location=np.linalg.lstsq(A, b)
        return torch.from_numpy(new_location[0]), np.linalg.norm(new_location[0] - trans_location, axis=1).tolist()
    
    def render_outputs(self):
        
        opts=self.opts
        visuals = {}
        visuals['img']=visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))
        rois=self.rois.data
        visuals['img_roi']=render_utils.vis_detections(visuals['img'], self.rois_pos_vis[:, 1:])
        
        mesh_dir=osp.join(opts.rendering_dir)
        vis_codes=[self.codes_pred_vis]
        vis_names=['b_pred']
        for vx, v_name in enumerate(vis_names):
            os.system('rm {}/*.obj'.format(mesh_dir))
            self.save_codes_mesh(mesh_dir, vis_codes[vx])
            visuals['{}_objects_cam_view'.format(v_name)], visuals['{}_scene_cam_view'.format(v_name)] =self.render_visuals(mesh_dir, obj_name='codes')
        return visuals
        
