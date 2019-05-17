"""Script for dwr experiment.
"""
# Sample usage: 

# (init) : python -m factored3d.experiments.suncg.dwr --name=dwr_base --classify_rot --pred_voxels=False --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=1000 --display_id=100 --box3d_ft --shape_loss_wt=10 --label_loss_wt=10  --batch_size=8

# shape_ft : python -m factored3d.experiments.suncg.dwr --name=dwr_shape_ft --classify_rot --pred_voxels=True --shape_dec_ft --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=1000 --display_id=100 --shape_loss_wt=10  --label_loss_wt=10 --batch_size=8 --ft_pretrain_epoch=1

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

from ...data import suncg as suncg_data
from ...utils import suncg_parse
from ...nnutils import train_utils
from ...nnutils import net_blocks
from ...nnutils import loss_utils
from ...nnutils import oc_net
from ...nnutils import disp_net
from ...utils import visutil
from ...renderer import utils as render_utils

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'), 'Directory where intermittent renderings are saved')

flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')
flags.DEFINE_boolean('shape_dec_ft', False, 'If predicting voxels, should we pretrain from an existing deocder')

flags.DEFINE_string('box3d_pretrain_name', 'box3d_base', 'Experiment name for pretrained box3d experiment')
flags.DEFINE_integer('box3d_pretrain_epoch', 8, 'Experiment name for shape decoder')
flags.DEFINE_boolean('box3d_ft', False, 'Finetune from existing net trained with gt boxes')

flags.DEFINE_string('ft_pretrain_name', 'dwr_base', 'Experiment name from which we will pretrain the OCNet')
flags.DEFINE_integer('ft_pretrain_epoch', 0, 'Network epoch from which we will finetune')

flags.DEFINE_float('label_loss_wt', 1, 'Label loss weight')

flags.DEFINE_string('set', 'train', 'dataset to use')

flags.DEFINE_integer('max_rois', 100, 'If we have more objects than this per image, we will subsample. Set to very large value')
flags.DEFINE_integer('max_total_rois', 100, 'If we have more objects than this per batch, we will reject the batch.')
flags.DEFINE_float('split_size', 1.0, 'Split size of the train set')

FLAGS = flags.FLAGS


class DWRTrainer(train_utils.Trainer):
    def define_model(self):
        '''
        Define the pytorch net 'model' whose weights will be updated during training.
        '''

        opts = self.opts

        assert(not (opts.ft_pretrain_epoch > 0 and opts.num_pretrain_epochs > 0))
        assert(not (opts.ft_pretrain_epoch > 0 and opts.box3d_ft))
        assert(not (opts.num_pretrain_epochs > 0 and opts.box3d_ft))

        self.voxel_encoder, nc_enc_voxel = net_blocks.encoder3d(
            opts.n_voxel_layers, nc_max=opts.voxel_nc_max, nc_l1=opts.voxel_nc_l1, nz_shape=opts.nz_shape)

        self.voxel_decoder = net_blocks.decoder3d(
            opts.n_voxel_layers, opts.nz_shape, nc_enc_voxel, nc_min=opts.voxel_nc_l1)
        self.model = oc_net.OCNet(
            (opts.img_height, opts.img_width), opts=opts,
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=opts.pred_voxels, nz_shape=opts.nz_shape,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot, b_size=opts.batch_size,
            pred_labels=True, filter_positives=True)

        if opts.box3d_ft:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.box3d_pretrain_name)
            self.load_network(
                self.model,
                'pred', opts.box3d_pretrain_epoch, network_dir=network_dir)

        self.model.add_label_predictor()

        if opts.ft_pretrain_epoch > 0:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.ft_pretrain_name)
            self.load_network(
                self.model, 'pred', opts.ft_pretrain_epoch, network_dir=network_dir)

        if opts.pred_voxels:
            self.model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))

        if opts.pred_voxels and opts.shape_dec_ft:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.model.code_predictor.shape_predictor.decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)

        if self.opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs-1)
        self.model = self.model.cuda(device=self.opts.gpu_id)
        self.lsopt = loss_utils.LeastSquareOpt()

        ## Turn Batch_norm Off since you are finetuning.
        # net_blocks.turnBNoff(self.model)
        return

    def init_dataset(self):
        opts = self.opts
        self.real_iter = 1 # number of iterations we actually updated the net for
        self.data_iter = 1 # number of iterations we called the data loader
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        if opts.set == 'val':
            self.train_mode = False
            self.model.eval()
        else:
            rng = np.random.RandomState(10)
            rng.shuffle(self.split[opts.set])
            len_splitset = int(len(self.split[opts.set])*opts.split_size)
            self.split[opts.set] = self.split[opts.set][0:len_splitset]

        self.dataloader = suncg_data.suncg_data_loader(self.split[opts.set], opts)

        if not opts.pred_voxels:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.voxel_encoder,
                'encoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.load_network(
                self.voxel_decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.voxel_encoder.eval()
            self.voxel_encoder = self.voxel_encoder.cuda(device=self.opts.gpu_id)
            self.voxel_decoder.eval()
            self.voxel_decoder = self.voxel_decoder.cuda(device=self.opts.gpu_id)

        if opts.voxel_size < 64:
            self.downsample_voxels = True
            self.downsampler = render_utils.Downsample(
                64//opts.voxel_size, use_max=True, batch_mode=True
            ).cuda(device=self.opts.gpu_id)


        self.spatial_image = Variable(suncg_data.define_spatial_image(opts.img_height_fine, opts.img_width_fine, 1.0/16).unsqueeze(0).cuda())

        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)
            self.quat_medoids_var = Variable(self.quat_medoids).cuda()
            self.class_weights = None

        self.direction_medoids = torch.from_numpy(
            scipy.io.loadmat(osp.join(opts.cache_dir, 'direction_medoids_relative_{}_new.mat'.format(opts.nz_rel_dir)))['medoids']).type(torch.FloatTensor)
        self.direction_medoids_var = Variable(self.direction_medoids).cuda()


    def define_criterion(self):
        self.smoothed_factor_losses = {
            'shape': 0, 'scale': 0, 'quat': 0, 'trans': 0, 'rel_trans' : 0, 'rel_scale' : 0, 'rel_quat' : 0, 'class' : 0, 'var_mean':0, 'var_std':0, 'var_mean_rel_dir':0, 'var_std_rel_dir':0,
        }
        self.labels_criterion = torch.nn.BCELoss()
        self.smoothed_label_loss = 0

    def set_input(self, batch):
        opts = self.opts
        # pdb.set_trace()
        if batch is None or not batch or batch['empty']:
            self.invalid_batch = True
            return


        rois = suncg_parse.bboxes_to_rois(batch['bboxes_proposals'])
        roi_labels = batch['labels_proposals']
        self.data_iter += 1
        if roi_labels.sum() <= 1 or roi_labels.sum() >= opts.max_total_rois * (1.0 * opts.batch_size / 8.0): #with just one element, batch_norm will screw up
            self.invalid_batch = True
            return
        else:
            gt_rois = suncg_parse.bboxes_to_rois2(batch['bboxes'])
            pairs = self.count_number_pairs(gt_rois)
            if self.opts.only_pairs and pairs <= 1:
                self.invalid_batch = True
                print("Ignoring due to lack of pairs")
                return
            self.invalid_batch = False
            self.real_iter += 1

        self.house_names = batch['house_name']
        self.view_ids = batch['view_id']
        input_imgs_fine = batch['img_fine'].type(torch.FloatTensor)
        input_imgs = batch['img'].type(torch.FloatTensor)
        for b in range(input_imgs_fine.size(0)):
            input_imgs_fine[b] = self.resnet_transform(input_imgs_fine[b])
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        self.input_imgs = Variable(
            input_imgs.cuda(device=opts.gpu_id), requires_grad=False)

        self.input_imgs_fine = Variable(
            input_imgs_fine.cuda(device=opts.gpu_id), requires_grad=False)

        self.rois = Variable(
            rois.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)

        self.roi_labels = Variable(
            roi_labels.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)

        code_tensors = suncg_parse.collate_codes(batch['codes_proposals'])
        code_tensors_quats = code_tensors['quat']
        
        pos_inds = roi_labels.squeeze().nonzero().squeeze()
        pos_inds = torch.autograd.Variable(pos_inds.type(torch.LongTensor).cuda(), requires_grad=False)
        self.pos_rois =  torch.index_select(self.rois, 0, pos_inds)
        pos_rois = self.pos_rois.data.cpu()
        object_classes = code_tensors['class'].type(torch.LongTensor)
        self.class_gt = self.object_classes = Variable(object_classes.cuda(), requires_grad=False)
        
        code_tensors['shape'] = code_tensors['shape'].unsqueeze(1)  # unsqueeze voxels

        common_src_classes = []
        common_trj_classes = []
        self.common_masks = []
        
        for bx, ocs in enumerate(suncg_parse.batchify(object_classes, pos_rois[:,0])):
            nx = len(ocs)
            for ix in range(nx):
                for jx in range(nx):
                    common_src_classes.append(ocs[ix])
                    common_trj_classes.append(ocs[jx])

        common_src_classes = Variable(torch.cat(common_src_classes).cuda(async=True), requires_grad=False)
        common_trj_classes = Variable(torch.cat(common_trj_classes).cuda(async=True), requires_grad=False)
        self.common_classes = [common_src_classes, common_trj_classes]
        self.object_locations = suncg_parse.batchify(code_tensors['trans'], pos_rois[:,0])

        cam2voxs = code_tensors['transform_cam2vox']
        cam2voxs = suncg_parse.batchify(cam2voxs, pos_rois[:,0])
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
        
        self.relative_dir_mask = Variable(torch.cat(relative_dir_mask,dim=0).byte()).cuda(async=True)
        if opts.classify_dir and not opts.gmm_dir:
            relative_direction_rotation_binned  = [suncg_parse.directions_to_bininds(t, self.direction_medoids) for t in relative_direction_rotation]
            relative_direction_rotation_directions = [suncg_parse.bininds_to_directions(t, self.direction_medoids) for t in relative_direction_rotation_binned]
            self.relative_direction_rotation_unquantized = [Variable(t).cuda(async=True) for t in relative_direction_rotation]
            self.relative_direction_rotation = [Variable(t).cuda(async=True) for t in relative_direction_rotation_binned]
        else:
            self.relative_direction_rotation = [Variable(t).cuda(async=True) for t in relative_direction_rotation]

        if opts.classify_rot and not opts.gmm_rot:
            code_tensors_quats = [suncg_parse.quats_to_bininds(code_quat, self.quat_medoids) for code_quat in code_tensors_quats]


        self.codes_gt_quats = [
            Variable(t.cuda(async=True), requires_grad=False) for t in code_tensors_quats]
        codes_gt_keys = ['shape', 'scale', 'trans']
        self.codes_gt  ={key : Variable(code_tensors[key].cuda(async=True), requires_grad=False) 
                            for key in codes_gt_keys}
        self.codes_gt['quat'] = self.codes_gt_quats
       
        self.object_locations = [Variable(t_.cuda(async=True), requires_grad=False) for t_ in
                                 self.object_locations]
        self.object_scales = suncg_parse.batchify(code_tensors['scale'] + 1E-10, pos_rois[:,0])
        self.object_scales = [Variable(t_.cuda(async=True), requires_grad=False) for t_ in
                                 self.object_scales]


        self.relative_trans_gt = []
        self.relative_scale_gt = []
        for bx in range(len(self.object_locations)):
            relative_locations = self.object_locations[bx].unsqueeze(0) - self.object_locations[bx].unsqueeze(1)
            relative_locations = relative_locations.view(-1, 3)
            self.relative_trans_gt.append(relative_locations)
            relative_scales = self.object_scales[bx].unsqueeze(0).log() - self.object_scales[bx].unsqueeze(1).log() ## this is in log scale.
            relative_scales = relative_scales.view(-1, 3)
            self.relative_scale_gt.append(relative_scales)

        self.relative_scale_gt = torch.cat(self.relative_scale_gt, dim=0) # this is in log scale
        self.relative_trans_gt = torch.cat(self.relative_trans_gt, dim=0)


        if self.downsample_voxels:
            self.codes_gt['shape'] = self.downsampler.forward(self.codes_gt['shape'])

        if not opts.pred_voxels:
            self.codes_gt['shape'] = self.voxel_encoder.forward(self.codes_gt['shape'])

        self.relative_gt = {'relative_trans' : self.relative_trans_gt,
                    'relative_scale' : self.relative_scale_gt,
                    'relative_dir' : self.relative_direction_rotation,
                    'relative_mask' : self.relative_dir_mask,
                    }
        return



    def get_current_scalars(self):
        loss_dict = {'total_loss': self.smoothed_total_loss, 'iter_frac': self.real_iter/self.data_iter}
        loss_dict['label_loss'] = self.smoothed_label_loss
        for k in self.smoothed_factor_losses.keys():
            loss_dict['loss_' + k] = self.smoothed_factor_losses[k]
        return loss_dict

    def get_current_visuals(self):
        visuals = {}
        opts = self.opts
        visuals['img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))
        return visuals

    def get_current_points(self):
        pts_dict = {}
        return pts_dict

    def forward(self):
        opts = self.opts
        feed_dict = {}
        feed_dict['imgs_inp_fine'] = self.input_imgs_fine
        feed_dict['imgs_inp_coarse'] = self.input_imgs
        feed_dict['rois_inp'] = self.rois
        feed_dict['location_inp'] = self.object_locations
        feed_dict['class_inp'] = self.object_classes
        feed_dict['spatial_image'] = self.spatial_image
        feed_dict['roi_labels']  =  self.roi_labels
        all_predictions, _ = self.model.forward(feed_dict)
        
        self.bIndices_pairs = []
        self.bIndices = suncg_parse.batchify(self.pos_rois.data[:,0], self.pos_rois.data[:,0])
        self.bIndices_pairs = torch.cat([a.repeat(a.size(0)) for a in self.bIndices]).view(-1,1)

        self.codes_pred = all_predictions['codes_pred']
        self.relative_trans_predictions = None
        self.relative_predictions = None
        if opts.pred_relative:
            self.relative_predictions = all_predictions['codes_relative']

        class_pred = None
        if opts.pred_class:
            class_pred = all_predictions['class_pred']

        common_class_predictions = None
        if opts.common_pred_class:
            common_class_predictions = all_predictions['common_class_pred']

        self.total_loss, self.loss_factors = loss_utils.code_loss(
            self.codes_pred, self.codes_gt, self.pos_rois,
            self.relative_predictions, self.relative_gt, self.bIndices_pairs,
            class_pred, self.class_gt.squeeze(),
            quat_medoids = self.quat_medoids_var, direction_medoids = self.direction_medoids_var,
            pred_class=opts.pred_class,
            pred_voxels=opts.pred_voxels,
            classify_rot=opts.classify_rot,
            classify_dir = opts.classify_dir,
            shape_wt=opts.shape_loss_wt,
            scale_wt=opts.scale_loss_wt,
            quat_wt=opts.quat_loss_wt,
            trans_wt=opts.trans_loss_wt,
            pred_relative = opts.pred_relative,
            rel_trans_wt=opts.rel_trans_loss_wt,
            rel_quat_wt=opts.rel_quat_loss_wt,
            lsopt = self.lsopt, rel_opt=opts.rel_opt,
            class_weights=None,
            opts = opts
        )

        self.labels_pred = all_predictions['labels_pred']
        labels_loss = self.labels_criterion.forward(self.labels_pred, self.roi_labels.unsqueeze(1))
        self.total_loss += opts.label_loss_wt*labels_loss

        for k in self.smoothed_factor_losses.keys():
            if 'var' in k:
                self.smoothed_factor_losses[k] = self.loss_factors[k].item()
            else:
                self.smoothed_factor_losses[k] = 0.99*self.smoothed_factor_losses[k] + 0.01*self.loss_factors[k].item()
        self.smoothed_label_loss = 0.99*self.smoothed_label_loss + 0.01*labels_loss.item()*opts.label_loss_wt


def main(_):
    seed = 206
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    FLAGS.suncg_dl_out_codes = True
    FLAGS.suncg_dl_out_fine_img = True
    FLAGS.suncg_dl_out_proposals = True
    FLAGS.suncg_dl_out_voxels = False
    FLAGS.suncg_dl_out_layout = False
    FLAGS.suncg_dl_out_depth = False
  

    if not FLAGS.classify_rot:
        FLAGS.nz_rot = 4

    trainer = DWRTrainer(FLAGS)
    trainer.init_training()
    trainer.train()
    pdb.set_trace()


if __name__ == '__main__':
    app.run(main)
