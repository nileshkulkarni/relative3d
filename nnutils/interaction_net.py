'''
Object-centric prediction net.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
from . import net_blocks as nb
from . import nn_flags
from . import roi_pool_py as roi_pool
# from ..utils.roi_pooling.modules.roi_pool import RoIPool
from ..utils.roi_pooling.modules.roi_pool import _RoIPooling as RoIPool
from ..utils import suncg_parse
from . import common_blocks as cb
# from oc3d.nnutils import roi_pooling
import pdb

# -------------- flags -------------#
# ----------------------------------#

flags.DEFINE_boolean('interaction', True, 'Allow interaction')


class EffectEncoder(nn.Module):
    def __init__(self, nz_joint=300):
        super(EffectEncoder, self).__init__()
        self.encoder_joint = nb.fc_stack(nz_joint*2, nz_joint, 3, use_bn=False)

    def forward(self, feats):
        feat_src, feat_trj = feats
        feat_effect = torch.cat([feat_src, feat_trj], dim=1)
        feat_effect = self.encoder_joint.forward(feat_effect)
        return feat_effect



# ------------- Inter Net -------------#
# ----------------------------------#
class InterNet(nn.Module):
    def __init__(
            self, img_size_coarse, opts,
            roi_size=4,
            use_context=True, nz_feat=1000,
            pred_voxels=True, nz_shape=100,
            classify_rot=False, nz_rot=4, pred_graph=True, pred_trj=True, pred_pwd=True,
            pred_labels=False, filter_positives=False, b_size=1, n_g_layers=2, cython_roi=True,
            use_basic = False,
    ):
        super(InterNet, self).__init__()
        self.opts = opts
        self.pred_labels = pred_labels
        self.filter_positives = filter_positives
        self.nz_feat = nz_feat
        self.nz_rot = nz_rot
        self.nz_rel_dir = opts.nz_rel_dir
        self.b_size = b_size
        self.resnet_conv_fine = cb.ResNetConv(n_blocks=3)
        self.resnet_conv_coarse = cb.ResNetConv(n_blocks=4)
        self.roi_size = roi_size
        self.nz_feat = nz_feat
        self.nz_shape = nz_shape
        
        self.use_common = True
        self.max_object_classes = opts.max_object_classes
        self.use_spatial_map = opts.use_spatial_map
        self.var_gmm_rot = opts.var_gmm_rot
        self.gmm_dir = opts.gmm_dir
        self.not_use_basic  = not use_basic

        self.roi_pool = RoIPool(roi_size, roi_size, 1.0 / 16)

        resent_channels = 256
        img_fine_feat_channels = resent_channels
        if self.use_spatial_map and self.not_use_basic:
            img_fine_feat_channels += 2
        
        self.nc_inp_fine = nc_inp_fine = img_fine_feat_channels * roi_size * roi_size
        nc_inp_coarse = 512 * (img_size_coarse[0] // 32) * (img_size_coarse[1] // 32)

        nc_inp_common_layers = (2 + img_fine_feat_channels)

        self.roi_encoder = cb.RoiEncoder(nc_inp_fine, nc_inp_coarse, use_context=use_context, nz_joint=nz_feat, 
            use_object_class=False, max_object_classes=opts.max_object_classes)


        self.effect_encoder = EffectEncoder()
        self.code_predictor = cb.CodePredictor(
            nz_feat=nz_feat,
            pred_voxels=pred_voxels, nz_shape=nz_shape,
            classify_rot=classify_rot, nz_rot=nz_rot,
            var_gmm_rot=self.var_gmm_rot)

        nb.net_init(self.roi_encoder)
        nb.net_init(self.code_predictor)


    def add_label_predictor(self):
        self.label_predictor = cb.LabelPredictor(self.nz_feat)
        nb.net_init(self.label_predictor)
        return

    ## Uncollates features predictions
    def batchify(self, feature, bIndices):
        def uncollate_codes(code_tensors, batch_size, batch_inds):
            '''
            Assumes batch inds are 0 indexed, increasing
            '''
            start_ind = 0
            codes = []
            for b in range(batch_size):
                nelems = torch.eq(batch_inds, b).sum()
                if nelems > 0:
                    codes_b = code_tensors[start_ind:(start_ind + nelems)]
                    start_ind += nelems
                    codes.append(codes_b)
            return codes

        features = uncollate_codes(feature, self.b_size, bIndices)
        return features

    def forward_labels(self, feed_dict):
        imgs_inp_fine = feed_dict['imgs_inp_fine']
        imgs_inp_coarse = feed_dict['imgs_inp_coarse']
        rois_inp = feed_dict['rois_inp']
        location_inp = feed_dict['location_inp']
        class_inp = feed_dict['class_inp']
        spatial_image = feed_dict['spatial_image']
        bs = len(imgs_inp_coarse)
        spatial_image = spatial_image.repeat(bs, 1, 1, 1)
        img_feat_coarse = self.resnet_conv_coarse.forward(imgs_inp_coarse)
        img_feat_coarse = img_feat_coarse.view(img_feat_coarse.size(0), -1)
        img_feat_fine = self.resnet_conv_fine.forward(imgs_inp_fine)

        if self.use_spatial_map and self.not_use_basic:
            img_feat_fine = torch.cat([img_feat_fine, spatial_image], dim=1)

        roi_img_feat = self.roi_pool.forward(img_feat_fine, rois_inp)

        roi_img_feat = roi_img_feat.view(roi_img_feat.size(0), -1)
        roi_feat = self.roi_encoder.forward((roi_img_feat, img_feat_coarse, rois_inp, class_inp))

        if self.pred_labels:
            labels_pred = self.label_predictor.forward(roi_feat)
        
        return labels_pred

    def aggregrate_effects(self, effects):
        nx = len(effects)
        mask = Variable(1 - torch.eye(nx, nx).unsqueeze(-1).expand(effects.size())).cuda()
        effects = mask * effects
        effects = torch.nn.functional.max_pool2d(effects, kernel_size=(nx,1)).squeeze()
        return effects


    def forward(self, feed_dict):
        imgs_inp_fine = feed_dict['imgs_inp_fine']
        imgs_inp_coarse = feed_dict['imgs_inp_coarse']
        rois_inp = feed_dict['rois_inp']
        spatial_image = feed_dict['spatial_image']
        bs = len(imgs_inp_coarse)
        spatial_image = spatial_image.repeat(bs, 1, 1, 1)
        img_feat_coarse = self.resnet_conv_coarse.forward(imgs_inp_coarse)
        img_feat_coarse = img_feat_coarse.view(img_feat_coarse.size(0), -1)
        img_feat_fine = self.resnet_conv_fine.forward(imgs_inp_fine)

        if self.use_spatial_map:
            img_feat_fine = torch.cat([img_feat_fine, spatial_image], dim=1)

        roi_img_feat = self.roi_pool.forward(img_feat_fine, rois_inp)

        roi_img_feat = roi_img_feat.view(roi_img_feat.size(0), -1)
        roi_feat = self.roi_encoder.forward((roi_img_feat, img_feat_coarse, rois_inp, None))

        if self.pred_labels:
            labels_pred = self.label_predictor.forward(roi_feat)

        if self.filter_positives:
            pos_inds = feed_dict['roi_labels'].squeeze().data.nonzero().squeeze()
            pos_inds = torch.autograd.Variable(
                pos_inds.type(torch.LongTensor).cuda(), requires_grad=False)
            roi_feat = torch.index_select(roi_feat, 0, pos_inds)
            rois_inp = torch.index_select(rois_inp, 0, pos_inds)

        roi_feat_by_batch = self.batchify(roi_feat, rois_inp[:, 0].data)
        rois_inp_by_batch = self.batchify(rois_inp.data, rois_inp[:, 0].data)


        feat_src = []
        feat_trj = []
        paired_rois_inp = []
        
        for bx, (rois_example, example) in enumerate(zip(rois_inp_by_batch, roi_feat_by_batch)):
            for sx, (roi_src, roi_feat_src) in enumerate(zip(rois_example, example)):
               for tx, (roi_trj, roi_feat_trj) in enumerate(zip(rois_example, example)):
                    feat_src.append(roi_feat_src)
                    feat_trj.append(roi_feat_trj)
                    paired_rois_inp.append(bx)
       
        feat_src = torch.stack(feat_src)
        feat_trj = torch.stack(feat_trj)
        paired_rois_inp = torch.FloatTensor([paired_rois_inp]).squeeze().cuda()

        pairwise_effects = self.effect_encoder.forward((feat_src, feat_trj))
        pairwise_effects_by_batch = self.batchify(pairwise_effects, paired_rois_inp)
        ## Now aggregrate all the effects?
        new_roi_feat_by_batch = []
        for bx, (example_roi_feat, example_effects) in enumerate(zip(roi_feat_by_batch, pairwise_effects_by_batch)):
            nx = len(roi_feat_by_batch[bx])
            assert nx*nx == len(example_effects), 'we should have n^2 effects'
            example_effects = example_effects.view(nx, nx, -1)
            aggregrate_effect = self.aggregrate_effects(example_effects) ## N x 300
            new_feat = example_roi_feat + aggregrate_effect
            new_roi_feat_by_batch.append(new_feat)

        new_roi_feat = torch.cat(new_roi_feat_by_batch, dim=0)

        if self.opts.interaction:
            codes_pred = self.code_predictor.forward(new_roi_feat)
        else:
            codes_pred = self.code_predictor.forward(roi_feat)


        return_stuff = {}
        return_stuff['codes_pred'] = codes_pred
        if self.pred_labels:
            return_stuff['labels_pred'] = labels_pred

        return return_stuff, paired_rois_inp
