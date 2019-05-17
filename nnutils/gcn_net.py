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
#
flags.DEFINE_boolean('interaction', True, 'Use GCN layers')
# ------------- Modules ------------#
# ----------------------------------#

class AffinityNet(nn.Module):
    def __init__(self,  nz_feat, nz_af ):
        super(AffinityNet, self).__init__()
        self.encoder_src  = nn.Linear(nz_feat, nz_af, bias=False)
        self.encoder_trj  = nn.Linear(nz_feat, nz_af, bias=False)

    def forward(self, feats):
        feat_src, feat_trj = feats
        feat_src = self.encoder_src(feat_src)
        feat_trj = self.encoder_trj(feat_trj)
        affinity = torch.sum(feat_src * feat_trj, dim=-1)
        return affinity


class GCN(nn.Module):
    def __init__(self, nz_feat, max_nodes, n_g_layers=2):
        super(GCN, self).__init__()
        self.max_nodes = max_nodes
        self.gcn_layers = torch.nn.ModuleList([GraphConvolution(nz_feat, nz_feat) for _ in range(n_g_layers)])
        # self.bn_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(nz_feat) for _ in range(n_g_layers)])
        self.n_g_layers = n_g_layers

    def forward(self, feats):
        roi_feat, adj_matrix = feats
        x = roi_feat
        for lx in range(self.n_g_layers):
            x = self.gcn_layers[lx].forward((adj_matrix, x))
            # x = self.bn_layers[lx].forward(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x




class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.transform_layer = nn.Linear(in_features, out_features, bias=False)

    def forward(self, feats):
        '''
        :param features:  B x N x D  ( where some rows might be zeros, if there are few elements
        :param amasks: B x N x N (A mask)
        :return:
        '''
        ## B x N x  nz_feat (maybe there has to be A_mask)
        ## A --> B x N x N
        ## Adj is all connected graph?

        # Compute the A matrix?
        adj, X = feats
        trans_features = self.transform_layer(X)  # B x N x in_Features --> B x N x out_features
        output_features = torch.bmm(adj, trans_features)  # B x N x out_features
        return output_features


# ------------- OC Net -------------#
# ----------------------------------#
class GCNNet(nn.Module):
    def __init__(
            self, img_size_coarse, opts,
            roi_size=4,
            use_context=True, nz_feat=1000,
            pred_voxels=True, nz_shape=100,
            classify_rot=False, nz_rot=4, pred_labels=False,
            filter_positives=False, b_size=1,
            use_basic = False,
    ):
        super(GCNNet, self).__init__()
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
        self.max_nodes = opts.max_rois

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

        self.code_predictor = cb.CodePredictor(
            nz_feat=nz_feat,
            pred_voxels=pred_voxels, nz_shape=nz_shape,
            classify_rot=classify_rot, nz_rot=nz_rot,
            var_gmm_rot=self.var_gmm_rot)

        self.affinity_predictor = AffinityNet(nz_feat, nz_feat)
        self.gcn = GCN(nz_feat, self.max_nodes,)

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
        # location_inp = feed_dict['location_inp']
        # class_inp = feed_dict['class_inp']
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


    def forward(self, feed_dict):
        imgs_inp_fine = feed_dict['imgs_inp_fine']
        imgs_inp_coarse = feed_dict['imgs_inp_coarse']
        rois_inp = feed_dict['rois_inp']
        # location_inp = feed_dict['location_inp']
        # class_inp = feed_dict['class_inp']
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

        # pdb.set_trace()
        
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

        
        pairwise_affinity = self.affinity_predictor.forward((feat_src, feat_trj))
        pairwise_affinity_batched = self.batchify(pairwise_affinity, paired_rois_inp)
        roi_feat_by_batch_padded = []
        ## Normalize the affinity
        pairwise_normalized_affinity_batched = []
        adj_mask = []
        
        for bx, (affinity, roi_feat_example) in enumerate(zip(pairwise_affinity_batched, roi_feat_by_batch)):
            nx = len(roi_feat_example)
            affinity = affinity.view(nx, nx)
            affinity = torch.nn.functional.softmax(affinity).unsqueeze(0).unsqueeze(0)
            adj_mask.append(torch.FloatTensor([1 if i < nx else 0 for i in range(self.max_nodes)]))
            affinity = torch.nn.functional.pad(affinity, (0, self.max_nodes - nx, 0, self.max_nodes -nx), value=0)
            affinity = affinity.squeeze(0).squeeze(0)
            pairwise_normalized_affinity_batched.append(affinity)
           
            roi_feat_example = torch.nn.functional.pad(roi_feat_example.unsqueeze(0).unsqueeze(0),
                                                        (0, 0, 0, self.max_nodes - nx), value=0)

            roi_feat_example = roi_feat_example.squeeze(0).squeeze(0)
            
            roi_feat_by_batch_padded.append(roi_feat_example)

        affinity_adjacency = torch.stack(pairwise_normalized_affinity_batched)
        adj_mask = torch.stack(adj_mask).type(torch.LongTensor).view(-1)
        valid_indices = torch.nonzero(adj_mask).squeeze().cuda()
        roi_feat_padded  = torch.stack(roi_feat_by_batch_padded)
        roi_feat_context = self.gcn.forward((roi_feat_padded, affinity_adjacency))
        roi_feat_context = roi_feat_context.view(-1, roi_feat_context.size(-1))
        roi_feat_context = roi_feat_context[valid_indices]
        if self.opts.interaction:
            codes_pred = self.code_predictor.forward(roi_feat_context)
        else:
            codes_pred = self.code_predictor.forward(roi_feat)

        return_stuff = {}
        return_stuff['codes_pred'] = codes_pred
        if self.pred_labels:
            return_stuff['labels_pred'] = labels_pred

        return return_stuff
