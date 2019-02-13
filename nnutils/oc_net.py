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
from . import roi_pool_py as roi_pool
# from ..utils.roi_pooling.modules.roi_pool import RoIPool
from ..utils.roi_pooling.modules.roi_pool import _RoIPooling as RoIPool
from ..utils import suncg_parse

# from oc3d.nnutils import roi_pooling
import pdb

# -------------- flags -------------#
# ----------------------------------#
flags.DEFINE_integer('roi_size', 4, 'RoI feat spatial size.')
flags.DEFINE_integer('nz_shape', 20, 'Number of latent feat dimension for shape prediction')
flags.DEFINE_integer('nz_feat', 300, 'RoI encoded feature size')
flags.DEFINE_boolean('use_context', True, 'Should we use bbox + full image features')
flags.DEFINE_boolean('pred_voxels', True, 'Predict voxels, or code instead')
flags.DEFINE_boolean('pred_relative', False, 'Predict relative location')
flags.DEFINE_boolean('pred_class', False, 'Predict objcet class')
flags.DEFINE_boolean('common_pred_class', False, 'Predict object class for src and trj of the common')
flags.DEFINE_boolean('classify_rot', True, 'Classify rotation, or regress quaternion instead')
flags.DEFINE_boolean('classify_dir', True, 'Classify direction, or regress direction instead')
flags.DEFINE_integer('nz_rot', 24, 'Number of outputs for rot prediction. Value overriden in code.')
flags.DEFINE_integer('nz_rel_dir', 24, 'Number of outputs for relative direction prediction')
flags.DEFINE_boolean('cython_roi', True, 'Use cython roi')
flags.DEFINE_boolean('use_object_class', False, 'Use object class')
flags.DEFINE_boolean('gmm_rot', False, 'Use GMM for abs rotation')
flags.DEFINE_boolean('var_gmm_rot', False, 'predict Variance for  GMM abs rotation')
flags.DEFINE_boolean('gmm_dir', False, 'Use GMM for abs rotation')
flags.DEFINE_boolean('use_common', True, 'Use common features')
flags.DEFINE_boolean('use_mask_in_common', False, 'Add mask to identify src and trj in roi pool')
flags.DEFINE_boolean('upsample_mask', False, 'Upsample mask')
flags.DEFINE_boolean('use_spatial_map', False, 'Add spatial map after resnet')

# ------------- Modules ------------#
# ----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class ShapePredictor(nn.Module):
    def __init__(self, nz_feat, nz_shape, pred_voxels=True):
        super(ShapePredictor, self).__init__()
        self.pred_layer = nb.fc(True, nz_feat, nz_shape)
        self.pred_voxels = pred_voxels

    def forward(self, feat):
        # pdb.set_trace()
        shape = self.pred_layer.forward(feat)
        # print('shape: ( Mean = {}, Var = {} )'.format(shape.mean().data[0], shape.var().data[0]))
        if self.pred_voxels:
            shape = torch.nn.functional.sigmoid(self.decoder.forward(shape))
        return shape

    def add_voxel_decoder(self, voxel_decoder=None):
        # if self.pred_voxels:
        self.decoder = voxel_decoder


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot, classify_rot=True, var_gmm_rot=False):
        super(QuatPredictor, self).__init__()
        self.classify_rot = classify_rot
        self.var_gmm_rot = var_gmm_rot
        if self.var_gmm_rot:
            self.pred_layer = nn.Linear(nz_feat, nz_rot)
            self.var_layer = nn.Linear(nz_feat, nz_rot)
        else:
            self.pred_layer = nn.Linear(nz_feat, nz_rot)


    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.var_gmm_rot:
            log_variance = torch.nn.functional.relu(self.var_layer(feat) + 6.9) - 6.9
            # variance = torch.nn.functional.relu(torch.exp(log_variance) - 1E-5) + 1E-5
            # variance = torch.exp(log_variance)
            return (quat, log_variance)
        elif self.classify_rot:
            quat = quat
            # quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1  # biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        # print('scale: ( Mean = {}, Var = {} )'.format(scale.mean().data[0], scale.var().data[0]))
        return scale


class TransPredictor(nn.Module):
    def __init__(self, nz):
        super(TransPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        # pdb.set_trace()
        trans = self.pred_layer.forward(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class LabelPredictor(nn.Module):
    def __init__(self, nz_feat, classify_rot=True):
        super(LabelPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 1)

    def forward(self, feat):
        pred = self.pred_layer.forward(feat)
        pred = torch.nn.functional.sigmoid(pred)
        return pred


class CodePredictor(nn.Module):
    def __init__(
            self, nz_feat=200,
            pred_voxels=True, nz_shape=100,
            classify_rot=True, nz_rot=4, var_gmm_rot=False,
    ):
        super(CodePredictor, self).__init__()
        self.quat_predictor = QuatPredictor(nz_feat, classify_rot=classify_rot, nz_rot=nz_rot, var_gmm_rot = var_gmm_rot)
        self.shape_predictor = ShapePredictor(nz_feat, nz_shape=nz_shape, pred_voxels=pred_voxels)
        self.scale_predictor = ScalePredictor(nz_feat)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self, feat):
        shape_pred = self.shape_predictor.forward(feat)
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        trans_pred = self.trans_predictor.forward(feat)
        stuff = {'shape' : shape_pred,
                 'scale' : scale_pred,
                 'quat'  : quat_pred,
                 'trans' : trans_pred
                 }
        return stuff

class RoiEncoder(nn.Module):
    def __init__(self, nc_inp_fine, nc_inp_coarse, use_context=True, nz_joint=300, nz_roi=300, nz_coarse=300,
                 nz_box=50, use_object_class=False, max_object_classes=10):
        super(RoiEncoder, self).__init__()

        self.encoder_fine = nb.fc_stack(nc_inp_fine, nz_roi, 2)
        self.encoder_coarse = nb.fc_stack(nc_inp_coarse, nz_coarse, 2)
        self.encoder_bbox = nb.fc_stack(4, nz_box, 3)
        self.use_object_class = use_object_class
        if use_object_class:
            self.class_embedding = nn.Embedding(max_object_classes, nz_box)
            self.encoder_class = nb.fc_stack(nz_box, nz_box, 2)
            self.encoder_joint = nb.fc_stack(nz_roi + nz_coarse + nz_box + nz_box, nz_joint, 2)
        else:
            self.encoder_joint = nb.fc_stack(nz_roi + nz_coarse + nz_box, nz_joint, 2)

        self.use_context = use_context

    def forward(self, feats):
        roi_img_feat, img_feat_coarse, rois_inp, class_inp = feats
        feat_fine = self.encoder_fine.forward(roi_img_feat)
        feat_coarse = self.encoder_coarse.forward(img_feat_coarse)

        # pdb.set_trace()
        # dividing by img_height that the inputs are not too high
        feat_bbox = self.encoder_bbox.forward(rois_inp[:, 1:5] / 480.0)
        if not self.use_context:
            feat_bbox = feat_bbox * 0
            feat_coarse = feat_coarse * 0
        feat_coarse_rep = torch.index_select(feat_coarse, 0, rois_inp[:, 0].type(torch.LongTensor).cuda())
        if self.use_object_class:
            feat_class = self.class_embedding(class_inp).squeeze(1)
            feat_class = self.encoder_class(feat_class)
            feat_roi = self.encoder_joint.forward(torch.cat((feat_fine, feat_coarse_rep, feat_bbox, feat_class), dim=1))
        # print(feat_fine.size(), feat_coarse_rep.size(), feat_bbox.size())
        else:
            feat_roi = self.encoder_joint.forward(torch.cat((feat_fine, feat_coarse_rep, feat_bbox), dim=1))
        return feat_roi

class UpSampleMask(nn.Module):
    def __init__(self, nc_inp_channels, nc_out_channels):
        super(UpSampleMask, self).__init__()
        self.mask_conv = nb.conv2d(batch_norm=True, in_planes=nc_inp_channels, out_planes=nc_out_channels)

    def forward(self, feats):
        mask  = feats
        return self.mask_conv(mask)




class CommonRoiEncoder(nn.Module):
    def __init__(self, nc_inp_channels, nc_inp_coarse, roi_size=4, use_context=True, nz_joint=300, nz_roi=300, nz_coarse=300,
                 nz_box=50, nz_common=50,  use_object_class=False, max_object_classes=10):
        super(CommonRoiEncoder, self).__init__()
        self.fine_conv1 = nb.conv2d(batch_norm=True, in_planes=nc_inp_channels, out_planes=nc_inp_channels//2, kernel_size=3, stride=1)
        self.fine_conv2 = nb.conv2d(batch_norm=True, in_planes=nc_inp_channels//2, out_planes=nc_inp_channels//4, kernel_size=3, stride=1)
        self.encoder_fine = nb.fc_stack(nc_inp_channels//4*roi_size*roi_size, nz_roi, 2)
        self.encoder_coarse = nb.fc_stack(nc_inp_coarse, nz_coarse, 2)
        self.encoder_bbox = nb.fc_stack(4, nz_box, 3)
        self.use_object_class = use_object_class
        if use_object_class:
            self.class_embedding = nn.Embedding(max_object_classes, nz_box)
            self.encoder_class = nb.fc_stack(nz_box, nz_box, 2)
            self.encoder_joint = nb.fc_stack(nz_roi + nz_coarse + nz_box + nz_box, nz_joint, 2)
        else:
            self.encoder_joint = nb.fc_stack(nz_roi + nz_coarse + nz_box, nz_joint, 2)

        self.use_context = use_context

    def forward(self, feats):
        roi_img_feat, common_masks, img_feat_coarse, rois_inp, class_inp = feats
        roi_img_feat = torch.cat([roi_img_feat, common_masks], dim=1)
        bs = len(roi_img_feat)
        roi_img_feat = self.fine_conv1(roi_img_feat)
        roi_img_feat = self.fine_conv2(roi_img_feat)
        roi_img_feat = roi_img_feat.view(bs, -1)

        feat_fine = self.encoder_fine.forward(roi_img_feat)
        feat_coarse = self.encoder_coarse.forward(img_feat_coarse)
        # dividing by img_height that the inputs are not too high
        feat_bbox = self.encoder_bbox.forward(rois_inp[:, 1:5] / 480.0)
        if not self.use_context:
            feat_bbox = feat_bbox * 0
            feat_coarse = feat_coarse * 0
        feat_coarse_rep = torch.index_select(feat_coarse, 0, rois_inp[:, 0].type(torch.LongTensor).cuda())
        if self.use_object_class:
            feat_class = self.class_embedding(class_inp).squeeze(1)
            feat_class = self.encoder_class(feat_class)
            feat_roi = self.encoder_joint.forward(torch.cat((feat_fine, feat_coarse_rep, feat_bbox, feat_class), dim=1))
        # print(feat_fine.size(), feat_coarse_rep.size(), feat_bbox.size())
        else:
            feat_roi = self.encoder_joint.forward(torch.cat((feat_fine, feat_coarse_rep, feat_bbox), dim=1))
        return feat_roi

class ClassPredictor(nn.Module):
    def __init__(self, nz_feat, max_object_classes):
        super(ClassPredictor, self).__init__()
        self.predictor = nn.Linear(nz_feat, max_object_classes)
    def forward(self, feats):
        class_logits = self.predictor(feats)
        return torch.nn.functional.log_softmax(class_logits)


class TrajectoryEncoder(nn.Module):
    '''
    ## TODO
    1. Different Encoders for trans ( needs relative location) and scale (does not need relative location).
    '''
    def __init__(self, in_size, in_common_size, nz_size, nz_box=50, use_common=True, max_object_classes=10):
        super(TrajectoryEncoder, self).__init__()
        nz_src = nz_trj = nz_size
        rel_nz_size = nz_size
        self.encoder_source = nb.fc_stack(in_size, nz_src, 2)
        self.encoder_target = nb.fc_stack(in_size, nz_trj, 2)
        self.encoder_common = nb.fc_stack(in_size, nz_size, 2)
        joint_encoder_scale_size = nz_src + nz_trj + nz_size
        joint_encoder_input_size = nz_src + nz_trj + nz_size
        self.use_common = use_common
        self.encoder_joint = nb.fc_stack(joint_encoder_input_size, nz_size, 2)
        
    def forward(self, feats):
        source_feat, target_feat, common_feat = feats
        enc_feat_source = self.encoder_source.forward(source_feat)
        enc_feat_target = self.encoder_target.forward(target_feat)
        enc_feat_common = self.encoder_common.forward(common_feat)

        if not self.use_common:
            enc_feat_common = enc_feat_common * 0

        trj_feat = [enc_feat_source, enc_feat_target, enc_feat_common]
        trj_feat = self.encoder_joint.forward(torch.cat(trj_feat, dim=1))
        return trj_feat


class RelativeTransPredictor(nn.Module):
    def __init__(self, in_size, out_size):
        super(RelativeTransPredictor, self).__init__()
        # self.encoder = nb.fc_stack(in_size, nz_size, 2 ,use_bn=False)
        self.predictor = nn.Linear(in_size, out_size)

    def forward(self, feat):
        # feat = self.encoder.forward(feats)
        predictions = self.predictor.forward(feat)
        return predictions

class RelativeScalePredictor(nn.Module):
    def __init__(self, in_size, out_size):
        super(RelativeScalePredictor, self).__init__()
        # self.encoder = nb.fc_stack(in_size, nz_size, 2 ,use_bn=False)
        self.predictor = nn.Linear(in_size, out_size)

    def forward(self, feat):
        predictions = self.predictor.forward(feat) + 1 # biasing the scale to one.
        predictions = torch.nn.functional.relu(predictions) + 1E-12
        return predictions.log()

class RelativeQuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot, classify_rot=True):
        super(RelativeQuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot
    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            return quat
            # quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat

class RelativeDirectionPredictor(nn.Module):
    def __init__(self, nz_feat, nz_dir, classify_dir=True, gmm_dir=False):
        super(RelativeDirectionPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_dir)
        self.classify_dir = classify_dir
        self.gmm_dir = gmm_dir
        if gmm_dir:
            self.var_layer = nn.Linear(nz_feat, nz_dir)
    def forward(self, feat):
        direction = self.pred_layer.forward(feat)
        if self.gmm_dir:
            log_variance = torch.nn.functional.relu(self.var_layer(feat) + 6.9) - 6.9
            return (direction, log_variance)
        elif self.classify_dir:
            return direction
        else:
            direction = torch.nn.functional.normalize(direction, dim=1)
        return direction


# ------------- OC Net -------------#
# ----------------------------------#
class OCNet(nn.Module):
    def __init__(
            self, img_size_coarse, opts,
            roi_size=4,
            use_context=True, nz_feat=1000,
            pred_voxels=True, nz_shape=100,
            classify_rot=False, nz_rot=4,
            pred_labels=False, filter_positives=False, b_size=1, cython_roi=True,
            use_basic = False,
    ):
        super(OCNet, self).__init__()
        self.opts = opts
        self.pred_labels = pred_labels
        self.filter_positives = filter_positives
        self.nz_feat = nz_feat
        self.nz_rot = nz_rot
        self.nz_rel_dir = opts.nz_rel_dir
        self.b_size = b_size
        self.resnet_conv_fine = ResNetConv(n_blocks=3)
        self.resnet_conv_coarse = ResNetConv(n_blocks=4)
        self.roi_size = roi_size
        self.nz_feat = nz_feat
        self.nz_shape = nz_shape
        self.use_common = True
        self.pred_relative = opts.pred_relative
        self.pred_class = opts.pred_class
        self.max_object_classes = opts.max_object_classes
        self.use_spatial_map = opts.use_spatial_map
        self.upsample_mask = opts.upsample_mask
        self.common_pred_class = opts.common_pred_class
        self.var_gmm_rot = opts.var_gmm_rot
        self.gmm_dir = opts.gmm_dir
        self.not_use_basic  = not use_basic

        if cython_roi:
            self.roi_pool = RoIPool(roi_size, roi_size, 1.0 / 16)
        else:
            self.roi_pool = roi_pool.RoIPool(roi_size, roi_size, 1 / 16)

        resent_channels = 256
        img_fine_feat_channels = resent_channels
        if self.use_spatial_map and self.not_use_basic:
            img_fine_feat_channels += 2
        
        self.nc_inp_fine = nc_inp_fine = img_fine_feat_channels * roi_size * roi_size
        nc_inp_coarse = 512 * (img_size_coarse[0] // 32) * (img_size_coarse[1] // 32)

        nc_inp_common_layers = (2 + img_fine_feat_channels)

        self.roi_encoder = RoiEncoder(nc_inp_fine, nc_inp_coarse, use_context=use_context, nz_joint=nz_feat, 
            use_object_class=opts.use_object_class, max_object_classes=opts.max_object_classes)

        if self.upsample_mask and self.not_use_basic:
            upsample_channels = 64
            nc_inp_common_layers += -2 + upsample_channels
            self.mask_upsampler = UpSampleMask(2, upsample_channels)

        if opts.use_mask_in_common and self.not_use_basic:
            self.common_roi_encoder = CommonRoiEncoder(nc_inp_common_layers, nc_inp_coarse, roi_size = roi_size,  use_context=use_context, nz_joint=nz_feat,
            use_object_class=False, max_object_classes=opts.max_object_classes)

        self.code_predictor = CodePredictor(
            nz_feat=nz_feat,
            pred_voxels=pred_voxels, nz_shape=nz_shape,
            classify_rot=classify_rot, nz_rot=nz_rot,
            var_gmm_rot=self.var_gmm_rot)

        if self.pred_relative and self.not_use_basic:
            self.add_relative_pred()

        if self.pred_class and self.not_use_basic:
            self.add_class_pred()


        if self.common_pred_class and not_use_basic:
            self.add_class_pred_common()


        nb.net_init(self.roi_encoder)
        nb.net_init(self.code_predictor)
        
        if self.var_gmm_rot and not_use_basic:
            self.code_predictor.quat_predictor.var_layer.bias.data.fill_(-6.71)
            self.code_predictor.quat_predictor.var_layer.weight.data.mul_(0.1)
        if self.gmm_dir and not_use_basic:
            self.relative_direction_predictor.var_layer.bias.data.fill_(-6.71)
            self.relative_direction_predictor.var_layer.weight.data.mul_(0.1)


    def add_relative_pred(self):
        opts = self.opts
        nz_feat = self.nz_feat
        nz_shape = self.nz_shape
        self.relative_encoder = TrajectoryEncoder(nz_feat, self.nc_inp_fine, nz_feat,
         use_common=self.use_common,
         max_object_classes=self.opts.max_object_classes)
        self.relative_trans_predictor = RelativeTransPredictor(nz_feat, 3)
        self.relative_scale_predictor = RelativeScalePredictor(nz_feat, 3)
        self.relative_direction_predictor = RelativeDirectionPredictor(nz_feat, self.nz_rel_dir, classify_dir=opts.classify_dir, gmm_dir=opts.gmm_dir)
        nb.net_init(self.relative_encoder)
        nb.net_init(self.relative_trans_predictor)
        nb.net_init(self.relative_scale_predictor)
        # nb.net_init(self.relative_quat_predictor)
        nb.net_init(self.relative_direction_predictor)
        return

    def create_common_masks(self, src_rois, trj_rois, common_rois, roi_size):
        '''
        src_rois  N x 4, Variable(FloatTensor.cuda)
        trj_rois  N x 4, Variable(FloatTensor.cuda)
        common_rois N x 4, Variable(FloatTensor.cuda)
        roi_size  int
        returns : mask (N x 2 x roi_size x roi_size)
        '''

        def get_roi_center(rois):
            return torch.stack([(rois[:, 0] + rois[:, 2])/2, (rois[:, 1] + rois[:, 3])/2], dim=1)

        offsets = torch.cat([common_rois[:, 0:2], common_rois[:, 0:2]], dim=1) ## N x 4
        bs = len(src_rois)
        src_rois_center = get_roi_center(src_rois - offsets) ## N x 2
        trj_rois_center = get_roi_center(trj_rois - offsets) ## N x 2
        common_roi_size = (common_rois - offsets)[:, 2:]   ## N x 2
        src_indices_pool = src_rois_center*roi_size/common_roi_size
        trj_indices_pool = trj_rois_center*roi_size/common_roi_size
        src_indices_pool = torch.clamp(src_indices_pool, 0, roi_size -1E-4).long()
        trj_indices_pool = torch.clamp(trj_indices_pool, 0, roi_size - 1E-4).long()
        src_indices_pool = (src_indices_pool[:,0] + src_indices_pool[:,1]*roi_size).unsqueeze(-1).unsqueeze(-1)  ## (y, x indexing in the 2D mask)
        trj_indices_pool = (trj_indices_pool[:,0] + trj_indices_pool[:,1]*roi_size).unsqueeze(-1).unsqueeze(-1)  ## (y, x indexing in the 2D mask)

        mask_src = torch.zeros(bs, 1, roi_size*roi_size).cuda()
        mask_trj = torch.zeros(bs, 1, roi_size*roi_size).cuda()
        mask_src.scatter_(2, src_indices_pool.data, 1)
        mask_src = mask_src.view(bs, 1, roi_size, roi_size)
        mask_trj.scatter_(2, trj_indices_pool.data, 1)
        mask_trj = mask_trj.view(bs, 1, roi_size, roi_size)
        return Variable(torch.cat([mask_src, mask_trj], dim=1))



    def add_class_pred(self):
        self.class_predictor = ClassPredictor(self.nz_feat, self.max_object_classes)
        nb.net_init(self.class_predictor)
        return

    def add_class_pred_common(self):
        self.src_class_predictor = ClassPredictor(self.nz_feat, self.max_object_classes)
        nb.net_init(self.src_class_predictor)
        self.trj_class_predictor = ClassPredictor(self.nz_feat, self.max_object_classes)
        nb.net_init(self.trj_class_predictor)
        return

    def add_label_predictor(self):
        self.label_predictor = LabelPredictor(self.nz_feat)
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

    def forward(self, feed_dict):
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

        if self.use_spatial_map:
            img_feat_fine = torch.cat([img_feat_fine, spatial_image], dim=1)

        roi_img_feat = self.roi_pool.forward(img_feat_fine, rois_inp)

        roi_img_feat = roi_img_feat.view(roi_img_feat.size(0), -1)
        roi_feat = self.roi_encoder.forward((roi_img_feat, img_feat_coarse, rois_inp, class_inp))

        if self.pred_labels:
            labels_pred = self.label_predictor.forward(roi_feat)

        if self.filter_positives:
            pos_inds = feed_dict['roi_labels'].squeeze().data.nonzero().squeeze()
            pos_inds = torch.autograd.Variable(
                pos_inds.type(torch.LongTensor).cuda(), requires_grad=False)
            roi_feat = torch.index_select(roi_feat, 0, pos_inds)
            rois_inp = torch.index_select(rois_inp, 0, pos_inds)

        # pdb.set_trace()
        codes_pred = self.code_predictor.forward(roi_feat)
        roi_feat_by_batch = self.batchify(roi_feat, rois_inp[:, 0].data)
        rois_inp_by_batch = self.batchify(rois_inp.data, rois_inp[:, 0].data)

        paired_rois_inp = []
        src_rois_inp = []
        trj_rois_inp = []

        for rois_inp_in_image in rois_inp_by_batch:
            num_objects = rois_inp_in_image.size(0)
            paired_rois = []
            src_rois = []
            trj_rois = []
            for object_i_roi in rois_inp_in_image:
                for object_j_roi in rois_inp_in_image:
                    batch_id = object_i_roi[0:1]
                    new_roi_min = torch.min(torch.stack([object_i_roi[1:3], object_j_roi[1:3]]), dim=0)[0]
                    new_roi_max = torch.max(torch.stack([object_i_roi[3:5], object_j_roi[3:5]]), dim=0)[0]
                    new_roi = torch.cat([batch_id, new_roi_min, new_roi_max])
                    paired_rois.append(new_roi)
                    src_rois.append(object_i_roi)
                    trj_rois.append(object_j_roi)
            paired_rois = torch.stack(paired_rois)
            src_rois_inp.append(torch.stack(src_rois))
            trj_rois_inp.append(torch.stack(trj_rois))
            paired_rois_inp.append(paired_rois)

        if self.pred_class:
            class_pred = self.class_predictor.forward(roi_feat)
        
        src_rois_inp = Variable(torch.cat(src_rois_inp, dim=0))
        trj_rois_inp = Variable(torch.cat(trj_rois_inp, dim=0))
        paired_rois_inp = Variable(torch.cat(paired_rois_inp, dim=0))
        roi_common_img_feat = self.roi_pool.forward(img_feat_fine, paired_rois_inp)
        
        if self.opts.use_mask_in_common:
            common_masks = self.create_common_masks(src_rois_inp[:,1:5], trj_rois_inp[:,1:5], paired_rois_inp[:,1:5], self.roi_size)
            if self.upsample_mask:
                common_masks = self.mask_upsampler(common_masks)
            roi_common_img_feat = self.common_roi_encoder.forward((roi_common_img_feat, common_masks, img_feat_coarse, paired_rois_inp, None))
        else:
            roi_common_img_feat = roi_common_img_feat.view(paired_rois_inp.size(0), -1)
            roi_common_img_feat = self.roi_encoder.forward((roi_common_img_feat, img_feat_coarse, paired_rois_inp, None))
        
        roi_common_img_feat_by_batch = self.batchify(roi_common_img_feat, paired_rois_inp[:, 0].data)
        object_class_by_batch = self.batchify(class_inp, rois_inp[:, 0].data)

        if self.pred_relative:
            pwd_pred = []
            trajectory_pred = []
            relative_trans_pred = []
            source_x = []
            target_x = []
            common_x = []
            source_roi = []
            target_roi = []
            source_class = []
            target_class = []
            for example, common, roi_example  in zip(roi_feat_by_batch, roi_common_img_feat_by_batch, rois_inp_by_batch):
                ## features for all rois in the image. Create pair wise trajectories?
                num_objects = example.size(0)
                joint_features = common.view(num_objects, num_objects, -1)
                for ix in range(num_objects):
                    for jx in range(num_objects):
                        source_x.append(example[ix])
                        target_x.append(example[jx])
                        common_x.append(joint_features[ix][jx])
                        source_roi.append(roi_example[ix])
                        target_roi.append(roi_example[jx])
            
            source_roi = Variable(torch.stack(source_roi))
            target_roi = Variable(torch.stack(target_roi))
            
            source_center = torch.stack([(source_roi[:,3] + source_roi[:,1])/2, (source_roi[:,2] + source_roi[:,4])/2],dim=1)
            source_size = torch.stack([(source_roi[:,3] - source_roi[:,1])/2, (source_roi[:,4] - source_roi[:,2])/2],dim=1)
            
            target_center = torch.stack([(target_roi[:,1] + target_roi[:,3])/2, (target_roi[:,2] + target_roi[:,4])/2], dim=1)
            target_size = torch.stack([(target_roi[:,3] - target_roi[:,1])/2, (target_roi[:,4] - target_roi[:,2])/2],dim=1)
            

            relative_roi = (target_center - source_center)/480
            relative_size = torch.log(target_size /(source_size + 1E-3))
            relative_features = relative_roi

            source_x = torch.stack(source_x)
            target_x = torch.stack(target_x)
            common_x = torch.stack(common_x)
            relative_encodings = self.relative_encoder.forward((source_x, target_x, common_x))

            # relative_trans_pred = self.relative_trans_predictor.forward(relative_encodings)
            # relative_scale_pred = self.relative_scale_predictor.forward(relative_encodings_scale) ## This prediction is in log scale.
            # relative_quat_pred = self.relative_quat_predictor.forward(relative_encodings_scale)  ## This prediction is un-normalized logits
            
            relative_trans_pred = self.relative_trans_predictor.forward(relative_encodings)
            relative_scale_pred = self.relative_scale_predictor.forward(relative_encodings) ## This prediction is in log scale.
            
            relative_direction_pred = self.relative_direction_predictor.forward(relative_encodings)
            codes_relative  = {'relative_trans' : relative_trans_pred,
                                'relative_scale' : relative_scale_pred,
                                'relative_dir' : relative_direction_pred,
                            }

            if self.common_pred_class:
                src_class_predictions = self.src_class_predictor(roi_common_img_feat)
                trj_class_predictions = self.trj_class_predictor(roi_common_img_feat)
                common_class_predictions = [src_class_predictions, trj_class_predictions]

        return_stuff = (codes_pred,)

        return_stuff = {}
        return_stuff['codes_pred'] = codes_pred
        if self.pred_relative:
            return_stuff['codes_relative'] = codes_relative
        if self.pred_class:
            return_stuff['class_pred'] = class_pred
        if self.common_pred_class:
            return_stuff['common_class_pred'] = common_class_predictions
        if self.pred_labels:
            return_stuff['labels_pred'] = labels_pred

        return return_stuff, paired_rois_inp