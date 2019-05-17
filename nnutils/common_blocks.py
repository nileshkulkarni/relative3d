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
from ..utils.roi_pooling.modules.roi_pool import _RoIPooling as RoIPool

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

class ClassPredictor(nn.Module):
    def __init__(self, nz_feat, max_object_classes):
        super(ClassPredictor, self).__init__()
        self.predictor = nn.Linear(nz_feat, max_object_classes)
    def forward(self, feats):
        class_logits = self.predictor(feats)
        return torch.nn.functional.log_softmax(class_logits)


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