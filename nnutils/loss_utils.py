'''
Loss building blocks.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from absl import flags
from ..utils import suncg_parse
# from ..utils import quatUtils
import numpy as np
import pdb
# -------------- flags -------------#
# ----------------------------------#
flags.DEFINE_float('shape_loss_wt', 1, 'Shape loss weight.')
flags.DEFINE_float('scale_loss_wt', 1, 'Scale loss weight.')
flags.DEFINE_float('quat_loss_wt', 1, 'Quat loss weight.')
flags.DEFINE_float('trans_loss_wt', 1, 'Trans loss weight.')
flags.DEFINE_float('delta_trans_loss_wt', 1, 'Delta Trans loss weight.')
flags.DEFINE_float('rel_trans_loss_wt', 1,
                   'Relative location loss weight.')
flags.DEFINE_float('rel_quat_loss_wt', 1,
                   'Relative location loss weight.')
flags.DEFINE_boolean('rel_opt', False,
                   'rel optim to locations')
flags.DEFINE_integer('auto_rel_opt', -1,
                   'rel optim to locations and scale after half epochs')
flags.DEFINE_boolean('train_var', False,
                   'Train variance for the GMM')
kernel = Variable(torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]])).cuda()

# kernel = Variable(
#     torch.FloatTensor([[[0.00, 0.0, 0.0, 1, 0.0, 0.00, 0.00]]])).cuda()


class LeastSquareOpt:
    def __init__(self):
        self.Adict = {}
        self.lmbda = 1

    def get_matrix_Ab(self, n_objects, trans_location, relative_locations):
        lmbda = self.lmbda
        b = []
        # b = torch.cat([relative_locations, trans_location], dim=0)

        for i in range(n_objects):
                for j in range(n_objects):
                    if i == j:
                        continue
                    b.append(relative_locations[i*n_objects + j])
        for j in range(n_objects):
            b.append(trans_location[j])
        b = torch.stack(b)

        if n_objects in self.Adict.keys():
            return self.Adict[n_objects], b
        else:
            A = np.zeros((n_objects*n_objects, n_objects))
            index = 0
            for i in range(n_objects):
                for j in range(n_objects):
                    if i == j:
                        continue
                    A[index][i] = -1
                    A[index][j] = 1
                    index += 1
            for i in range(n_objects):
                A[index][i] = lmbda*1
                index += 1
            AtAInvAt = np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)), A.transpose())
            self.Adict[n_objects] = Variable(torch.from_numpy(AtAInvAt).float().cuda())
            return self.Adict[n_objects], b

def normalize_probs(probs):
    return probs/torch.sum(probs, dim=-1, keepdim=True,)


def quat_nll_loss_or(quat_pred, quat_gt):
    loss = []
    quat_gt_tensor = quat_gt.data.cpu()
    quat_pred = quat_pred.exp() + 1E-5
    for i in range(len(quat_gt)):
        mask = torch.zeros(len(quat_pred)) + 1
        mask = mask.scatter_(0, quat_gt_tensor, 0)
        mask.scatter_(0, quat_gt[i].data.cpu(), 1)
        mask = Variable(mask.cuda())
        quat_probs = normalize_probs(quat_pred*mask) + 1E-5
        gt_probs = Variable(torch.zeros(len(quat_pred)).scatter_(0, quat_gt[i].data.cpu(), 1)).cuda()
        loss.append(-1*quat_probs[quat_gt[i]].log())

    loss = -1*torch.nn.functional.max_pool1d(-1*torch.stack(loss).view(1,1, -1),  kernel_size=len(quat_gt))
    return loss

def quat_nll_loss_and(quat_pred, quat_gt, class_weights=None):
    loss = []
    quat_gt_tensor = quat_gt.data.cpu()
    # quat_pred = quat_pred.exp() + 1E-5
    gts = 0*quat_pred.data
    gts.scatter_(0, quat_gt.data, 1.0/len(quat_gt))
    if class_weights is None:
        loss = -1*(Variable(gts)* quat_pred ).sum()
    else:
        loss = -1*(Variable(gts)* quat_pred*class_weights).sum()

    return loss


def quat_loss(q1, q2, average=True):
    '''
    Anti-podal squared L2 loss.

    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : scalar
    '''

    # return  quat_loss_geo(q1,q2, average)

    q_diff_loss = (q1 - q2).pow(2).sum(1)
    q_sum_loss = (q1 + q2).pow(2).sum(1)
    q_loss, _ = torch.stack((q_diff_loss, q_sum_loss), dim=1).min(1)
    if average:
        return q_loss.mean()
    else:
        return q_loss


def dir_loss1(q1, q2, average=True):
    loss = torch.acos(torch.clamp(torch.sum(q1*q2, dim=1), -1 + 1E-5, 1 - 1E-5))

    if average:
        loss = loss.mean()
        # if loss.data[0] < 1E-2:
        #     pdb.set_trace()
        #     print(loss.data[0])
        return loss
    else:
        return loss

def dir_loss2(q1, q2, average=True):
    return torch.sum((q1 - q2)**2, 1).mean()

def dir_loss(q1, q2, average=True):
    dot_p = 1 - torch.sum(q1*q2, dim=1)

    if average:
        loss = dot_p.mean()
        # if loss.data[0] < 1E-2:
        #     pdb.set_trace()
        #     print(loss.data[0])
        return loss
    else:
        return dot_p

    # dot_p = torch.sum(q1 *q2, dim=1)
    # if average:
    #     return dot_p.mean()
    # else:
    #     return dot_p

def nll_loss_with_mask(log_probs, gts, mask):
    '''
    Mask some of the examples
    '''
    bins  = log_probs.size(-1)
    mask_expanded =  mask.unsqueeze(1).expand(log_probs.size())
    log_probs_sel = torch.masked_select(log_probs, mask_expanded)
    log_probs_sel = log_probs_sel.view(-1, bins)
    gts_sel = torch.masked_select(gts, mask)
    loss = Variable(torch.Tensor([0]).type_as(log_probs_sel.data))
    if len(gts_sel) > 0:
        loss = torch.nn.functional.nll_loss(log_probs_sel, gts_sel)
    return loss

def quat_dist(q1, q2):
    '''
    N x M x 4, N x M x 4
    '''
    return torch.acos(torch.clamp(2*(q1*q2).sum(-1).pow(2) -1, -1, 1))

def dir_dist(q1, q2):
    '''
    N x M x 4, N x M x 4
    '''
    dot = (q1*q2).sum(-1)
    return torch.acos(torch.clamp(dot, -1, 1))


def code_loss(
    code_pred, code_gt, rois,
    relative_pred, relative_gt, bIndices_pairs,
    class_pred, class_gt,
    quat_medoids = None, direction_medoids=None,
    pred_class=False, pred_voxels=True, classify_rot=True, classify_dir=True, classify_trj=True, pred_relative=False, 
    shape_wt=1.0, scale_wt=1.0, quat_wt=1.0, trans_wt=1.0, rel_trans_wt=1.0, rel_quat_wt=1.0, class_wt=1.0,
    lsopt=None, rel_opt=False, class_weights=None, opts=None):
    ''' 
    Code loss

    Args:
        code_pred: [shape, scale, quat, trans, delta_trans]
        code_gt: [shape, scale, quat, trans]
        trajectories_pred : [ n x n x 10 x 3]

    Returns:
        total_loss : scalar
    '''
    if opts is None:
        gmm_rot = False
        var_gmm_rot = False
        train_var = False
        gmm_dir = False
    else:
        gmm_rot = False
        var_gmm_rot = opts.var_gmm_rot
        train_var = opts.train_var
        gmm_dir = opts.gmm_dir

    if pred_voxels:
        s_loss = torch.nn.functional.binary_cross_entropy(code_pred['shape'],
                                                          code_gt['shape'])
    else:
        # print('Shape gt/pred mean : {}, {}'.format(code_pred[0].mean().data[0], code_gt[0].mean().data[0]))
        s_loss = (code_pred['shape'] - code_gt['shape']).pow(2).mean()
    
    if classify_rot and not gmm_rot:
        q_loss = []
        for code_pred_quat, code_gt_quat in zip(code_pred['quat'], code_gt['quat']):
            q_loss.append(quat_nll_loss_or(code_pred_quat, code_gt_quat))
        q_loss = torch.stack(q_loss).mean()
        # pdb.set_trace()
        # q_loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(code_pred[2]), code_gt[2])
        # assert torch.abs(q_loss - q_loss2).data.cpu().sum() < 1E-4, 'Something incorrect in computation {} , {}'.format(q_loss.data[0], q_loss2.data[0])
    elif var_gmm_rot:
        assert quat_medoids is not None, 'Quat medoids not passed, cannot compute'
        expected_log_var = math.log((2*3.14/180)**2)
        one_by_sqrt_2pi_log = math.log(float(np.sqrt(1.0/(2*np.pi))))
        
        mixture_weights, log_variances = code_pred['quat']
        mixture_weights= torch.nn.functional.log_softmax(mixture_weights).exp()
        # pdb.set_trace()
        nll = []
        if not train_var:
            log_variances = log_variances * 1 + 0*expected_log_var

        # pdb.set_trace()
        for mixture_weight, log_variance, code_gt_quat in zip(mixture_weights, log_variances, code_gt['quat']):
            qd = quat_dist(code_gt_quat.unsqueeze(1) , quat_medoids.unsqueeze(0)).pow(2)
            mixture_weight = mixture_weight.unsqueeze(0).expand(qd.size())
            log_variance = log_variance.unsqueeze(0).expand(qd.size())
            per_mixture_prob =  one_by_sqrt_2pi_log - 0.5*log_variance  - qd/(1E-8 + 2*log_variance.exp())
            log_prob = torch.log((mixture_weight*per_mixture_prob.exp()).sum(-1) + 1E-6)
            log_prob = log_prob.mean()
            nll.append(-1*log_prob)
        q_loss = torch.cat(nll).mean()

    elif gmm_rot:
        assert quat_medoids is not None, 'Quat medoids not passed, cannot compute'
        sigmasq = (2*3.14/180)**2
        one_by_sqrt_2pi_sigmasq = float(np.sqrt(1.0/(2*np.pi*(sigmasq))))
        log_one_by_sqrt_2pi_sigmasq = float(np.log(np.sqrt(1.0/(2*np.pi*(sigmasq)))))
        # for code_pred_quat, code_gt_quat
        mixture_weights = torch.nn.functional.log_softmax(code_pred['quat']).exp()
        nll = []
        for mixture_weight, code_gt_quat in zip(mixture_weights, code_gt['quat']):
            qd = quat_dist(code_gt_quat.unsqueeze(1) , quat_medoids.unsqueeze(0)).pow(2)
            mixture_weight = mixture_weight.unsqueeze(0).expand(qd.size())
            per_mixture_prob = one_by_sqrt_2pi_sigmasq*mixture_weight*torch.exp(- qd/(1E-8  + 2*sigmasq))
            log_prob = torch.log(per_mixture_prob.sum(-1) + 1E-6)
            log_prob = log_prob.mean()
            nll.append(-1*log_prob)
        q_loss = torch.cat(nll).mean()
        # pdb.set_trace()
    else:
        q_loss = quat_loss(code_pred['quat'], code_gt['quat'])

    class_loss = 0*q_loss
    if pred_class:
        class_loss = torch.nn.functional.nll_loss(class_pred, class_gt)


    sc_loss = (code_pred['scale'].log() - code_gt['scale'].log()).abs().mean()
    tr_loss = (code_pred['trans'] - code_gt['trans']).pow(2).mean()


    ## batchify data and compute losses for examples:
    if rel_opt:
        ## For translations.
        trans_locations_batched = suncg_parse.batchify(code_pred['trans'], rois[:,0].data)
        relative_trans_pred_batched = suncg_parse.batchify(relative_pred['relative_trans'], bIndices_pairs)
        relative_trans_gt_batched = suncg_parse.batchify(relative_gt['relative_trans'], bIndices_pairs)
        new_locs = []
        # for locations, relative_locations in zip(trans_locations_batched, relative_trans_gt_batched):
        for locations, relative_locations in zip(trans_locations_batched, relative_trans_pred_batched):
            A, b = lsopt.get_matrix_Ab(len(locations), locations, relative_locations)
            locations = torch.matmul(A, b)
            new_locs.append(locations)
        new_locs = torch.cat(new_locs, dim=0)
        tr_loss += (new_locs - code_gt['trans']).pow(2).mean()

        ## For scales.
        scales_batched = suncg_parse.batchify(code_pred['scale'].log(), rois[:,0].data)
        relative_scales_pred_batched = suncg_parse.batchify(relative_pred['relative_scale'], bIndices_pairs)
        relative_scales_gt_batched = suncg_parse.batchify(relative_gt['relative_scale'], bIndices_pairs)
        new_scales = []
        for scales, relative_scales in zip(scales_batched, relative_scales_pred_batched):
            A, b = lsopt.get_matrix_Ab(len(scales), scales, relative_scales)
            scales = torch.matmul(A, b)
            new_scales.append(scales)
        new_scales = torch.cat(new_scales, dim=0)
        # pdb.set_trace()
        sc_loss += (new_scales - code_gt['scale'].log()).abs().mean()


    # tr_loss = torch.nn.functional.smooth_l1_loss(code_pred[3],  code_gt[3])
    rel_trans_loss = torch.zeros(1).cuda().mean()
    rel_scale_loss = torch.zeros(1).cuda().mean()
    rel_q_loss = torch.zeros(1).cuda().mean()

    if pred_relative:
        rel_trans_loss = (relative_pred['relative_trans'] - relative_gt['relative_trans']).pow(2).mean()
        rel_scale_loss = (relative_pred['relative_scale'] - relative_gt['relative_scale']).abs().mean()
        mask = relative_gt['relative_mask']
        if classify_dir and not gmm_dir:
            relative_dir_gt = relative_gt['relative_dir']
            relative_pred_sel = torch.masked_select(relative_pred['relative_dir'], mask.unsqueeze(1).expand(relative_pred['relative_dir'].size()))
            relative_pred_sel = relative_pred_sel.view(-1, relative_pred['relative_dir'].size(1))
            # pdb.set_trace()
            relative_gt_sel = [relative_dir_gt[i] for i, m in enumerate(mask) if m.item() == 1]
            # if len(relative_pred_sel) > 0: 
            rel_q_loss = []
            for pred_dir, gt_dir in zip(relative_pred_sel, relative_gt_sel):
                rel_q_loss.append(quat_nll_loss_or(pred_dir, gt_dir))
            rel_q_loss = torch.stack(rel_q_loss).mean()

            # relative_gt_sel = torch.masked_select(torch.cat(relative_dir_gt), mask)
            # if len(relative_pred_sel) > 0:
            #     rel_q_loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(relative_pred_sel), relative_gt_sel)

        elif gmm_dir:
            assert direction_medoids is not None, 'Quat medoids not passed, cannot compute'
            expected_log_var = math.log((2*3.14/180)**2)
            one_by_sqrt_2pi_log = math.log(float(np.sqrt(1.0/(2*np.pi))))
            mixture_weights, rel_dir_log_variances = relative_pred['relative_dir']
            size_rel = mixture_weights.size()

            mixture_weights = torch.masked_select(mixture_weights, mask.unsqueeze(1).expand(size_rel))
            mixture_weights = mixture_weights.view(-1, size_rel[1])

            rel_dir_log_variances = torch.masked_select(rel_dir_log_variances, mask.unsqueeze(1).expand(size_rel))
            rel_dir_log_variances = rel_dir_log_variances.view(-1, size_rel[1])

            mixture_weights= torch.nn.functional.log_softmax(mixture_weights).exp()
            trans_rotation = self.relative_gt['relative_dir']
            trans_rotation = [trans_rotation[i] for i, m in enumerate(mask) if m.data[0] == 1]
            # rel_dir_log_variances = 0*rel_dir_log_variances + 1*expected_log_var
            rel_dir_log_variances = 1*rel_dir_log_variances + 0*expected_log_var
            nll = []
            for mixture_weight, log_variance, gt_trans_rot in zip(mixture_weights, rel_dir_log_variances, trans_rotation):
                qd = dir_dist(gt_trans_rot.unsqueeze(1) , direction_medoids.unsqueeze(0)).pow(2)
                mixture_weight = mixture_weight.unsqueeze(0).expand(qd.size())
                log_variance = log_variance.unsqueeze(0).expand(qd.size())
                per_mixture_prob =  one_by_sqrt_2pi_log - 0.5*log_variance  - qd/(1E-8 + 2*log_variance.exp())
                log_prob = torch.log((mixture_weight*per_mixture_prob.exp()).sum(-1) + 1E-6)
                log_prob = log_prob.mean()
                nll.append(-1*log_prob)
            rel_q_loss = torch.cat(nll).mean()
        
        else:
            relative_pred_sel = torch.masked_select(relative_pred['relative_dir'], mask.unsqueeze(1).expand(relative_pred[3].size()))
            relative_pred_sel = relative_pred_sel.view(-1, relative_pred['relative_dir'].size(1))
            # relative_pred_sel = torch.nn.functional.normalize(relative_pred_sel)
            # pdb.set_trace()
            relative_gt_sel = [relative_gt['relative_dir'][ix] for i, m in enumerate(mask) if m.data[0] == 1]
            # if len(relative_pred_sel) > 0:
            # trans_rotation = torch.cat(trans_rotation)
            # relative_gt_sel = torch.masked_select(trans_rotation, mask.unsqueeze(1).expand(trans_rotation.size()))
            # relative_gt_sel = relative_gt_sel.view(-1, trans_rotation.size(1))
            if len(relative_pred_sel) > 0:
                rel_q_loss = dir_loss1(relative_pred_sel, relative_gt_sel, average=True)
                # rel_q_loss = dir_loss2(relative_pred_sel, relative_gt_sel, average=True)
                
                # pdb.set_trace()
                # rel_q_loss = rel_q_loss.mean()

    total_loss = sc_loss * scale_wt
    total_loss += q_loss * quat_wt
    total_loss += tr_loss * trans_wt
    total_loss += s_loss * shape_wt

    if pred_relative:
        total_loss += rel_trans_loss * rel_trans_wt
        total_loss += rel_scale_loss * rel_trans_wt
        total_loss += rel_q_loss

    total_loss += class_loss * class_wt
    
    loss_factors = {
        'shape': s_loss * shape_wt, 'scale': sc_loss * scale_wt,
        'quat': q_loss * quat_wt,
        'trans': tr_loss * trans_wt,
        'rel_trans' : rel_trans_loss * rel_trans_wt,
        'rel_scale' : rel_scale_loss * rel_trans_wt,
        'rel_quat' : rel_q_loss * rel_quat_wt,
        'class' : class_loss * class_wt,
        'var_mean' : s_loss * 0,
        'var_std' : s_loss * 0,
        'var_mean_rel_dir' : s_loss *0,
        'var_std_rel_dir' : s_loss * 0,
    }
    if var_gmm_rot:
        loss_factors['var_mean'] = log_variances.mean()
        loss_factors['var_std'] = log_variances.std()
    if gmm_dir:
        loss_factors['var_mean_rel_dir'] = rel_dir_log_variances.mean()
        loss_factors['var_std_rel_dir'] = rel_dir_log_variances.std()

    return total_loss, loss_factors