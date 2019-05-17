"""Script for box3d prediction experiment.
"""
# Sample usage: python -m factored3d.experiments.suncg.box3d --plot_scalars --display_visuals --display_freq=2000 --save_epoch_freq=1 --batch_size=8  --name=box3d_base --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10

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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ....data import suncg as suncg_data
from ....utils import suncg_parse
from ....nnutils import train_utils
from ....nnutils import net_blocks
from ....nnutils import loss_utils
from ....nnutils import oc_net
from ....nnutils import disp_net
from ....utils import visutil
from ....renderer import utils as render_utils
from ....utils.visualizer import Visualizer
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.offline as offline
# import plotly.figure_factory as ff
import numpy as np
from ....utils import quatUtils
from collections import Counter
import scipy.io as sio

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')

flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'),
                    'Directory where intermittent renderings are saved')
flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')
flags.DEFINE_boolean('shape_dec_ft', False, 'If predicting voxels, should we pretrain from an existing deocder')

flags.DEFINE_string('no_graph_pretrain_name', 'box3d_base', 'Experiment name from which we will pretrain the OCNet')
flags.DEFINE_integer('no_graph_pretrain_epoch', 0, 'Network epoch from which we will finetune')

flags.DEFINE_string('ft_pretrain_name', 'box3d_base', 'Experiment name from which we will pretrain the OCNet')
flags.DEFINE_integer('ft_pretrain_epoch', 0, 'Network epoch from which we will finetune')
flags.DEFINE_string('set', 'train', 'dataset to use')
flags.DEFINE_integer('max_rois', 10, 'If we have more objects than this per image, we will subsample.')
flags.DEFINE_integer('max_total_rois', 100, 'If we have more objects than this per batch, we will reject the batch.')
flags.DEFINE_boolean('use_class_weights', False, 'Use class weights')
flags.DEFINE_float('split_size', 1.0, 'Split size of the train set')


FLAGS = flags.FLAGS


class CRFTrainer(train_utils.Trainer):
    def define_model(self):
        '''
        Define the pytorch net 'model' whose weights will be updated during training.
        '''
        self.device = "cpu"
        self.num_classes = len(suncg_parse.valid_object_classes)
        class_ids = suncg_parse.object_class2index.values()
        self.mem_relative_trans  = {src_cls: {trj_class:[] for trj_class in class_ids} for src_cls in class_ids}
        self.mem_relative_direction = {src_cls: {trj_class:[] for trj_class in range(1,self.num_classes+1)} for src_cls in class_ids}
        self.mem_relative_direction_quant = {src_cls: {trj_class:[] for trj_class in class_ids} for src_cls in class_ids}
        self.mem_relative_scale = {src_cls: {trj_class:[] for trj_class in class_ids} for src_cls in class_ids}
        self.mem_trans = {obj_class:[] for obj_class in class_ids}
        self.mem_scale = {obj_class:[] for obj_class in class_ids}
        self.mem_quat_quant = {obj_class:[] for obj_class in class_ids}
        self.mem_quat = {obj_class:[] for obj_class in class_ids}
        
        self.model = torch.nn.Linear(1,1) 
        return

    def init_dataset(self):
        opts = self.opts
        self.real_iter = 1  # number of iterations we actually updated the net for
        self.data_iter = 1  # number of iterations we called the data loader
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
        
        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)
            self.quat_medoids_var = Variable(self.quat_medoids)

        if opts.classify_dir:
            self.direction_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'direction_medoids_relative_{}_new.mat'.format(opts.nz_rel_dir)))['medoids']).type(torch.FloatTensor)
            self.direction_medoids_var = Variable(self.direction_medoids)

    def define_criterion(self):
        self.smoothed_factor_losses = {
            'shape': 0, 'scale': 0, 'quat': 0, 'trans': 0, 'rel_trans' : 0, 'rel_scale' : 0, 'rel_quat' : 0, 'class' : 0, 'var_mean':0, 'var_std':0, 'var_mean_rel_dir':0, 'var_std_rel_dir':0,
        }


    def set_input(self, batch):
        opts = self.opts
        if batch is None or not batch:
            self.invalid_batch = True
            return

        rois = suncg_parse.bboxes_to_rois(batch['bboxes'])
        self.data_iter += 1

        if rois.numel() <= 5 or rois.numel() > (5 * opts.max_total_rois) * (
                1.0 * opts.batch_size) / 8.0:  # with just one element, batch_norm will screw up
            self.invalid_batch = True
            return
        else:
            pairs = self.count_number_pairs(rois)
            if self.opts.only_pairs and pairs <= 1:
                self.invalid_batch = True
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
            input_imgs, requires_grad=False)

        self.input_imgs_fine = Variable(
            input_imgs_fine.to(self.device), requires_grad=False)

        self.rois = Variable(
            rois.type(torch.FloatTensor).to(self.device), requires_grad=False)


        code_tensors = suncg_parse.collate_codes(batch['codes'])
        code_tensors_quats = code_tensors['quat']
        object_classes = code_tensors['class'].type(torch.LongTensor)
        self.class_gt = self.object_classes = Variable(object_classes.to(self.device), requires_grad=False)
        code_tensors['shape'] = code_tensors['shape'].unsqueeze(1)  # unsqueeze voxels

        common_src_classes = []
        common_trj_classes = []
        self.common_masks = []
        for bx, ocs in enumerate(suncg_parse.batchify(object_classes, self.rois[:, 0].data.cpu())):
            nx = len(ocs)
            for ix in range(nx):
                for jx in range(nx):
                    common_src_classes.append(ocs[ix])
                    common_trj_classes.append(ocs[jx])

        common_src_classes = Variable(torch.cat(common_src_classes).to(self.device), requires_grad=False)
        common_trj_classes = Variable(torch.cat(common_trj_classes).to(self.device), requires_grad=False)
        self.common_classes = [common_src_classes, common_trj_classes]
        self.object_locations = suncg_parse.batchify(code_tensors['trans'], self.rois[:, 0].data.cpu())

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
        self.relative_dir_mask = Variable(torch.cat(relative_dir_mask,dim=0).byte()).to(self.device)
        if opts.classify_dir and not opts.gmm_dir:
            relative_direction_rotation_binned  = [suncg_parse.directions_to_bininds(t, self.direction_medoids) for t in relative_direction_rotation]
            relative_direction_rotation_directions = [suncg_parse.bininds_to_directions(t, self.direction_medoids) for t in relative_direction_rotation_binned]
            self.relative_direction_rotation_unquantized = [Variable(t).to(self.device) for t in relative_direction_rotation]
            self.relative_direction_rotation = [Variable(t).to(self.device) for t in relative_direction_rotation_binned]
        else:
            self.relative_direction_rotation = [Variable(t).to(self.device) for t in relative_direction_rotation]

        self.code_tensors_quats_non_quant = code_tensors_quats
        if opts.classify_rot and not opts.gmm_rot:
            code_tensors_quats = [suncg_parse.quats_to_bininds(code_quat, self.quat_medoids) for code_quat in code_tensors_quats]


        self.codes_gt_quats = [
            Variable(t.to(self.device), requires_grad=False) for t in code_tensors_quats]
        codes_gt_keys = ['shape', 'scale', 'trans']
        self.codes_gt  ={key : Variable(code_tensors[key].to(self.device), requires_grad=False) 
                            for key in codes_gt_keys}
        self.codes_gt['quat'] = self.codes_gt_quats
        
        self.object_locations = [Variable(t_.to(self.device), requires_grad=False) for t_ in
                                 self.object_locations]
        self.object_scales = suncg_parse.batchify(code_tensors['scale'] + 1E-10, self.rois[:, 0].data.cpu())
        self.object_scales = [Variable(t_.to(self.device), requires_grad=False) for t_ in
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
      

        self.relative_gt = {'relative_trans' : self.relative_trans_gt,
                            'relative_scale' : self.relative_scale_gt,
                            'relative_dir' : self.relative_direction_rotation,
                            'relative_mask' : self.relative_dir_mask,
                            }
        return


    def get_current_scalars(self):
        loss_dict = {'iter_frac': self.real_iter / self.data_iter}
        return loss_dict

    def render_codes(self, code_vars, prefix='mesh'):
        opts = self.opts
        code_list = suncg_parse.uncollate_codes(code_vars, self.input_imgs.data.size(0), self.rois.data.cpu()[:, 0])
        mesh_dir = osp.join(opts.rendering_dir, opts.name)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        new_codes_list = suncg_parse.convert_codes_list_to_old_format(code_list[0])
        render_utils.save_parse(mesh_file, new_codes_list, save_objectwise=False)
        png_dir = mesh_file.replace('.obj', '/')
        render_utils.render_mesh(mesh_file, png_dir)

        return scipy.misc.imread(osp.join(png_dir, prefix + '_render_000.png'))

    def get_current_visuals(self):
        visuals = {}
        opts = self.opts
        visuals['img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))
        codes_gt_vis = {k:t for k,t in self.codes_gt.items()}
        if not opts.pred_voxels:
            codes_gt_vis['shape'] = torch.nn.functional.sigmoid(
                self.voxel_decoder.forward(self.codes_gt['shape'])
            )
        if opts.classify_rot:
            codes_gt_vis['quat'] = [Variable(suncg_parse.bininds_to_quats(d.cpu().data, self.quat_medoids), requires_grad=False) for d in codes_gt_vis['quat']]        
        visuals['codes_gt'] = self.render_codes(codes_gt_vis, prefix='gt')

        return visuals

    def get_current_points(self):
        pts_dict = {}
        return pts_dict

    def forward(self):
        opts = self.opts
        feed_dict = {}
     
        self.bIndices = suncg_parse.batchify(self.rois.data[:,0], self.rois.data[:,0])
        self.bIndices_pairs = torch.cat([a.repeat(a.size(0)) for a in self.bIndices]).view(-1,1)
       
        for i in range(len(self.codes_gt['trans'])):
            t = self.codes_gt['trans'][i]
            s = self.codes_gt['scale'][i]
            q = self.codes_gt['quat'][i]
            obj_c = self.class_gt[i].item()
            self.mem_trans[obj_c].append(t)
            self.mem_quat_quant[obj_c].append(q)
            self.mem_quat[obj_c].append( self.code_tensors_quats_non_quant[i])
            self.mem_scale[obj_c].append(s)
        
        ignore_index = []
        for i in range(len(self.bIndices)):
            nobjs = len(self.bIndices[i])
            ignore_index.extend([j==k for j in range(nobjs) for k in range(nobjs)])

        for i in range(len(self.common_classes[0])):
            if ignore_index[i]:
                continue
            src_cls = self.common_classes[0][i].item()
            trg_cls = self.common_classes[1][i].item()
            bIndex = self.bIndices_pairs[i]

            rt = self.relative_gt['relative_trans'][i]
            rs = self.relative_gt['relative_scale'][i]
            rd = self.relative_direction_rotation_unquantized[i]
            rd_quant = self.relative_gt['relative_dir'][i]

            self.mem_relative_trans[src_cls][trg_cls].append(rt)
            self.mem_relative_scale[src_cls][trg_cls].append(rs)
            self.mem_relative_direction[src_cls][trg_cls].append(rd)
            self.mem_relative_direction_quant[src_cls][trg_cls].append(rd_quant)

            ## Store gt by c1,c2 here. Iteration over all the relative gts

     

    def train(self,):
        opts = self.opts
        total_steps = 0
        start_time = time.time()
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        for i, batch in enumerate(self.dataloader):
            iter_start_time = time.time()
            self.set_input(batch)
            if not self.invalid_batch:
                self.forward()
            if opts.print_scalars and (total_steps % opts.print_freq == 0):
                scalars = self.get_current_scalars()
                time_diff = time.time() - start_time
                scalars_print = {k : v for k,v in scalars.items() if v != 0.0}
                visualizer.print_current_scalars(time_diff, 0, total_steps, scalars_print)
            if total_steps == opts.num_iter:
                break
            total_steps += 1
        self.save_data_statistics(opts.name)
        return
                
    def save_data_statistics(self, network_label, file_prefix='stats'):
        save_filename = '{}_{}_small.mat'.format(network_label, file_prefix)
        save_filename = osp.join(self.opts.checkpoint_dir,  save_filename)
        data_stats = {}
        class_ids = self.mem_trans.keys()
        trans = {}
        scale = {}
        quat_quant = {}
        quat = {}
        relative_trans = {}
        relative_dir = {}
        relative_dir_quant = {}
        relative_scale = {}
        for key in class_ids:
            trans[key] = torch.stack(self.mem_trans[key])
            scale[key] = torch.stack(self.mem_scale[key])
            quat_quant[key] = torch.cat(self.mem_quat_quant[key])
            quat[key] = torch.cat(self.mem_quat_quant[key])
        
        for src_key in class_ids:
            relative_trans[src_key] = {}
            relative_scale[src_key] = {}
            relative_dir[src_key] = {}
            relative_dir_quant[src_key] = {}
            for trg_key in class_ids:
                if len(self.mem_relative_trans[src_key][trg_key]) > 1:
                    relative_trans[src_key][trg_key] = torch.stack(self.mem_relative_trans[src_key][trg_key])
                    relative_scale[src_key][trg_key] = torch.stack(self.mem_relative_scale[src_key][trg_key])
                    relative_dir[src_key][trg_key] = torch.cat(self.mem_relative_direction[src_key][trg_key])
                    relative_dir_quant[src_key][trg_key] = torch.cat(self.mem_relative_direction_quant[src_key][trg_key])
                else:
                    relative_trans[src_key][trg_key] = torch.Tensor([])
                    relative_scale[src_key][trg_key] = torch.Tensor([])
                    relative_dir[src_key][trg_key] = torch.Tensor([])
                    relative_dir_quant[src_key][trg_key] = torch.Tensor([])


        data_stats['trans']  = {'cls_{}'.format(k) : v.numpy() for k,v in trans.items()}
        data_stats['scale'] = {'cls_{}'.format(k)  : v.numpy() for k,v in scale.items()}
        data_stats['quat_medoids'] = self.quat_medoids.numpy()
        data_stats['quat_quant']= {'cls_{}'.format(k)  : v.numpy() for k,v in quat_quant.items()}
        data_stats['quat']= {'cls_{}'.format(k)  : v.numpy() for k,v in quat.items()}
        
        data_stats['relative_trans'] = {"cls_{}_{}".format(k1,k2):v2.numpy() for (k1, v1) in relative_trans.items()  for k2,v2 in v1.items()}
        data_stats['relative_scale'] = {"cls_{}_{}".format(k1,k2):v2.numpy() for k1, v1 in relative_scale.items()  for k2,v2 in v1.items() }
        data_stats['relative_direction'] = {"cls_{}_{}".format(k1,k2):v2.numpy() for k1, v1 in relative_dir.items()  for k2,v2 in v1.items()}
        data_stats['relative_direction_quant'] = {"cls_{}_{}".format(k1,k2):v2.numpy()  for k1, v1 in relative_dir_quant.items()  for k2,v2 in v1.items()}
        data_stats['relative_dir_medoids'] = self.direction_medoids.numpy()

        pdb.set_trace()
        import scipy.io as sio
        sio.savemat(save_filename, data_stats)

        return





def main(_):
    seed = 206
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    FLAGS.rendering_dir = osp.join(FLAGS.cache_dir, 'rendering')
    FLAGS.checkpoint_dir = osp.join(FLAGS.cache_dir, 'snapshots')
    if not FLAGS.classify_rot:
        FLAGS.nz_rot = 4

    if not FLAGS.classify_dir:
        FLAGS.nz_rel_dir = 3

    # FLAGS.n_data_workers = 0 # code crashes otherwise due to json not liking parallelization
    trainer = CRFTrainer(FLAGS)
    trainer.init_training()
    trainer.train()
    pdb.set_trace()


    ###

    # import scipy.io as sio
    # abs_quats = torch.cat(trainer.stored_absolute_quats, dim=0)
    # abs_quats = abs_quats.numpy()
    # sio.savemat('absolute_quats/absolute_quats.mat', {'quats': abs_quats})



    # with open('relative_quats_medium_sym_all.npy','w') as f:
    #     # pdb.set_trace()
    #     rel_quats = torch.stack(trainer.stored_relative_quats, dim=0)
    #     rel_quats = rel_quats.numpy()
    #     np.save(f, rel_quats)
    # pdb.set_trace()
    # rel_dirs = torch.stack(trainer.stored_translation_dependent_relative_angle, dim=0).numpy()
    # sio.savemat('relative_small_quats_directions2.mat', {'dirs' : rel_dirs})




if __name__ == '__main__':
    app.run(main)
