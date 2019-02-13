import numpy as np
import os
import ntpath
import time
import visdom
from . import visutil as util
from . import html
from .logger import Logger
import os
from datetime import datetime
import os.path as osp
import pdb
import torch

class TBVisualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.log_dir = osp.join(opt.cache_dir, 'logs', opt.name)
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)

        log_name = datetime.now().strftime('%H_%M_%d_%m_%Y')
        print("Logging to {}".format(log_name))
        self.display_id = opt.display_id
        self.use_html = opt.is_train and opt.use_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.viz = Logger(self.log_dir,opt.name)
        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, global_step):
        if self.display_id > 0: # show images in the browser
            idx = 1
            for label, image_numpy in visuals.items():
                self.viz.add_image(label,image_numpy, global_step)
                # self.viz.image(
                #     image_numpy.transpose([2,0,1]), opts=dict(title=label),
                #     win=self.display_id + idx)

    def log_trj(self, global_step, trj_pred, trj_gt, trj_mask, classify_trj=False):
        # tps = []
        # tgts = []
        # for tp, tgt,tm  in zip(trj_pred, trj_gt, trj_mask):
        #     tps.append(torch.exp(tp).view(-1))
        #     tgts.append((tgt).view(-1))
        #
        # tps = torch.cat(tps).data
        # tgts = torch.cat(tgts).data
        # self.viz.histo_summary('trajectory/preds', tps.cpu().numpy().reshape((-1)), global_step)
        # self.viz.histo_summary('trajectory/gts', tgts.cpu().numpy().reshape((-1)), global_step)
        if classify_trj:
            self.hist_summary_trj(global_step, 'trajectory/preds_probs', trj_pred, trj_mask, use_exp=classify_trj)
        else:
            self.hist_summary_trj(global_step, 'trajectory/preds', trj_pred, trj_mask, use_exp=classify_trj)
        self.hist_summary_trj(global_step, 'trajectory/gts', trj_gt, trj_mask)
        return

    def hist_summary_list(self, global_step, tag, data_list):
        t = []
        for l in data_list:
            t.append(l.view(-1))
        t = torch.cat(t)
        self.viz.histo_summary(tag, t.cpu().numpy().reshape(-1), global_step)

    def log_pwd(self, global_step, pwd_pred, pwd_gt):
        self.hist_summary_list(global_step, 'pwd/pred', pwd_pred)
        self.hist_summary_list(global_step, 'pwd/gt', pwd_gt)
        return

    def log_histogram(self, global_step, log_dict):
        for tag, value in log_dict.items():
            self.viz.histo_summary(tag, value.data.cpu().numpy(), global_step)
        return

    def log_trj_max_indices(self, global_step, trj_pred, trj_mask, classify_trj=False):
        ts = []
        for t, tm in zip(trj_pred, trj_mask):
            n_elements = torch.sum(tm).data[0]
            if n_elements > 0:
                indices = torch.nonzero(torch.ge(tm.view(-1), 0.5).data).view(-1)
                # pdb.set_trace()
                t_selected = t.view(-1, t.size(-1))
                t_selected = t_selected[indices]
                t_selected = t_selected.max(dim=-1)[1]
                ts.append(t_selected.view(-1))

        if len(ts) > 0:
            ts = torch.cat(ts)
            ts = ts.data
            self.viz.histo_summary('trajectory/preds', ts.cpu().numpy().reshape((-1)), global_step)
        return


    def hist_summary_trj(self, global_step, name, trj, trj_mask, use_exp=False):
        ts = []

        for t, tm  in zip(trj, trj_mask):
            n_elements = torch.sum(tm).data[0]
            if n_elements > 0:
                indices = torch.nonzero(torch.ge(tm.view(-1), 0.5).data).view(-1)
                t_selected = t.view(-1)[indices]
                if use_exp:
                    ts.append(torch.exp(t_selected).view(-1))
                else:
                    ts.append(t_selected.view(-1))

        if len(ts) > 0:
            ts = torch.cat(ts)
            ts = ts.data
            self.viz.histo_summary(name, ts.cpu().numpy().reshape((-1)), global_step)
        return


    def log_adj_matrix(self, global_step, adj_matrix, adj_mask, non_masked):
        self.viz.histo_summary('adjacency/values', adj_matrix.cpu().numpy().reshape((-1)), global_step)
        self.viz.histo_summary('adjacency/mask', adj_mask.cpu().numpy().reshape((-1)), global_step)
        self.viz.histo_summary('adjacency/non_masked_values', non_masked.cpu().numpy().reshape((-1)), global_step)




    # scalars: dictionary of scalar labels and values
    # def plot_current_scalars(self, epoch, counter_ratio, opt, scalars):
    #
    #
    #
    #
    #     if not hasattr(self, 'plot_data'):
    #         self.plot_data = {'X':[],'Y':[], 'legend':list(scalars.keys())}
    #     self.plot_data['X'].append(epoch + counter_ratio)
    #     self.plot_data['Y'].append([scalars[k] for k in self.plot_data['legend']])
    #     self.vis.line(
    #         X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
    #         Y=np.array(self.plot_data['Y']),
    #         opts={
    #             'title': self.name + ' loss over time',
    #             'legend': self.plot_data['legend'],
    #             'xlabel': 'epoch',
    #             'ylabel': 'loss'},
    #         win=self.display_id)

    def plot_current_scalars(self,global_step, opt, scalars):
        for key, value in scalars.items():
            self.viz.scalar_summary(key, value, global_step)

    # scatter plots
    def plot_current_points(self, points, disp_offset=10):
        idx = disp_offset
        for label, pts in points.items():
            #image_numpy = np.flipud(image_numpy)
            self.vis.scatter(
                pts, opts=dict(title=label, markersize=1), win=self.display_id + idx)
            idx += 1

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars, start_time=None):
        if start_time is None:
            message = '(time : %.3f, epoch: %d, iters: %d) ' % (epoch, i)
        else:
            time_diff = (time.time() - start_time)
            message = '(time : %.2f, epoch: %d, iters: %d) ' % (time_diff, epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
