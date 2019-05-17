from __future__ import division
import pdb
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os

def fitGMMmodel(data, components, filename, plot=False):

    max_d = np.max(data,axis=0)
    min_d = np.min(data, axis=0)
    clf = mixture.GaussianMixture(n_components=components, covariance_type='spherical')
    clf.fit(data)


    if plot:
        x = np.linspace(min_d[0], max_d[0], num=100)
        y = np.linspace(min_d[1], max_d[1], num=100)
        z = np.linspace(min_d[2], max_d[2], num=100)
        Blues = plt.get_cmap('Blues')
        X, Y, Z = np.meshgrid(x,y,z)
        XX = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
        scores = -clf.score_samples(XX)
        # scores = np.log(1 + scores - np.min(scores))
        # scores = scores/np.max(scores)
        scores = scores.reshape(X.shape[0], Y.shape[0], Z.shape[0])
        X, Z = np.meshgrid(x,z)
        plt.close()
        CS = plt.contour(X,Z, scores.mean(1), cmap="autumn_r", norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
        plt.xlabel('X')
        plt.ylabel('Z')
        # plt.colorbar()
        plt.title('Negative log-likelihood predicted by a GMM')
        plt.axis('tight')
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        #plt.zlabel('Z')
        # fig.show()
        plt.savefig('{}.png'.format(filename))
        plt.close()
    
    return clf



def plot_histogram(data, ax, title):
    ax.hist(data, bins=50)
    ax.set_title(title)
    ax.set_xlim(0, 5)
    return

class GaussianMixture:

    def __init__(self,gmm):
        self.means = gmm.means_
        self.weights = gmm.weights_
        self.covariance_type = gmm.covariance_type
        self.covariance = gmm.covariances_
        self.n_components = gmm.n_components

    def get_gmm_object(self, ):
        data = {}
        data['means'] = self.means
        data['weights'] = self.weights
        data['covariance'] = self.covariance
        # data['covariance_type'] = self.covariance_type
        # data['n_components'] = gmm.n_components
        return data



def process_data_stats():
    np.random.seed(0)
    snapshot_dir = '/home/nileshk/Research3/3dRelnet/relative3d/nnutils/../cachedir/snapshots/box3d_base_crf_potentials/'
    data = sio.loadmat(osp.join(snapshot_dir, 'stats.mat'))
    images_dir = 'plots'
    if not osp.exists(images_dir):
        os.makedirs(images_dir)

    # fig, ax = plt.subplots(3, 6, tight_layout=True, figsize=(35,15))
    # pdb.set_trace()
    # for index, clsid in enumerate(data['scale'].dtype.names):
    #     plot_histogram(data['scale'][clsid][0,0][:,0], ax[0][index], '{}_x'.format(clsid))
    #     plot_histogram(data['scale'][clsid][0,0][:,1], ax[1][index], '{}_y'.format(clsid))
    #     plot_histogram(data['scale'][clsid][0,0][:,2], ax[2][index], '{}_z'.format(clsid))
    
    # plt.savefig(osp.join(images_dir, 'scale_variations.png'))

    keys = ['relative_trans', 'relative_scale']
    relative_models = {}
    for key in keys:
        relative_models[key] = {}
        for clsid in data[key].dtype.names:
            if len(data[key][clsid][0,0]) > 1:
                relative_models[key][clsid] = fitGMMmodel(data[key][clsid][0,0], min(10, len(data[key][clsid][0,0])), osp.join(images_dir, 'rt_{}_{}'.format(key, clsid)))
                relative_models[key][clsid] = GaussianMixture(relative_models[key][clsid]).get_gmm_object()
            else:
                relative_models[key][clsid] = None


    
    keys = ['relative_direction_quant']
    num_rel_dir_classes = len(data['relative_dir_medoids'])
    rel_direction_pdf = {}
    for key in keys:
        for clsid in data[key].dtype.names:
            uniq, count = np.unique(data[key][clsid][0][0], return_counts=True)
            if len(uniq) == 0:
                pdb.set_trace()
            rel_direction_pdf[clsid] = np.zeros(num_rel_dir_classes)
            rel_direction_pdf[clsid][uniq] = count
            rel_direction_pdf[clsid] /= np.sum(rel_direction_pdf[clsid])

    relative_models['relative_direction_quant'] = rel_direction_pdf

    unary_models = {}
    for key in ['trans', 'scale', 'quat']:
        unary_models[key] = {}
        for clsid in data[key].dtype.names:
            # pdb.set_trace()
            if len(data[key][clsid][0,0]) > 0:
                unary_models[key][clsid] = {'mean' :np.mean(data[key][clsid][0,0], axis=0), 'median' : np.median(data[key][clsid][0,0],axis=0)}

   
    key = 'quat_quant'
    unary_models[key] = {}
    for clsid in data[key].dtype.names:
        # pdb.set_trace()
        unary_models[key][clsid] = np.zeros(24)*1.0
        if len(data[key][clsid][0,0]) > 0:
            for d in data[key][clsid][0,0].squeeze():
                unary_models[key][clsid][d] +=1
            unary_models[key][clsid] /= len(data[key][clsid][0,0][0])

    np.save(osp.join(snapshot_dir,'abs_parameters'), unary_models)



    np.save(osp.join(snapshot_dir,'parameters_10_with_rot'), relative_models) 
    relative_models_loaded = np.load(osp.join(snapshot_dir, 'parameters_10_with_rot') + '.npy').item()
    # pdb.set_trace()



if __name__=="__main__":
    process_data_stats()
