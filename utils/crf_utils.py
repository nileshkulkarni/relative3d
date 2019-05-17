from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
import pdb
import scipy.stats
from . import transformations
from . import suncg_parse


'''
Input is numpy array. Gives a binray potential function.

'''
def convert_params_to_binary_potential(weights, means, variance):
    ncomponents = len(weights)
    KDim = means.shape[1]
    covariance_matrix = [variance[n] * torch.eye(KDim).float() for n in range(ncomponents)]
    covariance_matrix = torch.stack(covariance_matrix).float()
    means = torch.from_numpy(means).float()
    weights = torch.from_numpy(weights).float()
    cbp = CRFBinaryPotential(weights, means, covariance_matrix)
    return cbp

'''
Will work on torch tensors
This does not support batching.
'''
class CRFUnaryPotential:
    def __init__(self, unary_parameters):
        self.unary_parameters = unary_parameters.clone()
        if type(self.unary_parameters) == np.ndarray:
            self.unary_parameters = torch.from_numpy(unary_parameters).float()

        return
 
    def __call__(self, x):
        return self.score(x)
    
    def score(self, x):
        # return torch.nn.functional.mse_loss(x, self.unary_parameters)
        return torch.nn.functional.smooth_l1_loss(x, self.unary_parameters)

class CRFZeroPotential:
    def __init__(self,):
        return
    def __call__(self, x):
        return self.score(x)
    def score(self, x):
        return torch.zeros(1).mean() + 1000

class CRFBinaryPotential:
    def __init__ (self, mixture_weights, mixture_means, mixture_covariances):
        if type(mixture_weights) == np.ndarray:
            mixture_weights = torch.from_numpy(mixture_weights).float()
        
        if type(mixture_means) == np.ndarray:
            mixture_means = torch.from_numpy(mixture_means).float()

        if type(mixture_covariances) == np.ndarray:
            mixture_covariances = torch.from_numpy(mixture_covariances).float()

        # variance = torch.diag(mixture_covariances)

        # print(variance.view(-1).max())

        self.mixture_weights = mixture_weights
        self.mixture_means = mixture_means
        self.mixture_covariances = mixture_covariances
        self.reg = 1E-9*0
        self.delta_reg = 1E-6

        Kdim = len(mixture_means[0])
        self.normalization = np.sqrt(np.power(2*np.pi, len(mixture_means[0])))
        self.binary_score_functions  = [self.gaussian_mixture_score(w, mu, covar) for w, mu, covar in  zip(self.mixture_weights, self.mixture_means, self.mixture_covariances)]
    

    def gaussian_mixture_score(self, weight, mean, covar):
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, covar)
        def score_func(x):
            x = dist.log_prob(x).exp()
            return weight*x
        return score_func

    def __call__(self, x):
        return self.score(x)
    
    def score(self, x):
        x = x.float()
        ncomponents = len(self.mixture_weights)
        scores = []
        for i in range(ncomponents):
            scores.append(self.binary_score_functions[i](x))
        
        # liklhihood = torch.stack(scores).sum()
        scores = torch.stack(scores)
        # pdb.set_trace()
        distance = torch.norm(x).detach()
        score = scores.sum()
        # liklhihood = torch.max(liklhihood, 0.01*torch.ones([1]).mean().type(liklhihood.type()))
        # pdb.set_trace()
        # liklhihood = liklhihood * torch.exp(-1*torch.norm(x)).data
        score = -1*torch.log(score+1E-12) ## Negative log liklhihood.
        # pdb.set_trace()
        
        # score = score * torch.exp(-1*torch.norm(x)).data
        # score = score * (distance < 5).float()
        return score


class CRFModel(object):
    def __init__(self,):
        return
    def parameters(self,):
        raise NotImplementedError

    def forward(self,):
        raise NotImplementedError


class CRFModelTrans(CRFModel):
    
    def __init__(self, unaries, object_classes, class_pair_potentials, pairwise_wt=1.0, unary_wt=1.0, use_max_pool=False):
        super(CRFModelTrans, self).__init__()
        self.unaries = torch.nn.Parameter(unaries) ## N x 3
        self.nobjects = len(self.unaries)
        self.score_funcs = []
        self.weights = []
        self.use_max_pool = use_max_pool 
        if type(object_classes) == torch.Tensor:
            object_classes = [x.item() for x in object_classes]
        
        ## N*N pairwise functions
        for ox_src , obj_class_src in enumerate(object_classes):
            for ox_trg , obj_class_trg in enumerate(object_classes):
                pw_clsid = "cls_{}_{}".format(obj_class_src, obj_class_trg)
                self.weights.append(pairwise_wt)
                if ox_src != ox_trg:
                    self.score_funcs.append(class_pair_potentials[pw_clsid])
                else:
                    self.score_funcs.append(CRFZeroPotential())

        ## N unary fuctions
        # pbd.set_trace()
        for ox_src in range(len(object_classes)):
            self.weights.append(unary_wt)
            self.score_funcs.append(CRFUnaryPotential(unaries[ox_src].data.clone()))
       
        return
        
    def parameters(self,):
        return self.unaries

    def update_parameters(self, new_params):
        self.unaries = torch.nn.Parameter(new_params.clone())
        return
        
    def forward(self,):
        unaries = self.unaries
        relative_trans = unaries[None,:,:] - unaries[:,None,:]
        params = torch.cat([relative_trans.view(self.nobjects*self.nobjects, -1), unaries])
        scores = []

        nobjects = self.nobjects
        pairwise_funcs = self.score_funcs[0:nobjects*nobjects]
        weights  = self.weights
        score_funcs = self.score_funcs
        for i in range(nobjects):
            objectwise_potentials = []
            for j in range(nobjects):
                index = i*nobjects + j
                objectwise_potentials.append(weights[index]*self.score_funcs[index](params[index]))

            if self.use_max_pool: ## Max over all 
                if len(objectwise_potentials) > 0:
                    # pdb.set_trace()
                    min_potential = torch.stack(objectwise_potentials).min()
                    scores.append(min_potential)
            else:
                scores.extend(objectwise_potentials)

        for index in range(nobjects*nobjects, (nobjects+1)*nobjects):
            scores.append(weights[index]*score_funcs[index](params[index]))

        # for param, weight, score_func in zip(params, self.weights, self.score_funcs): 
        #     scores.append(weight*score_func(param))

        scores = torch.stack(scores).sum()
        return scores



class CRFModelTransDebug(CRFModel):
    
    def __init__(self, unaries, object_classes, relative_translations, pairwise_wt=1.0, unary_wt=1.0):
        super(CRFModelTransDebug, self).__init__()
        self.unaries = torch.nn.Parameter(unaries) ## N x 3
        self.nobjects = len(self.unaries)
        self.score_funcs = []
        self.weights = []
        if type(object_classes) == torch.Tensor:
            object_classes = [x.item() for x in object_classes]
        
        weights = torch.FloatTensor([1.0])
        covariance_matrix = torch.FloatTensor([[0.01]])*torch.eye(3)
        self.relative_translations = relative_translations
        for rt in relative_translations:
            self.weights.append(pairwise_wt)
            gmm = CRFBinaryPotential(weights, [rt], [covariance_matrix])
            self.score_funcs.append(gmm)

        ## N*N pairwise functions
        # for ox_src , obj_class_src in enumerate(object_classes):
        #     for ox_trg , obj_class_trg in enumerate(object_classes):
        #         pw_clsid = "cls_{}_{}".format(obj_class_src, obj_class_trg)
        #         self.weights.append(pairwise_wt)
        #         if ox_src != ox_trg:
        #             self.score_funcs.append(class_pair_potentials[pw_clsid])
        #         else:
        #             self.score_funcs.append(CRFZeroPotential())

        ## N unary fuctions
        # pbd.set_trace()
        for ox_src in range(len(object_classes)):
            self.weights.append(unary_wt)
            self.score_funcs.append(CRFUnaryPotential(unaries[ox_src].data.clone()))
       
        return
        
    def parameters(self,):
        return self.unaries

    def update_parameters(self, new_params):
        self.unaries = torch.nn.Parameter(new_params.clone())
        return
        
    def forward(self,):
        unaries = self.unaries
        relative_trans = unaries[None,:,:] - unaries[:,None,:]
        params = torch.cat([relative_trans.view(self.nobjects*self.nobjects, -1), unaries])
        scores = []
        for param, weight, score_func in zip(params, self.weights, self.score_funcs):
            scores.append(weight*score_func(param))
        scores = torch.stack(scores).sum()
        return scores

class CRFOptimizer:
    def __init__(self, model, opts, verbose=False, max_iter=100):
        self.model = model
        self.learning_rate = opts.learning_rate
        self.delta_reg = opts.delta_reg
        self.init_optimizers(self.learning_rate)
        self.verbose = verbose
        self.max_iter = max_iter
        return

    def init_optimizers(self, learning_rate):
        # self.optimizer = torch.optim.Adam([self.model.parameters()], lr=learning_rate, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD([self.model.parameters()], lr=learning_rate,)
        self.optimizer = torch.optim.LBFGS([self.model.parameters()], lr=learning_rate,)
        return

    def update_learning_rate(self, optimizer, new_lr, old_params):
        self.model.update_parameters(old_params)
        self.init_optimizers(new_lr)
        return

    # def optimize(self,):
    #     current_params = self.model.parameters().data.clone()
    #     delta_change = 1000
    #     previous_score = None
    #     lr = self.learning_rate
    #     prev_params = None
    #     iter = 0

    #     while(delta_change>self.delta_reg and lr > 1E-4 and iter < self.max_iter):
            
    #         score = self.model.forward()
    #         self.optimizer.zero_grad()
    #         score.backward()
    #         if previous_score is not None:
    #             if previous_score < score:
    #                 lr = lr/2
    #                 self.update_learning_rate(self.optimizer,lr, prev_params)
    #                 # pdb.set_trace()
    #                 continue
    #             else:
    #                 previous_score = score.data
    #         else:
    #             previous_score = score.data
    #         iter += 1 
    #         prev_params = current_params.data.clone()
    #         self.optimizer.step()
            
    #         # pdb.set_trace()
    #         delta_change = torch.norm((prev_params - current_params).view(-1)).item()
    #         current_params = self.model.parameters().data.clone()
    #         delta_change  = 100
    #         if self.verbose:
    #             print(" {} : lr {}, old {} , new {}, diff {}, delta_change : {}".format(iter , lr,  previous_score.item(), score.item(), score.item()-previous_score.item(), delta_change))
            
    #     return

    def optimize(self,):
        current_params = self.model.parameters().data.clone()
        delta_change = 1000
        previous_score = None
        lr = self.learning_rate
        prev_params = None
        iter = 0
        lr = self.learning_rate
        def closure():
            score = self.model.forward()
            self.optimizer.zero_grad()
            score.backward()
            return score
        old_params = self.model.parameters().data.clone()
        for lr in lr * .5**np.arange(5):
            self.optimizer.step(closure)
            current_params = self.model.parameters()
            if np.any(np.isnan(current_params.data.numpy())):
                self.model.update_parameters(old_params)
            old_params = self.model.parameters().data.clone()
        return


class RelativeRotationOptimizer:
    def __init__(self, absolute_rotation_medoids, relative_dir_medoids, lambda_weight, class_pair_potentials ):
        
        self.quat_medoids = absolute_rotation_medoids.numpy()
        self.direction_medoids = relative_dir_medoids.numpy()
        self.lambda_weight = 10*lambda_weight
        self.class_pair_potentials = class_pair_potentials

    def get_relative_dir(self, object_classes):
        if type(object_classes) == torch.Tensor:
            object_classes = [x.item() for x in object_classes]
        
        ## N*N pairwise functions
        relative_dir = []
        
        for ox_src , obj_class_src in enumerate(object_classes):
            for ox_trg , obj_class_trg in enumerate(object_classes):
                pw_clsid = "cls_{}_{}".format(obj_class_src, obj_class_trg)
                relative_dir.append(self.class_pair_potentials[pw_clsid])

        relative_dir = torch.stack(relative_dir)
        return relative_dir
    '''
    absolute_locations: numpy_array
    unary_rotation_log_probs : numpy_array

    '''
    def forward(self, absolute_locations, unary_rotation_log_probs, relative_dir):

        
        n_objects = len(unary_rotation_log_probs)
        relative_dir = relative_dir.reshape(n_objects, n_objects, -1)
        
        n_absoulte_bins = unary_rotation_log_probs.shape[1]
        n_relative_bins = relative_dir.shape[2]
        
        bin_scores = np.zeros((n_objects, n_objects, n_absoulte_bins))
        
        quat_medoids = self.quat_medoids
        direction_medoids = self.direction_medoids
        new_probability = unary_rotation_log_probs*1

        lambda_weight = self.lambda_weight * 1./n_objects
        adaptive_weight = np.ones(n_objects)

        
        for nx in range(n_objects):
            ignore_bin_scores = False
            for mx in range(n_objects):
                if mx == nx:
                    continue
                expected_direction = absolute_locations[mx] - absolute_locations[nx] ## make it unit norm
                dist = (1E-5 + np.linalg.norm(expected_direction))
                if dist > 4:
                    continue

                expected_direction = expected_direction/ (1E-5 + np.linalg.norm(expected_direction))
                expected_direction = expected_direction.reshape(1, -1)
                alignment_scores = []
                indices = []
                entropy = -1*np.sum(np.exp(relative_dir[nx, mx]) * relative_dir[nx, mx])
                for abinx in range(n_absoulte_bins):
                    prob_bin = unary_rotation_log_probs[nx][abinx]
                    quaternion_abinx = quat_medoids[abinx]
                    rotation = transformations.quaternion_matrix(quaternion_abinx)
                    transform = rotation.copy()
                    transform[0:3, 3] = np.array(absolute_locations[nx], copy=True)
                    
                    relative_direction = direction_medoids
                    predicted_direction = suncg_parse.transform_coordinates(transform, relative_direction) - absolute_locations[nx].reshape(1, -1)

                    alignment_score = (1 - np.matmul(expected_direction, predicted_direction.transpose()).squeeze())
                    index = np.argmin(alignment_score, axis=0)

                    # alignment_score = np.min(alignment_score, axis=0) + relative_dir[nx, mx, index]# absolute_log_probabilites[nx][abinx]
                    alignment_score = (1-alignment_score)*np.exp(relative_dir[nx, mx,:])# absolute_log_probabilites[nx][abinx]
                    # alignment_score = np.min(relative_dir[nx, mx, index])
                    alignment_score = np.sum(alignment_score)
                    alignment_scores.append(alignment_score)

                # alignment_scores = np.exp(np.array(alignment_scores))
                # alignment_scores = np.log(alignment_scores/np.sum(alignment_scores) + 1E-10)
                bin_scores[nx,mx,:] = alignment_scores
       
        bin_scores = np.sum(bin_scores, axis=1) ## N x 24
        bin_scores = np.exp(bin_scores) ## 
        bin_scores = np.log(1E-10 + bin_scores/np.sum(bin_scores, 1, keepdims=True))
        new_probability = 1.0 * new_probability  + np.minimum(lambda_weight, 1.0)*(1*bin_scores) 
        new_probability = torch.from_numpy(new_probability).float()
        new_probability = torch.nn.functional.normalize(new_probability.exp(),1)
        return  new_probability.cuda()


def main():
    # test_potential_univariate()
    # test_potential_multivariate()
    # test_potential_gmm_sum()
    # test_optimzer1d()
    # test_optimzer3d()
    test_mixture_gradients()
    return


def test_potential_univariate():
    mixture_means = torch.from_numpy(np.array([[0.]])).float() ## Idealy a N x K array. N --> Num of mixture componenets. K is feature dimension
    mixture_weights = torch.from_numpy(np.array([[1.]])).float()
    mixture_covariances = torch.from_numpy(np.array([[[1.0]]])).float()
    CPO = CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mixture_means[0], covariance_matrix=mixture_covariances[0])
    x = torch.zeros(1)    
    score = CPO.score(x)
    assert (score.item() - 0.3989) < 1E-4
    return

def test_potential_multivariate():
    mixture_means = torch.from_numpy(np.array([[0.,0.,0.]])).float() ## Idealy a N x K array. N --> Num of mixture componenets. K is feature dimension
    mixture_weights = torch.from_numpy(np.array([[1.]])).float()
    # mixture_variances = torch.eye(mixture_means.shape[1]) * mixture_variances[0,0]
    mixture_covariances = torch.stack([1*torch.eye(mixture_means.shape[1])])

    dist = torch.distributions.multivariate_normal.MultivariateNormal(mixture_means[0], covariance_matrix=mixture_covariances[0])
    CPO = CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances)
    x = torch.zeros(3)    
    score = CPO.score(x)
    assert (score.item() - -1*dist.log_prob(x).item()) < 1E-4
    x = torch.zeros(3) + 0.5
    score = CPO.score(x)
    assert (score.item() - -1*dist.log_prob(x).item()) < 1E-4
    return


def test_potential_gmm_sum():
   
    ncomponents = 10

    mixture = generate_random_mixture(ncomponents)
    mixture_means = mixture['mixture_means']
    mixture_weights = mixture['mixture_weights']
    mixture_covariances = mixture['mixture_covariances']

    dists = []
    for nx in range(ncomponents):
        dists.append(torch.distributions.multivariate_normal.MultivariateNormal(mixture_means[nx], covariance_matrix=mixture_covariances[nx]))
    CPO = CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances)
    
    points = torch.zeros(100,3)
    for x in points:
        pytscore = []
        for w,dist in zip(mixture_weights, dists):
            pytscore.append(w*dist.log_prob(x).exp())

        pytscore = torch.stack(pytscore).sum().log()
        score = CPO.score(x)
        assert (score.item() - -1*pytscore.item()) < 1E-4
    return

def generate_random_mixture(ncomponents, kdim=3):
    mixture_means = torch.randn(ncomponents,kdim).float()
    mixture_weights = (torch.randn(ncomponents,1).float())**2
    mixture_weights = mixture_weights/mixture_weights.sum()
    mixture_covariances = [var*torch.eye(kdim,kdim).float() for var in torch.randn(ncomponents,1)**2]
    return {'mixture_means' : mixture_means, 'mixture_weights': mixture_weights, 'mixture_covariances' :mixture_covariances}

def generate_fixed_mixture_3d():
    mixture_means = torch.FloatTensor([[0.0,0.0,0.0], [1.0,1.0,1.0]])
    mixture_weights = torch.FloatTensor([0.5, 0.5])
    mixture_weights = mixture_weights/mixture_weights.sum()
    mixture_covariances = [0.25*torch.eye(3,3).float() for _ in range(2)]    
    return {'mixture_means' : mixture_means, 'mixture_weights': mixture_weights, 'mixture_covariances' :mixture_covariances}




# def generate_fixed_mixture_1d():
    
#     mixture_means = torch.FloatTensor([[0.0,], [1.0]])
#     mixture_weights = torch.FloatTensor([0.5, 0.5])
#     mixture_weights = mixture_weights/mixture_weights.sum()
#     mixture_covariances = [0.25*torch.eye(1,1).float() for _ in range(2)]
    
#     return {'mixture_means' : mixture_means, 'mixture_weights': mixture_weights, 'mixture_covariances' :mixture_covariances}

def generate_fixed_mixture_1d():
    
    mixture_means = torch.FloatTensor([ [1.0]])
    mixture_weights = torch.FloatTensor([1.0])
    mixture_weights = mixture_weights/mixture_weights.sum()
    mixture_covariances = [0.25*torch.eye(1,1).float() for _ in range(1)]
    
    return {'mixture_means' : mixture_means, 'mixture_weights': mixture_weights, 'mixture_covariances' :mixture_covariances}


def test_optimzer1d():
    mixture = generate_fixed_mixture_1d()
    mixture_means = mixture['mixture_means']
    mixture_weights = mixture['mixture_weights']
    mixture_covariances = mixture['mixture_covariances']
    unaries = torch.FloatTensor([[0.2], [0.9]])
    class_pair_potentials = {
                            '1_1'  : CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances),
                            '1_2' : CRFBinaryPotential(mixture_weights, 1*mixture_means, mixture_covariances), 
                            '2_1' : CRFBinaryPotential(mixture_weights, -1*mixture_means, mixture_covariances), 
                            '2_2'  : CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances),
                            }
    
    crfmodel = CRFModelTrans(unaries, ['1','2'], class_pair_potentials, pairwise_wt=0.3, unary_wt=100.)
    opts = lambda x: 0
    opts.learning_rate = 0.01
    opts.delta_reg = 1E-5
    crfoptim = CRFOptimizer(crfmodel, opts)
    crfoptim.optimize()
    print(crfmodel.parameters())
    return


def test_mixture_gradients():
    param = torch.nn.Parameter(torch.zeros(3).normal_())
    # gmm = CRFBinaryPotential(torch.FloatTensor([0.5, 0.5]), torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), [torch.eye(3)*0.1, torch.eye(3)*0.1])
    gmm = CRFBinaryPotential(torch.FloatTensor([1.0]), torch.FloatTensor([[0.0, 0.0, 0.0]]), [torch.eye(3)*0.1])
    optimizer =  torch.optim.SGD([param], lr=0.1)
    # optimizer =  torch.optim.Adam([param], lr=0.000001, betas=(0.9, 0.999))
    for i in range(100):
        score = gmm(param)
        print(param)
        optimizer.zero_grad()
        score.backward()
        optimizer.step()
    print(param.data)
    assert torch.norm(param.data).item() < 1E-5


def test_optimzer3d():
    mixture = generate_fixed_mixture_3d()
    mixture_means = mixture['mixture_means']
    mixture_weights = mixture['mixture_weights']
    mixture_covariances = mixture['mixture_covariances']
    unaries = torch.FloatTensor([[0.2,0.2, 0.2], [0.9, 0.9, 0.9]])
    class_pair_potentials = {
                            'cls_1_1'  : CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances),
                            'cls_1_2' : CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances), 
                            'cls_2_1' : CRFBinaryPotential(mixture_weights, -1*mixture_means, mixture_covariances), 
                            'cls_2_2'  : CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances),
                            }
    
    crfmodel = CRFModelTrans(unaries, ['1','2'], class_pair_potentials, pairwise_wt=0.3, unary_wt=100.)
    opts = lambda x: 0
    opts.learning_rate = 0.001
    opts.delta_reg = 1E-5
    print('initial {}'.format(crfmodel.parameters()))
    crfoptim = CRFOptimizer(crfmodel, opts)
    crfoptim.optimize()
    print('final {}'.format(crfmodel.parameters()))
    return


if __name__=="__main__":
    main()
            