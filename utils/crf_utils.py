import torch
import numpy as np
import pdb
import scipy.stats
'''
Will work on torch tensors/ numpy tensors
This does not support batching.
'''

class CRFUnaryPotential:
    def __init__(self, unary_parameters):
        self.unary_parameters = unary_parameters
        return
    
    def __call__(self, x):
        return self.score(x)
    
    def score(self, x):
        return torch.nn.functional.mse_loss(x, self.unary_parameters)

class CRFZeroPotential:
    def __init__(self,):
        return
    def __call__(self, x):
        return self.score(x)
    def score(self, x):
        return torch.zeros(1).mean()


class CRFBinaryPotential:
    def __init__ (self, mixture_weights, mixture_means, mixture_covariances):
        if type(mixture_weights) == np.ndarray:
            mixture_weights = torch.from_numpy(mixture_weights).float()
        
        if type(mixture_means) == np.ndarray:
            mixture_means = torch.from_numpy(mixture_means).float()

        if type(mixture_covariances) == np.ndarray:
            mixture_covariances = torch.from_numpy(mixture_covariances).float()

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
        score = -1*torch.log(torch.stack(scores).sum()+1E-9) ## Negative log liklhihood.
        return score


class CRFModel(object):
    def __init__(self,):
        return
    def parameters(self,):
        raise NotImplementedError

    def forward(self,):
        raise NotImplementedError


class CRFModelTrans(CRFModel):
    def __init__(self, unaries, object_classes, class_pair_potentials, pairwise_wt=1.0, unary_wt=1.0):
        super(CRFModelTrans, self).__init__()
        self.unaries = torch.nn.Parameter(unaries) ## N x 3
        self.nobjects = len(self.unaries)
        self.score_funcs = []
        self.weights = []
        ## N*N pairwise functions
        for ox_src , obj_class_src in enumerate(object_classes):
            for ox_trg , obj_class_trg in enumerate(object_classes):
                pw_clsid = "{}_{}".format(obj_class_src, obj_class_trg)
                self.weights.append(pairwise_wt)
                if ox_src != ox_trg:
                    self.score_funcs.append(class_pair_potentials[pw_clsid])
                else:
                    self.score_funcs.append(CRFZeroPotential())

        ## N unary fuctions
        for ox_src in range(len(object_classes)):
            self.weights.append(1)
            self.score_funcs.append(CRFUnaryPotential(unaries[ox_src]))
       
        return
        
    def parameters(self,):
        return self.unaries
        
    def forward(self,):
        unaries = self.unaries
        relative_trans = unaries[:,None,:] - unaries[None,:,:]
        params = torch.cat([relative_trans.view(self.nobjects*self.nobjects, -1), self.unaries])
        scores = []

        for param, weight, score_func in zip(params, self.weights, self.score_funcs): 
            scores.append(weight*score_func(param))
        
        scores = torch.stack(scores).sum()
        return scores



class CRFOptimizer:
    def __init__(self, model, opts):
        self.model = model
        self.learning_rate = opts.learning_rate
        self.delta_reg = opts.delta_reg
        self.init_optimizers()
        return

    def init_optimizers(self,):
        self.optimizer = torch.optim.SGD([self.model.parameters()], lr=self.learning_rate)
        return

    def update_learning_rate(self, optimizer, new_lr):
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        return

    def optimize(self, learning_rate=0.0001):
        current_params = self.model.parameters().data.clone()
        delta_change = 1000
        previous_score = None
        lr = learning_rate
        prev_params = None
        iter = 0
        while(delta_change>self.delta_reg and lr > 1E-5):

            score = self.model.forward()
            self.optimizer.zero_grad()
            score.backward()
            
            if previous_score is not None:
                if previous_score <= score:
                    lr = lr/2
                    self.update_learning_rate(self.optimizer,lr)
                    continue
                else:
                    previous_score = score.data
            else:
                previous_score = score.data

            self.optimizer.step()
            iter += 1
            if prev_params is None:
                prev_params = current_params
            current_params = self.model.parameters().data.clone()
            delta_change = torch.norm((prev_params - current_params).view(-1)).sum().item()
            print(" {} : old {} , new {}, diff {}, delta_change : {}".format(iter , previous_score.item(), score.item(), score.item()-previous_score.item(), delta_change))
            
        return



def main():
    # test_potential_univariate()
    # test_potential_multivariate()
    # test_potential_gmm_sum()
    # test_optimzer()
    test_optimzer1d()
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
                            '1_2' : CRFBinaryPotential(mixture_weights, -0.9*mixture_means, mixture_covariances), 
                            '2_1' : CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances), 
                            '2_2'  : CRFBinaryPotential(mixture_weights, mixture_means, mixture_covariances),
                            }
    
    crfmodel = CRFModelTrans(unaries, ['1','2'], class_pair_potentials, pairwise_wt=0.3, unary_wt=100.)
    opts = lambda x: 0
    opts.learning_rate = 0.01
    opts.delta_reg = 1E-5
    crfoptim = CRFOptimizer(crfmodel, opts)
    crfoptim.optimize()
    print(crfmodel.parameters())
    pdb.set_trace()
    return


if __name__=="__main__":
    main()
            