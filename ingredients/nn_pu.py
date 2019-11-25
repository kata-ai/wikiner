"""
TODO port https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py to pytorch
"""
# import chainer.functions as F
# import numpy
# from chainer import cuda, function, Variable
# from torch import functional as F
# from chainer.utils import type_check
import torch
import torch.nn.functional as F

from torch.autograd import Function

class PULoss(Function):
    """wrapper of loss function for PU learning"""
    def __init__(self, prior: float, loss: F = F.sigmoid, 
                 gamma: float = 1, beta: float = 0, nnPU=True):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.x_out = None

    # def check_type_forward(self, in_types):
    #     type_check.expect(in_types.size() == 2)

    #     x_type, t_type = in_types
    #     type_check.expect(
    #         x_type.dtype == numpy.float32,
    #         t_type.dtype == numpy.int32,
    #         t_type.ndim == 1,
    #         x_type.shape[0] == t_type.shape[0],
    #     )

    def forward(self, inputs):
        # xp = cuda.get_array_module(*inputs)
        x, t = inputs
        t = t[:, None]
        positive, unlabeled = t == self.positive, t == self.unlabeled
        n_positive, n_unlabeled = max([1., torch.sum(positive)]), max([1., torch.sum(unlabeled)])
        # x_in = Variable(x)
        x_in = x
        y_positive = self.loss_func(x_in)
        y_unlabeled = self.loss_func(-x_in)
        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)
        objective = positive_risk + negative_risk
        if self.nnPU:
            if negative_risk.data < -self.beta:
                objective = positive_risk - self.beta
                self.x_out = -self.gamma * negative_risk
            else:
                self.x_out = objective
        else:
            self.x_out = objective
        loss = torch.array(objective.data, dtype=self.x_out.data.dtype)
        return loss

    def backward(self, inputs, gy):
        self.x_out.backward()
        x_in, _ = inputs
        gx = gy[0].reshape(gy[0].shape + (1,) * (x_in.data.ndim - 1)) * x_in.grad
        return gx, None


def pu_loss(x, t, prior, loss=(lambda x: F.sigmoid(-x)), nnPU=True):
    """wrapper of loss function for non-negative/unbiased PU learning
        .. math::
            \\begin{array}{lc}
            L_[\\pi E_1[l(f(x))]+\\max(E_X[l(-f(x))]-\\pi E_1[l(-f(x))], \\beta) & {\\rm if nnPU learning}\\\\
            L_[\\pi E_1[l(f(x))]+E_X[l(-f(x))]-\\pi E_1[l(-f(x))] & {\\rm otherwise}
            \\end{array}
    Args:
        x (~torch.autograd.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, 1).
        t (~torch.autograd.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, ).
        prior (float): Constant variable for class prior.
        loss (~torch.nn.function): loss function.
            The loss function should be non-increasing.
        nnPU (bool): Whether use non-negative PU learning or unbiased PU learning.
            In default setting, non-negative PU learning will be used.
    Returns:
        ~torch.autograd.Variable: A variable object holding a scalar array of the
            PU loss.
    See:
        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
        "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
        Advances in neural information processing systems. 2017.
        du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama.
        "Convex formulation for learning from positive and unlabeled data."
        Proceedings of The 32nd International Conference on Machine Learning. 2015.
    """
    return PULoss(prior=prior, loss=loss, nnPU=nnPU)(x, t)
