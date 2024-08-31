import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from scipy.stats import wasserstein_distance
import numpy as np
def _cdf(p, u_values, v_values, u_weights=None, v_weights=None):
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)
class WassersteinLoss(nn.Module):
    reduction: str

    def __init__(self, p = 1, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.p = p
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
    
    def forward(self, input: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        # ta = input.detach().cpu().numpy()
        # tb = target.detach().cpu().numpy()
        # wd = _cdf(1, ta[0], tb[0])
        # print(f"WD: {wd}")
        # return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)
        # normalize distribution, add 1e-14 to divisor to avoid 0/0
        # input  = input / (torch.sum(input, dim=-1, keepdim=True) + 1e-14)
        # target = target / (torch.sum(target, dim=-1, keepdim=True) + 1e-14)

        all_values = torch.cat([input, target], dim=1)
        sall_values, sidx = all_values.sort(dim=-1)
        deltas = sall_values[:, 1:] - sall_values[:, :-1]
        s_values = sall_values[:, :-1].contiguous()
        u_values, u_sorter = torch.sort(input)
        v_values, v_sorter = torch.sort(target)
        cdf_tensor_a = torch.searchsorted(u_values, s_values, right=True)
        cdf_tensor_b = torch.searchsorted(v_values, s_values, right=True)

        cdf_tensor_a = cdf_tensor_a.to(input.dtype) / input.shape[-1]
        cdf_tensor_b = cdf_tensor_b.to(input.dtype) / target.shape[-1]
        # make cdf with cumsum
        p = self.p
        # choose different formulas for different norm situations
        
        if p == 1:
            diff = torch.abs((cdf_tensor_a - cdf_tensor_b))
            # cdf_distance = torch.sum(diff, dim=-1)
            cdf_distance = torch.sum(torch.mul(diff, deltas), dim=-1)
        elif p == 2:
            diff = torch.pow((cdf_tensor_a - cdf_tensor_b),2)
            cdf_distance = torch.sqrt(torch.sum(torch.mul(diff, deltas), dim=-1))
        else:
            cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)
        if self.reduction == "mean":
            cdf_loss = cdf_distance.mean()
        elif self.reduction == "sum":
            cdf_loss = cdf_distance.sum()
        return cdf_loss

    def forward_ori(self, input: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    
        input  = input / (torch.sum(input, dim=-1, keepdim=True) + 1e-14)
        target = target / (torch.sum(target, dim=-1, keepdim=True) + 1e-14)
        # make cdf with cumsum
        cdf_tensor_a = torch.cumsum(input,  dim=-1)
        cdf_tensor_b = torch.cumsum(target, dim=-1)
        p = self.p
        # choose different formulas for different norm situations
        
        if p == 1:
            cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)),dim=-1)
        elif p == 2:
            cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b),2),dim=-1))
        else:
            cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)
        if self.reduction == "mean":
            cdf_loss = cdf_distance.mean()
        elif self.reduction == "sum":
            cdf_loss = cdf_distance.sum()

        return cdf_loss

def torch_wasserstein_loss(tensor_a: "torch.Tensor", tensor_b: "torch.Tensor"):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a, tensor_b, p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def torch_cdf_loss(tensor_a: "torch.Tensor", tensor_b : "torch.Tensor", p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss