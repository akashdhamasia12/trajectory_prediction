from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from sklearn.metrics import auc

def accuracy_vs_uncertainty(ade, n_log_likelihood,
                            optimal_threshold_ade, optimal_threshold_uncertainty):
    # number of samples accurate and certain
    n_lc = torch.zeros(1, device=ade.device)
    # number of samples inaccurate and certain
    n_hc = torch.zeros(1, device=ade.device)
    # number of samples accurate and uncertain
    n_lu = torch.zeros(1, device=ade.device)
    # number of samples inaccurate and uncertain
    n_hu = torch.zeros(1, device=ade.device)

    avu = torch.ones(1, device=ade.device)
    avu.requires_grad_(True)

    for i in range(len(ade)):
        if (ade[i].item() <= optimal_threshold_ade
                and n_log_likelihood[i].item() <= optimal_threshold_uncertainty):
            """ accurate and certain """
            n_lc += 1
        elif (ade[i].item() <= optimal_threshold_ade
                and n_log_likelihood[i].item() > optimal_threshold_uncertainty):
            """ accurate and uncertain """
            n_lu += 1
        elif (ade[i].item() > optimal_threshold_ade
                and n_log_likelihood[i].item() <= optimal_threshold_uncertainty):
            """ inaccurate and certain """
            n_hc += 1
        elif (ade[i].item() > optimal_threshold_ade
                and n_log_likelihood[i].item() > optimal_threshold_uncertainty):
            """ inaccurate and uncertain """
            n_hu += 1

    print('n_lc: ', n_lc, ' ; n_lu: ', n_lu, ' ; n_hc: ', n_hc, ' ;n_hu: ',
            n_hu)
    avu = (n_lc + n_hu) / (n_lc + n_lu + n_hc + n_hu)

    return avu
