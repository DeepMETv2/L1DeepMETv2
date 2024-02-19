"""Defines the neural network, loss function and metrics"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from model.dynamic_reduction_network import DynamicReductionNetwork
from model.graph_met_network import GraphMETNetwork

'''

Change from DeepMETv2
1. Add loss_fn_response for response-tuned MET

'''

class Net(nn.Module):
    def __init__(self, continuous_dim, categorical_dim, norm):
        super(Net, self).__init__()
        self.graphnet = GraphMETNetwork(continuous_dim, categorical_dim, norm,
                                        output_dim=1, hidden_dim=32,
                                        conv_depth=2)
    
    def forward(self, x_cont, x_cat, edge_index, batch):
        weights = self.graphnet(x_cont, x_cat, edge_index, batch)
        return torch.sigmoid(weights)

# tensor operations
def getdot(vx, vy):
    return torch.einsum('bi,bi->b',vx,vy)

def getscale(vx):
    return torch.sqrt(getdot(vx,vx))

def scalermul(a,v):
    return torch.einsum('b,bi->bi',a,v)

# loss function without response tune option
def loss_fn(weights, particles_vis, genMET, batch):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0]
    true_py = (-1)*genMET[:,1]

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean()

    return loss


# loss function with response tune
def loss_fn_response_tune(weights, particles_vis, genMET, batch, c = 5000):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0]
    true_py = (-1)*genMET[:,1]

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean() 

    # response correction
    v_true = torch.stack((true_px,true_py),dim=1)
    v_regressed = torch.stack((uTx, uTy),dim=1)
        
    # response = getdot( v_true, v_regressed ) / getdot( v_true, v_true )
    response = getscale(v_regressed) / getscale(v_true)

    #pT_thres = 0.         # calculate response only taking into account for events with genMET above threshold
    pT_thres = 50.
    resp_pos = torch.logical_and(response > 1., getscale(v_true) > pT_thres)
    resp_neg = torch.logical_and(response < 1., getscale(v_true) > pT_thres)
    
    response_term = c * (torch.sum(1 - response[resp_neg]) + torch.sum(response[resp_pos] - 1))

    loss += response_term

    return loss

# calculate performance metrics
def metric(weights, particles_vis, genMET, batch):
    # qT is the genMET
    qTx = genMET[:,0]
    qTy = genMET[:,1]

    v_qT=torch.stack((qTx,qTy),dim=1)

    # momentum of visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]

    # regressed uT: momentum of the system of all visible particles
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    # regressed MET
    METx = (-1) * uTx
    METy = (-1) * uTy

    v_MET=torch.stack((METx, METy),dim=1)

    # PUPPI MET using PUPPI weights
    wgt_puppi = particles_vis[:,5]

    puppiMETx = scatter_add(wgt_puppi*px, batch)
    puppiMETy = scatter_add(wgt_puppi*px, batch)
   
    v_puppiMET = torch.stack((puppiMETx, puppiMETy),dim=1)

    def compute(vector):
        response = getdot(vector,v_qT)/getdot(v_qT,v_qT)
        v_paral_predict = scalermul(response, v_qT)
        u_paral_predict = getscale(v_paral_predict)-getscale(v_qT)
        v_perp_predict = vector - v_paral_predict
        u_perp_predict = getscale(v_perp_predict)
        return [u_perp_predict.cpu().detach().numpy(), u_paral_predict.cpu().detach().numpy(), response.cpu().detach().numpy()]

    resolutions = {
        'MET':      compute(v_MET),
        'puppiMET': compute(v_puppiMET)
    }

    
    # gen MET, regressed MET, and PUPPI MET
    METs = {
        'genMETx': qTx.cpu().detach().numpy(),
        'genMETy': qTy.cpu().detach().numpy(),
        'genMET': getscale(v_qT).cpu().detach().numpy(),
        
        'METx': METx.cpu().detach().numpy(),
        'METy': METy.cpu().detach().numpy(),
        'MET': getscale(v_MET).cpu().detach().numpy(),
        
        'puppiMETx': puppiMETx.cpu().detach().numpy(),
        'puppiMETy': puppiMETy.cpu().detach().numpy(),
        'puppiMET': getscale(v_puppiMET).cpu().detach().numpy()
    }

    return resolutions, METs

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'resolution': metric
}
