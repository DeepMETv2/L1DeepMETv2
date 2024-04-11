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
                                        conv_depth=4)
    
    def forward(self, x_cont, x_cat, edge_index, batch):
        weights = self.graphnet(x_cont, x_cat, edge_index, batch)
        relu_layer = nn.ReLU()
        return relu_layer(weights)
        #return torch.sigmoid(weights)

# tensor operations
def getdot(vx, vy):
    return torch.einsum('bi,bi->b',vx,vy)

def getscale(vx):
    return torch.sqrt(getdot(vx,vx))

def scalermul(a,v):
    return torch.einsum('b,bi->bi',a,v)

# loss function without response tune option
def loss_fn(weights, particles_vis, genMET, batch, scale_momentum = 1.):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0] / scale_momentum
    true_py = (-1)*genMET[:,1] / scale_momentum

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean()

    return loss


# loss function with response tune
def loss_fn_response_tune(weights, particles_vis, genMET, batch, c = 5000, scale_momentum = 1.):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0] / scale_momentum
    true_py = (-1)*genMET[:,1] / scale_momentum

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean() 

    # response correction
    v_true = torch.stack((true_px,true_py),dim=1)
    v_regressed = torch.stack((uTx, uTy),dim=1)
        
    # response = getdot( v_true, v_regressed ) / getdot( v_true, v_true ) # dot product
    response = getscale(v_regressed) / getscale(v_true) # ratio of the MET scale

    #pT_thres = 0.         # calculate response only taking into account for events with genMET above threshold
    pT_thres = 50./scale_momentum
    resp_pos = torch.logical_and(response > 1., getscale(v_true) > pT_thres)
    resp_neg = torch.logical_and(response < 1., getscale(v_true) > pT_thres)
    
    c = c / scale_momentum
    
    response_term = c * (torch.sum(1 - response[resp_neg]) + torch.sum(response[resp_pos] - 1))

    loss += response_term

    return loss


# loss function; loss normalized with genMETx and genMETy
def loss_fn_relative(weights, particles_vis, genMET, batch, c = 5000):
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

    loss=0.5*( ( (uTx - true_px) / true_px)**2 + ( (uTy - true_py) / true_py)**2 ).mean() 

    return loss


# loss function; loss normalized with genMET scale
def loss_fn_relative_genMET(weights, particles_vis, genMET, batch, c = 5000):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0]
    true_py = (-1)*genMET[:,1]

    v_true = torch.stack((true_px,true_py),dim=1)
    true_uT = getscale(v_true)
    
    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ((uTx - true_px)**2 + (uTy - true_py)**2) / true_uT**2 ).mean() 

    return loss


# loss function flatten MET
def loss_fn_flattenMET(weights, particles_vis, genMET, batch, sample_weight = None):
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
    
    # flatten out MET
    if sample_weight != None:
        binnings = [0, 20, 40, 60, 80, 100, 120, 140, 160, 1000]

        per_genMET_bin_weight = [0.14890485, 0.05692836, 0.04364244, 0.04413395, 0.05471598, \
                                 0.08185655, 0.13920728, 0.25549501, 0.17511559]
        per_genMET_bin_weight = torch.tensor(per_genMET_bin_weight)

        v_true = torch.stack((true_px,true_py),dim=1)
        
        true_uT = getscale(v_true)

        for idx in range(len(binnings)-1):
            mask_uT = (true_uT > binnings[idx]) & (true_uT <= binnings[idx+1])

            sample_weight[mask_uT] = per_genMET_bin_weight[idx]

        #print(true_uT)
        #print(sample_weight)

        loss=0.5*( ( ( uTx - true_px)**2 + ( uTy - true_py)**2 ) * sample_weight ).mean()

    else:
        loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean()
        
    return loss



# calculate performance metrics
def metric(weights, particles_vis, genMET, batch, scale_momentum = 1.):
    # qT is the genMET
    qTx = genMET[:,0]
    qTy = genMET[:,1]

    v_qT=torch.stack((qTx,qTy),dim=1)

    # momentum of visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]

    # regressed uT: momentum of the system of all visible particles
    uTx = scatter_add(weights*px, batch) * scale_momentum
    uTy = scatter_add(weights*py, batch) * scale_momentum

    # regressed MET
    METx = (-1) * uTx
    METy = (-1) * uTy

    v_MET=torch.stack((METx, METy),dim=1)

    # PUPPI MET using PUPPI weights
    wgt_puppi = particles_vis[:,5]

    puppiMETx = (-1)*scatter_add(wgt_puppi*px, batch) * scale_momentum
    puppiMETy = (-1)*scatter_add(wgt_puppi*py, batch) * scale_momentum
   
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

    
    mask_down = torch.abs(particles_vis[:,6]) == 1
    mask_up = torch.abs(particles_vis[:,6]) == 2
    mask_electron = torch.abs(particles_vis[:,6]) == 11
    mask_muon = torch.abs(particles_vis[:,6]) == 13
    mask_photon = torch.abs(particles_vis[:,6]) == 22
    mask_kaon_zero = torch.abs(particles_vis[:,6]) == 130
    mask_pion_charged = torch.abs(particles_vis[:,6]) == 211
    
    weights = {
        'down': weights[mask_down].detach().cpu().numpy(),
        'up': weights[mask_up].detach().cpu().numpy(),
        'electron': weights[mask_electron].detach().cpu().numpy(),
        'muon': weights[mask_muon].detach().cpu().numpy(),
        'photon': weights[mask_photon].detach().cpu().numpy(),
        'kaon': weights[mask_kaon_zero].detach().cpu().numpy(),
        'pion': weights[mask_pion_charged].detach().cpu().numpy(),
        
    }
    
    puppi_weights = {
        'down': particles_vis[:,5][mask_down].detach().cpu().numpy(),
        'up': particles_vis[:,5][mask_up].detach().cpu().numpy(),
        'electron': particles_vis[:,5][mask_electron].detach().cpu().numpy(),
        'muon': particles_vis[:,5][mask_muon].detach().cpu().numpy(),
        'photon': particles_vis[:,5][mask_photon].detach().cpu().numpy(),
        'kaon': particles_vis[:,5][mask_kaon_zero].detach().cpu().numpy(),
        'pion': particles_vis[:,5][mask_pion_charged].detach().cpu().numpy(),
    }
    
    return resolutions, METs, weights, puppi_weights

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'resolution': metric
}
