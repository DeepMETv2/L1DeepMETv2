import json
import os.path as osp
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm
import argparse
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime

'''

Change from DeepMETv2

1. Change x_cont, x_cat, etaphi in train() in accordance with the change of training inputs 
2. Add n_features_cont, n_features_cat to keep track of these numbers here and there, i.e. when building a model these numbers go into arguments
3. Remove the resolution-MET plotting part in evaluate, as L1 doesn't have bunch of METs that DeepMETv2 has access to
4. Add input scaling to [0,1] (norm)
5. Add weight_decay to the optimizer, remove patience from the scheduler 

'''


parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='data',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")
parser.add_argument('--batch_size', default=6,
                    help="Batch size")
parser.add_argument('--lr', default=0.01,
                    help="Learning rate")
parser.add_argument('--weight_decay', default=0.001,
                    help="Weight decay")

n_features_cont = 6
n_features_cat = 2

def train(model, device, optimizer, scheduler, loss_fn, dataloader, epoch):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()

    #with tqdm(total=len(dataloader)) as t:
    #    for data in dataloader:
    #        print(data.x[0])
    #        break

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x_cont = data.x[:,:n_features_cont]       # include puppi
            #x_cont = data.x[:,:(n_features_cont-1)]  # remove puppi
            x_cat = data.x[:,n_features_cont:].long()

            #phi = torch.atan2(data.x[:,2], data.x[:,1])   # atan2(py, px)
            etaphi = torch.cat([data.x[:,3][:,None], data.x[:,4][:,None]], dim=1)

            # NB: there is a problem right now for comparing hits at the +/- pi boundary
            edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=True, max_num_neighbors=255)
            result = model(x_cont, x_cat, edge_index, data.batch)
            loss = loss_fn(result, data.x, data.y, data.batch)
            loss.backward()
            optimizer.step()
            # update the average loss
            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
    scheduler.step(np.mean(loss_avg_arr))
    print('Training epoch: {:02d}, MSE: {:.4f}'.format(epoch, np.mean(loss_avg_arr)))

    model_dir = osp.join(os.environ['PWD'],args.ckpts)
    os.system('mkdir -p {}/MODELS'.format(model_dir))

    #torch.save(model, '{}/MODELS/model_epoch{}.pt'.format(model_dir, epoch))

    return np.mean(loss_avg_arr)

if __name__ == '__main__':
    args = parser.parse_args()

    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],args.data), 
                                               batch_size=int(args.batch_size),
                                               validation_split=.2)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    print('Training dataloader: {}, Test dataloader: {}'.format(len(train_dl), len(test_dl)))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #norm = torch.tensor([1./2950., 1./2950, 1./2950, 1., 1., 1.]).to(device)
    
    norm = torch.tensor([1., 1., 1., 1., 1., 1.]).to(device)   # pt, px, py: match it to genMETx genMETx scale factor
    #norm = torch.tensor([1./1000., 1./1000., 1./1000., 1., 1., 1.]).to(device)   # pt, px, py: match it to genMETx genMETx scale factor
    
    #norm = torch.tensor([1./499.25, 1./491.312, 1./495.928, 1./5.035, 1./3.142, 1.]).to(device)   # Have inputs within [0,1]

    model = net.Net(n_features_cont, n_features_cat, norm).to(device) #include puppi
    #model = net.Net(n_features_cont-1, n_features_cat, norm).to(device) #remove puppi
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(args.lr), weight_decay=float(args.weight_decay))
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-5, max_lr = 1e-4, cycle_momentum=False)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.05)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, threshold=0.05)
    first_epoch = 0
    best_validation_loss = 10e7
    deltaR = 0.4
    deltaR_dz = 0.3

    loss_fn = net.loss_fn
    metrics = net.metrics

    model_dir = osp.join(os.environ['PWD'],args.ckpts)
    os.system('mkdir -p {}'.format(model_dir))
    loss_log = open(model_dir+'/loss.log', 'w')
    loss_log.write('# loss log for training starting in '+strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
    loss_log.write('epoch, loss, val_loss\n')
    loss_log.flush()

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']

    for epoch in range(first_epoch+1,101):

        print('Current best loss:', best_validation_loss)
        if '_last_lr' in scheduler.state_dict():
            print('Learning rate:', scheduler.state_dict()['_last_lr'][0])

        # compute number of batches in one epoch (one full pass over the training set)
        train_loss = train(model, device, optimizer, scheduler, loss_fn, train_dl, epoch)

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'sched_dict': scheduler.state_dict()},
                              is_best=False,
                              checkpoint=model_dir)

        # Evaluate for one epoch on validation set
        #test_metrics = evaluate(model, device, loss_fn, test_dl, metrics, deltaR,deltaR_dz, model_dir)
        test_metrics, resolutions = evaluate(model, device, loss_fn, test_dl, metrics, deltaR,deltaR_dz, model_dir)

        validation_loss = test_metrics['loss']
        loss_log.write('%d,%.8f,%.8f\n'%(epoch,train_loss, validation_loss))
        loss_log.flush()
        is_best = (validation_loss<=best_validation_loss)

        # If best_eval, best_save_path
        if is_best: 
            print('Found new best loss!') 
            best_validation_loss=validation_loss

            # Save weights
            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'sched_dict': scheduler.state_dict()},
                                  is_best=True,
                                  checkpoint=model_dir)
            
            # Save best val metrics in a json file in the model directory
            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))

        #break

    loss_log.close()
