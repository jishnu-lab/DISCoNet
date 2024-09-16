from .analyze_helpers import _sane_loader, _scaled_recon_loss
import torch.nn.functional as F
import argparse, logging, ast, os, pickle, sys, torch
from tqdm import tqdm
from torch_geometric.loader import LinkNeighborLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
logger = logging.getLogger(__name__)

def test(loader, device, control=None):
    all_author0,all_author1 =torch.tensor([], device=device),torch.tensor([], device=device)
    all_pred,all_label,all_loss = torch.tensor([], device=device),torch.tensor([], device=device),[]
    
    for batch in tqdm(loader, desc="Zero Shot Progress:"):
        batch.to(device)
        z = batch.x
        edge_lbl = torch.clamp(batch.edge_label, max=1)
        if control == 'shuffle_y':
            logger.info("Shuffling edge labels")
            edge_lbl = edge_lbl[torch.randperm(edge_lbl.size(0))]
        elif control == 'shuffle_x':
            logger.info("Shuffling node features")
            z = z[torch.randperm(z.size(0)),:]
        else:
            edge_lbl = edge_lbl

        pos_edge_index = batch.edge_label_index[:,edge_lbl==1]
        neg_edge_index = batch.edge_label_index[:,edge_lbl==0]

        author0 = torch.cat([pos_edge_index[0], neg_edge_index[0]],dim=0)
        author1 = torch.cat([pos_edge_index[1], neg_edge_index[1]],dim=0)

        pos_pred = torch.clamp((F.cosine_similarity(z[pos_edge_index[0]], z[pos_edge_index[1]], dim=1)*0.5)+0.5,min=0,max=1)
        neg_pred = torch.clamp((F.cosine_similarity(z[neg_edge_index[0]], z[neg_edge_index[1]], dim=1)*0.5)+0.5,min=0,max=1)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        label = torch.cat([pos_y, neg_y], dim=0)

        loss = _scaled_recon_loss(z,pos_edge_index,neg_edge_index)

        all_author0=torch.cat((all_author0, author0.detach()), dim=-1)
        all_author1=torch.cat((all_author1, author1.detach()), dim=-1)
        all_pred=torch.cat((all_pred, pred.detach()), dim=-1)
        all_label=torch.cat((all_label, label.detach()), dim=-1)
        all_loss.append(loss.item())

    print(f"ROC: {roc_auc_score(all_label.cpu(), all_pred.cpu())}, APS: {average_precision_score(all_label.cpu(), all_pred.cpu())}")

    return all_author0,all_author1,all_pred,all_label,all_loss

def data_loaders(all_data,test_data,batch_size,neg_ratio):
    all_loader = LinkNeighborLoader(
        data=all_data,
        num_neighbors=[2,3,1],
        neg_sampling='binary',
        neg_sampling_ratio=neg_ratio,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[2,3,1],
        neg_sampling='binary',
        neg_sampling_ratio=neg_ratio,
        batch_size=batch_size,
        shuffle=False,
    )
    # Sanity Check
    _sane_loader(next(iter(all_loader)), "All")
    _sane_loader(next(iter(test_loader)), "Test")

    return all_loader, test_loader

def run_zeroshot(path,all_data,test_data,batch_size,neg_ratio,device,control=None):
    logger.info(f"Running Zero Shot for {path}")
    all_loader, test_loader = data_loaders(all_data,test_data,batch_size,neg_ratio)
    #### Add shuffling functions here
    author0,author1,pred,label,loss = test(all_loader,device,control=control)
    train_df = pd.DataFrame({'author0': author0.cpu().numpy(), 'author1': author1.cpu().numpy(), 'pred': pred.cpu().numpy(), 'label': label.cpu().numpy()})
    train_df.to_pickle(f'{path}/all_df.pkl')
    t_author0,t_author1,t_pred,t_label,t_loss = test(test_loader,device,control=control)
    test_df = pd.DataFrame({'author0': t_author0.cpu().numpy(), 'author1': t_author1.cpu().numpy(), 'pred': t_pred.cpu().numpy(), 'label': t_label.cpu().numpy()})
    test_df.to_pickle(f'{path}/test_df.pkl')
    return