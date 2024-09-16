from .analyze_helpers import _sane_loader, _scaled_recon_loss
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import pandas as pd
from statistics import mean
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import ast, torch, argparse, logging, os, sys, pickle
from tqdm import tqdm
logger = logging.getLogger(__name__)

# Training, testing, and model
def train(loader, model, optimizer, device, neg_node=None, control=None):
    model.train()
    total_examples = total_loss = 0
    pseudo_node_flag = False

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad() # Clear gradients for each batch, adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py#L60
        batch.to(device)
        if neg_node is not None and pseudo_node_flag is False and neg_node in batch.n_id: #It will use neg_node from the main funciton. Hence not defined here
            pseudo_node_flag = True
        z = model(batch)
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
        loss = _scaled_recon_loss(z,pos_edge_index,neg_edge_index)
        loss.backward()
        optimizer.step()
        total_examples += edge_lbl.size(0)
        total_loss += float(loss) * edge_lbl.size(0)
    
    if neg_node is not None and pseudo_node_flag is False: #It will use neg_node from the main funciton. Hence not defined here
        logger.error("Pseudonode not found in train data even though it is present in all_data")
    return total_loss / total_examples

@torch.no_grad()
def test(loader, model, device, graph_ed=False, control=None, plt_hist=False, save='histogram.png'):
    model.eval()
    all_author0,all_author1 =torch.tensor([], device=device),torch.tensor([], device=device)
    all_pred,all_label,all_loss = torch.tensor([], device=device),torch.tensor([], device=device),[]
    graph_embedding = torch.tensor([], device=device)

    for batch in tqdm(loader, desc="Testing"):
        batch.to(device)
        z = model(batch)
        if graph_ed:
            graph_embedding = torch.cat((graph_embedding, z[batch.y==-1].detach()), dim=-1)
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

    if graph_ed:
        return all_author0,all_author1,all_pred,all_label,all_loss,graph_embedding
    else:
        return all_author0,all_author1,all_pred,all_label,all_loss

def data_loaders(all_data,train_data,val_data,test_data,batch_size,neg_ratio):
    all_loader = LinkNeighborLoader(
        data=all_data,
        num_neighbors=[2,3,1],
        neg_sampling='binary',
        neg_sampling_ratio=0,
        batch_size=batch_size,
        shuffle=True,
    )
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[2,3,1],
        neg_sampling='binary',
        neg_sampling_ratio=neg_ratio,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[2,3,1],
        neg_sampling='binary',
        neg_sampling_ratio=neg_ratio,
        batch_size=batch_size,
        shuffle=False,
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
    _sane_loader(next(iter(train_loader)), "Train")
    _sane_loader(next(iter(val_loader)), "Validation")
    _sane_loader(next(iter(test_loader)), "Test")

    return all_loader, train_loader, val_loader, test_loader

def run_gat(path, all_data, train_data, val_data, test_data, batch_size, neg_ratio, device, gat, optimizer, epochs, log_epoch, neg_node, graph_ed,control=None):
    logger.info(f"Running GAT for {path}")
    all_loader, train_loader, val_loader, test_loader = data_loaders(all_data, train_data,val_data,test_data,batch_size,neg_ratio)
    for epoch in tqdm(range(1,epochs+1)):
        loss = train(train_loader, gat, optimizer, device, neg_node=neg_node, control=control)
        print(f'Epoch: {epoch:02d}, Train_Loss: {loss:.4f}')
        if epoch%log_epoch==0:
            print(f'Epoch: {epoch:02d}, Running Validation')
            _,_,_,_,_ = test(val_loader,gat,device)
            print(f'Epoch: {epoch:02d}, Val_Loss: {loss:.4f}')
        if epoch==epochs:
            print(f'Epoch: {epoch:02d}, Running Test')
            if graph_ed:
                logger.info(f"Saving Graph Embedding for {path}")
                author0,author1,pred,label,loss,graph_ed = test(test_loader,gat,device,graph_ed=True,control=control)
                torch.save(graph_ed, f"{path}/graph_embedding_gat_{epoch:02d}.pt")
            else:
                author0,author1,pred,label,loss = test(test_loader,gat,device,control=control)
            logger.info(f"Saving Files for Epoch: {epoch:02d}...")
            test_df = pd.DataFrame({'author0': author0.cpu().numpy(), 'author1': author1.cpu().numpy(), 'pred': pred.cpu().numpy(), 'label': label.cpu().numpy()})
            test_df.to_pickle(f'{path}/test_df.pkl')
    return None