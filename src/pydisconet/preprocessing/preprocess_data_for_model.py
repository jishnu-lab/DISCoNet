from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T
from torch_geometric.data import Data
import logging,torch
logger = logging.getLogger(__name__)

def data_overview(data):
    logger.info('Overview of the dataset...')
    logger.info(f'Data has self loops: {data.has_self_loops()}')
    # Remove self loops from the data.
    if data.has_self_loops():
        data.edge_index, data.edge_attr=remove_self_loops(data.edge_index,data.edge_attr)
        logger.info(f'\tRemoving self loops')
    logger.info(f'Dataset validation passed: {data.validate()}')
    logger.info(f'# of nodes: {data.num_nodes}')
    logger.info(f'# of edges (after undirected transformed applied): {data.num_edges}')
    logger.info(f'Data has edge attributes: {"edge_attr" in data}')
    logger.info(f'Data has isolated nodes: {data.has_isolated_nodes()}')
    logger.info(f'Data is directed: {data.is_directed()}')
    return data

def save_data_objects(path, data_object, num_val, num_test, neg_sampling_ratio):
    split_transform = T.RandomLinkSplit(
            num_val=num_val,
            num_test=num_test,
            disjoint_train_ratio=0,
            neg_sampling_ratio=neg_sampling_ratio,
            add_negative_train_samples=False,
            split_labels=False,
            is_undirected=False,
            )
    train_data, val_data, test_data = split_transform(data_object)
    torch.save(data_object, f"{path}/all_data_object.pt")
    torch.save(train_data, f"{path}/train_data_object.pt")
    torch.save(val_data, f"{path}/val_data_object.pt")
    torch.save(test_data, f"{path}/test_data_object.pt")

    return None

def preprocessing_for_model(path,model,num_val=0.2,num_test=0.3,neg_sampling_ratio=0.0, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    logger.info(f"Preprocessing {path}/{model}...")
    if neg_sampling_ratio!=0:
        logger.warning("Negative Sampling Ratio is not 0. PLEASE KNOW by DEFAULT data loaders in subsequent steps will also add negative samples on the fly.") is not None
    torch.manual_seed(42)
    data_object = torch.load(f"{path}/embedded_dataset.pt",map_location=device)
    data_object.to('cpu')
    if model == 'zeroshot' or model =='gat':
        data_object_new = data_overview(data_object)
    else:
        pseudo_node_features = torch.rand((1, data_object.x.shape[1]))   # Features should be size because TFIDF may not necessarily have 768 features. DON'T HARD CODE IT TO 768
        x_new = torch.cat((data_object.x, pseudo_node_features), dim=0)  # Add pseudo-node to features

        edge_index = data_object.edge_index
        num_nodes = data_object.num_nodes
        pseudo_edges = torch.tensor([[num_nodes, i] for i in range(num_nodes)], dtype=torch.long)  # Edges from pseudo-node to all other nodes
        edge_index_new = torch.cat((edge_index, pseudo_edges.t()), dim=1)
        
        y_new = torch.cat([data_object.y, torch.tensor([-1])], dim=0)
        data_object_new = Data(x=x_new, edge_index=edge_index_new, y=y_new)
        data_object_new = data_overview(data_object_new) # Data will be directed here since only one way connection.

    path = f"{path}/{model}"
    save_data_objects(path, data_object_new, num_val, num_test, neg_sampling_ratio)
    logger.info(f"Saved Data Objects for {path}")
    return None