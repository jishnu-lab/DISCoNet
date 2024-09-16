import argparse, logging, ast, torch
import analyze
from torch_geometric.nn import GAT
class GAT_Module(torch.nn.Module):
    def __init__(self, out_channels = 256, hidden_layers = 2, v2=True, concat=True,heads=1,add_self_loops=False):
        super(GAT_Module,self).__init__()
        self.gat = GAT(in_channels = -1, hidden_channels = 2*out_channels, num_layers = hidden_layers, out_channnels=out_channels, concat=concat, heads=heads, add_self_loops=add_self_loops)

    def forward(self, data):
        x = data.x.to(torch.float32)
        z = self.gat(x,data.edge_index)
        return z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', help= 'Parameters', type=str, required=True)
    parser.add_argument('-s', '--save_path', help= 'Save Path', type=str, required=True)

    # GNN parameters
    gnn_args = parser.add_argument_group('GAT parameters', 'Set of parameters to define the architecture of the graph attention network.')
    gnn_args.add_argument('--embedding_size', help='The output embedding size of each node.', type=int, default=256)
    gnn_args.add_argument('--hidden_layers', help='Number of hidden layers in the graph attention network.', type=int, default=2)
    gnn_args.add_argument('--epochs', help='Number of epochs to train', type=int, default=4)
    gnn_args.add_argument('-b', '--batch_size', help='Batch size for the data loader', type=int, default=2048)
    gnn_args.add_argument('-n', '--neg', help='Edge negative sampling ratio', type=int, default=1)
    gnn_args.add_argument('-le', '--logging_epoch', help='Logging epoch', type=int, default=1)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = args.save_path
    year, data, component, embedding, model, control = ast.literal_eval(args.parameters)
    out_channels = args.embedding_size
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    batch_size = args.batch_size
    neg_ratio = args.neg
    log_epoch = args.logging_epoch

    path = f"{save_path}/{year}/{data}/{component}/{embedding}/{model}"
    
    if model != 'gat' and model != 'gat_graph_embed':
        logging.error("This script only supports 'gat' or 'gat_graph_embed'. 'gat' or 'gat_graph_embed' not found in models. Hence Exiting.")
        exit()
    
    if int(epochs/log_epoch)==epochs:
        logging.warning(f"Logging for every epoch will be slow. Consider increasing log_epoch parameter")

    gat = GAT_Module(out_channels,hidden_layers).to(device)
    optimizer = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=0.001)
    graph_ed = False
    neg_node = None

    all_data, train_data, val_data, test_data = analyze._read_data(path,device)
    if control is not None:
        path = f"{path}/{control}"
    analyze.run_gat(path, all_data, train_data, val_data, test_data, batch_size, neg_ratio, device, gat, optimizer, epochs, log_epoch, neg_node, graph_ed,control) 