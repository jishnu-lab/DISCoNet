import csv, torch, logging
import numpy as np
import networkit as nkit
logger = logging.getLogger(__name__)

def save_network_statistics(path, component, G, network_properties, data_object, num_nodes):
    path = f'{path}/{component}'
    nbw = nkit.centrality.ApproxBetweenness(G).run().scores()
    deg_c = nkit.centrality.DegreeCentrality(G,normalized=True).run().scores()
    local_clus_coeff = nkit.centrality.LocalClusteringCoefficient(G).run().scores()
    degrees = np.array([G.degree(node_i) for node_i in range(num_nodes)])
    counts = np.array(data_object.y.cpu())
    logger.info(f"Calculated Node Betweenness Centrality, Degree Centrality, Clustering Coefficient, Node Degrees and Paper Counts for {path}...")

    if component == 'full':
        network_properties['avg_node_betweenness'] = np.mean(nbw)
        network_properties['avg_degree_centrality'] = np.mean(deg_c)
        network_properties['avg_node_degrees'] = np.mean(degrees)
        network_properties['avg_paper_counts'] = np.mean(counts)
        logger.info(f"Saving Network Properties for {path}...")
        with open(f"{path}_network_properties.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Property', 'Value'])  # Optional: writing header
            for key, value in network_properties.items():
                writer.writerow([key, value])

    torch.save(nbw, f"{path}/node_betweenness.pt")
    torch.save(deg_c, f"{path}/degree_centrality.pt")
    torch.save(local_clus_coeff, f"{path}/clustering_coefficient.pt")
    torch.save(degrees, f"{path}/degrees.pt")
    torch.save(counts, f"{path}/paper_counts.pt")
    logger.info(f"Saving Node Betweenness Centrality, Degree Centrality, Clustering Coefficient, Node Degrees and Paper Counts for {path}...")

    return None

def get_network_properties(G):
    network_properties = {}
    network_properties['num_nodes'] = G.numberOfNodes()
    network_properties['num_edges'] = G.numberOfEdges()
    cc = nkit.components.ConnectedComponents(G).run()
    lcc = cc.extractLargestConnectedComponent(G)

    network_properties['num_nodes_lcc'] = lcc.numberOfNodes()
    network_properties['num_edges_lcc'] = lcc.numberOfEdges()
    network_properties['avg_local_clustering_coefficient'] = nkit.globals.ClusteringCoefficient.sequentialAvgLocal(G)
    network_properties['exact_global_clustering_coefficient'] = nkit.globals.ClusteringCoefficient.exactGlobal(G)
    network_properties['network_diameter'] = nkit.distance.Diameter(G).run().getDiameter()[0] # Network Diameter

    return network_properties

def calc_save_network_statistics(path, component, data_object):
    logger.info(f"Calculating Network Statistics for {path}...")
    num_nodes = data_object.x.shape[0]
    G = nkit.Graph(n=num_nodes)
    for node_i,node_j in data_object.edge_index.T:
        G.addEdge(node_i.item(), node_j.item())
    G.removeSelfLoops()
    network_properties = get_network_properties(G)
    save_network_statistics(path, component, G, network_properties, data_object, num_nodes)
    return None