import networkx as nx
import random
import pandas as pd
import numpy as np
import scipy.stats as sc
from collections import defaultdict

# Generative functions for graphs

def generate_er(n, avg_degree):
    p = avg_degree / (n - 1)
    return nx.erdos_renyi_graph(n, p)

def generate_ba(n, avg_degree):
    m = avg_degree // 2
    return nx.barabasi_albert_graph(n, m)

def generate_ws(n, avg_degree, rewiring = 0.1):
    return nx.watts_strogatz_graph(n, avg_degree, rewiring)

def generate_sbm(n, avg_degree):

    sizes = [n // 2, n // 2]
    p_intra = avg_degree / ((11*n / 20) - 1)
    p_inter = p_intra / 10

    probs = [[p_intra, p_inter],\
            [p_inter, p_intra]]

    return nx.stochastic_block_model(sizes, probs)

# Generative functions for dataset

def create_graph_dataset(n, avg_degree = [6, 8, 10, 12], num_graphs_per_class = 1000):

    dataset = []

    for _ in range(num_graphs_per_class):
        dataset.append((generate_er(n, avg_degree[0]), 0))
        dataset.append((generate_ba(n, avg_degree[1]), 1))
        dataset.append((generate_ws(n, avg_degree[2]), 2))
        dataset.append((generate_sbm(n, avg_degree[3]), 3))

    random.shuffle(dataset) # Shuffle the dataset before splitting
    return dataset

def create_features_dataset(graph_dataset):

    def finite_scalar(value, default=0.0):
        value = float(value)
        return value if np.isfinite(value) else default

    dct = defaultdict(list)

    for G, t in graph_dataset:
        list_degree = [x[1] for x in G.degree]
        dct["mean_degree"].append(finite_scalar(np.mean(list_degree)))
        dct["std_degree"].append(finite_scalar(np.std(list_degree)))

        if len(set(list_degree)) <= 1:
            dct["skew_degree"].append(0.0)
            dct["kurtosis_degree"].append(0.0)
        else:
            dct["skew_degree"].append(finite_scalar(sc.skew(list_degree)))
            dct["kurtosis_degree"].append(finite_scalar(sc.kurtosis(list_degree)))

        dct["max_degree"].append(finite_scalar(np.max(list_degree)))
        dct["min_degree"].append(finite_scalar(np.min(list_degree)))
        dct["assortativity_degree_coeff"].append(
            finite_scalar(nx.degree_assortativity_coefficient(G))
        )

        list_cluster_coeff = list(nx.clustering(G).values())
        dct["mean_cluster_coeff"].append(finite_scalar(nx.average_clustering(G)))
        dct["std_cluster_coeff"].append(finite_scalar(np.std(list_cluster_coeff)))
        dct["max_cluster_coeff"].append(finite_scalar(np.max(list_cluster_coeff)))
        dct["count_triangle"].append(finite_scalar(sum(nx.triangles(G).values()) / 3))

        # Global distance-based metrics are undefined on disconnected graphs.
        # Use the largest connected component for stable feature extraction.
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc_nodes).copy()

        dct["diameter"].append(finite_scalar(nx.diameter(G_lcc)))
        dct["mean_shortest_path"].append(
            finite_scalar(nx.average_shortest_path_length(G_lcc))
        )

        eccentricity_values = list(nx.eccentricity(G_lcc).values())
        dct["mean_eccentricity"].append(finite_scalar(np.mean(eccentricity_values)))
        dct["max_eccentricity"].append(finite_scalar(np.max(eccentricity_values)))

        list_betweenness_centrality = list(nx.betweenness_centrality(G).values())
        dct["mean_b_centrality"].append(finite_scalar(np.mean(list_betweenness_centrality)))
        dct["std_b_centrality"].append(finite_scalar(np.std(list_betweenness_centrality)))
        dct["max_b_centrality"].append(finite_scalar(np.max(list_betweenness_centrality)))

        try:
            list_eigenvector_centrality = list(nx.eigenvector_centrality(G, max_iter=500).values())
        except nx.PowerIterationFailedConvergence:
            list_eigenvector_centrality = [0.0] * G.number_of_nodes()

        dct["mean_e_centrality"].append(finite_scalar(np.mean(list_eigenvector_centrality)))
        dct["std_e_centrality"].append(finite_scalar(np.std(list_eigenvector_centrality)))
        dct["max_e_centrality"].append(finite_scalar(np.max(list_eigenvector_centrality)))

        list_lapl_eigenvalues = np.sort(np.real(nx.laplacian_spectrum(G)))
        for i in range(5):
            eig_value = list_lapl_eigenvalues[i] if i < len(list_lapl_eigenvalues) else 0.0
            dct[f"eig_{i + 1}"].append(finite_scalar(eig_value))

        dct["number_connected_components"].append(nx.number_connected_components(G))

        dct["target"].append(t)

    return pd.DataFrame(dct)


def split_dataset(dataset, train_ratio = 0.7, val_ratio = 0.15):
    n = len(dataset)
    train_len = int(n * train_ratio)
    val_len = int(n * val_ratio)

    train = dataset[:train_len]
    val = dataset[train_len:train_len + val_len]
    test = dataset[train_len + val_len:]
    return train, val, test
