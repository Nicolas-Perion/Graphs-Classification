import networkx as nx
import random
import pandas as pd
import numpy as np
import scipy.stats as sc
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Generative functions for graphs

def _generate_er(n, avg_degree, seed=None):
    p = avg_degree / (n - 1)
    return nx.erdos_renyi_graph(n, p, seed=seed)

def _generate_ba(n, avg_degree, seed=None):
    m = avg_degree // 2
    return nx.barabasi_albert_graph(n, m, seed=seed)

def _generate_ws(n, avg_degree, rewiring = 0.1, seed=None):
    return nx.watts_strogatz_graph(n, avg_degree, rewiring, seed=seed)

def _generate_sbm(n, avg_degree, seed=None):

    sizes = [n // 2, n // 2]
    # The value of p_intra and p_inter ensure that the average degree is equal to avg_degree.
    p_intra = avg_degree / ((11*n / 20) - 1)
    p_inter = p_intra / 10

    probs = [[p_intra, p_inter],\
            [p_inter, p_intra]]

    return nx.stochastic_block_model(sizes, probs, seed=seed)

# Generative functions for graph dataset

def create_graph_dataset(n, avg_degree = [6, 8, 10, 12], num_graphs_per_class = 1000, seed=None):

    dataset = []

    for _ in range(num_graphs_per_class):
        dataset.append((_generate_er(n, avg_degree[0] if isinstance(avg_degree, list) else avg_degree, seed=seed), 0))
        dataset.append((_generate_ba(n, avg_degree[1] if isinstance(avg_degree, list) else avg_degree, seed=seed), 1))
        dataset.append((_generate_ws(n, avg_degree[2] if isinstance(avg_degree, list) else avg_degree, seed=seed), 2))
        dataset.append((_generate_sbm(n, avg_degree[3] if isinstance(avg_degree, list) else avg_degree, seed=seed), 3))

    random.shuffle(dataset) # Shuffle the dataset before splitting
    return dataset

# Generative functions for features dataset

def create_features_dataset(graph_dataset):

    dct = defaultdict(list)

    for G, t in graph_dataset:

        list_degree = [x[1] for x in G.degree]
        dct["mean_degree"].append(np.mean(list_degree))
        dct["std_degree"].append(np.std(list_degree))
        dct["skew_degree"].append(sc.skew(list_degree))
        dct["kurtosis_degree"].append(sc.kurtosis(list_degree))
        dct["max_degree"].append(np.max(list_degree))
        dct["min_degree"].append(np.min(list_degree))
        dct["assortativity_degree_coeff"].append(nx.degree_assortativity_coefficient(G))

        list_cluster_coeff = list(nx.clustering(G).values())
        dct["mean_cluster_coeff"].append(nx.average_clustering(G))
        dct["std_cluster_coeff"].append(np.std(list_cluster_coeff))
        dct["max_cluster_coeff"].append(np.max(list_cluster_coeff))
        dct["count_triangle"].append(sum(nx.triangles(G).values()) / 3)

        # Global distance-based metrics are undefined on disconnected graphs.
        # Use the largest connected component for stable feature extraction.
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc_nodes).copy()

        dct["diameter"].append(nx.diameter(G_lcc))
        dct["mean_shortest_path"].append(nx.average_shortest_path_length(G_lcc))

        eccentricity_values = list(nx.eccentricity(G_lcc).values())
        dct["mean_eccentricity"].append(np.mean(eccentricity_values))
        dct["max_eccentricity"].append(np.max(eccentricity_values))

        list_betweenness_centrality = list(nx.betweenness_centrality(G).values())
        dct["mean_b_centrality"].append(np.mean(list_betweenness_centrality))
        dct["std_b_centrality"].append(np.std(list_betweenness_centrality))
        dct["max_b_centrality"].append(np.max(list_betweenness_centrality))

        try:
            list_eigenvector_centrality = list(nx.eigenvector_centrality(G_lcc, max_iter=500).values())
        except nx.PowerIterationFailedConvergence:
            list_eigenvector_centrality = [0.0] * G.number_of_nodes()

        dct["mean_e_centrality"].append(np.mean(list_eigenvector_centrality))
        dct["std_e_centrality"].append(np.std(list_eigenvector_centrality))
        dct["max_e_centrality"].append(np.max(list_eigenvector_centrality))

        list_lapl_eigenvalues = np.sort(np.real(nx.laplacian_spectrum(G)))
        for i in range(5):
            eig_value = list_lapl_eigenvalues[i] if i < len(list_lapl_eigenvalues) else 0.0
            dct[f"eig_{i + 1}"].append(eig_value)

        dct["number_connected_components"].append(nx.number_connected_components(G))

        dct["target"].append(t)

    return pd.DataFrame(dct)

# Splitting the features dataset

def splitting(X, y = None, test_size = 0.15, val_size = 0.15, seed=None):
    val_size = val_size / (1 - test_size)

    if y is not None: # If y is not None, we are splitting datasets (of features) X and y

        X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = seed,
        stratify = y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size = val_size,
            random_state = seed,
            stratify = y_train_val
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    else: # If y is None, we are splitting a list (of tuples) X

        X_train_val, X_test = train_test_split(
            X,
            test_size = test_size,
            random_state = seed,
            stratify = [t for _, t in X]
        )

        X_train, X_val = train_test_split(
            X_train_val,
            test_size = val_size,
            random_state = seed,
            stratify = [t for _, t in X_train_val]
        )
        
        return X_train, X_val, X_test