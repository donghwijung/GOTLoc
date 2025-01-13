import random

import numpy as np
from sklearn.cluster import DBSCAN

import torch

from scene_graph import SceneGraph

import spacy
nlp = spacy.load("en_core_web_lg")

def make_cross_graph(x_1_dim, x_2_dim):
    x_1_dim = x_1_dim[0]
    x_2_dim = x_2_dim[0]

    edge_index_cross = torch.tensor([[], []], dtype=torch.long)
    edge_attr_cross = torch.tensor([], dtype=torch.float)

    # Add edge from each node in x_1 to x_2
    for i in range(x_1_dim):
        for j in range(x_2_dim):
            edge_index_cross = torch.cat((edge_index_cross, torch.tensor([[i], [x_1_dim + j]], dtype=torch.long)), dim=1)
            # Add edge_attr which is dimension 1x300, all 0
            edge_attr_cross = torch.cat((edge_attr_cross, torch.zeros((1, 300), dtype=torch.float)), dim=0) # TODO: dimension 300

    assert(edge_index_cross.shape[1] == x_1_dim * x_2_dim)
    assert(edge_attr_cross.shape[0] == x_1_dim * x_2_dim)
    return edge_index_cross, edge_attr_cross

def cross_entropy(preds, targets, reduction='none', dim=-1): # TODO: This could be why the loss never goes above 1
    log_softmax = torch.nn.LogSoftmax(dim=dim) 
    loss = (-targets * log_softmax(preds)).sum(1)
    assert(all(loss >= 0))
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()    

def k_fold_by_scene(dataset, folds: int):
    '''
    dataset: should be a list of SceneGraphs
    '''
    # Separate the dataset by scene
    scene_dataset = {} # mapping from scene name to list of indices from the dataset
    for i, graph in enumerate(dataset):
        if graph.scene_id not in scene_dataset:
            scene_dataset[graph.scene_id] = []
        scene_dataset[graph.scene_id].append(i)

    # Create the folds based on the scene name
    random.seed(0)
    scene_names = list(scene_dataset.keys())
    random.shuffle(scene_names)
    fold_size = len(scene_names) // folds
    train_indices, val_indices = [], []
    train_scene_names_to_check, val_scene_names_to_check = [], []
    for i in range(folds):
        val_scene_names = scene_names[i * fold_size : (i + 1) * fold_size]
        val_indices.append([idx for scene_name in val_scene_names for idx in scene_dataset[scene_name]])
        train_indices.append([idx for scene_name in scene_names if scene_name not in val_scene_names for idx in scene_dataset[scene_name]])
        val_scene_names_to_check.append(val_scene_names)
        train_scene_names_to_check.append([scene_name for scene_name in scene_names if scene_name not in val_scene_names])

    return zip(train_indices, val_indices)

def combine_node_features(graph1, graph2):
    node_features1 = graph1.get_node_features()
    node_features2 = graph2.get_node_features()
    all_node_features = np.concatenate((node_features1, node_features2), axis=0)
    all_node_graph_index = np.concatenate((np.zeros(len(node_features1)), np.ones(len(node_features2))), axis=0) # graph1 is 0, graph2 is 1
    return all_node_features, all_node_graph_index

def get_matching_subgraph(graph1, graph2):
    # Cluster the nodes in both graphs with dbscan
    all_node_features, all_node_graph_index = combine_node_features(graph1, graph2)
    combined_node_idx = np.concatenate(([n1 for n1 in graph1.nodes], [n2 for n2 in graph2.nodes]), axis=0)
    assert(all([i == graph1.nodes[i].idx for i in graph1.nodes])) # key equals the idx
    assert(all([i == graph2.nodes[i].idx for i in graph2.nodes]))
    idx_mapping = {}
    for i, idx in enumerate(combined_node_idx):
        idx_mapping[i] = idx

    # Track the indices of the nodes that are matched, after combining into all_node_features
    clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(all_node_features) # default 0.05
    clusters = {}
    for i, cluster in enumerate(clustering.labels_):
        if cluster in clusters:
            clusters[cluster].append(i)
        else:
            clusters[cluster] = [i]

    # Process the clusters so that only clusters with nodes from both graphs remain
    graph1_keep_nodes = []
    graph2_keep_nodes = []
    for cluster in clusters:
        indices = clusters[cluster]
        graphs = [int(all_node_graph_index[i]) for i in indices]
        if 0 in graphs and 1 in graphs:
            graph1_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 0])
            graph2_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 1])

    # Get the subgraph
    assert(type(graph1) == SceneGraph)
    assert(type(graph2) == SceneGraph)        
    graph1_keep_nodes = list(set(graph1_keep_nodes))
    graph2_keep_nodes = list(set(graph2_keep_nodes))
    subgraph1 = graph1.get_subgraph(graph1_keep_nodes, return_graph=True)
    subgraph2 = graph2.get_subgraph(graph2_keep_nodes, return_graph=True)
    return subgraph1, subgraph2