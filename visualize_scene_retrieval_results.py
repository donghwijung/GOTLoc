from args import get_args
import copy
import os
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch

import config
from eval_utils import eval
from models import BigGNN

def scene_graph_to_osmnx_graph(scene_graph):
    graph = nx.DiGraph()
    nodes = scene_graph.nodes

    node_ids = []
    pos = {}

    for k,v in nodes.items():
        node_id = k
        node_ids.append(node_id)
        node = v
        graph.add_node(node_id, label=node.label, attribute=node.attributes, features=node.features)
        node_centroid = node.centroid
        pos[node_id] = node_centroid

    edge_relations = scene_graph.edge_relations
    edge_idx = scene_graph.edge_idx

    edge_relations = np.array(edge_relations)
    edge_idx = np.array(edge_idx)
    filtered_idx = np.where(edge_relations!="on-top")
    edge_idx = edge_idx.T[filtered_idx].T
    edge_relations = edge_relations[filtered_idx]

    for e_i in range(len(edge_relations)):
        source_id = edge_idx[0][e_i]
        target_id = edge_idx[1][e_i]
        edge_relation = edge_relations[e_i]
        if graph.edges.get([source_id, target_id]) is not None:
            graph.edges.get([source_id, target_id])["edge_relations"] = f'{graph.edges.get([source_id, target_id])["edge_relations"]}\n{edge_relation}'
        else:
            source_node_label = graph.nodes.get(source_id)["label"]
            target_node_label = graph.nodes.get(target_id)["label"]
            graph.add_edge(source_id, target_id, labels=f"{source_node_label},{target_node_label}", edge_relations=edge_relation)
    return graph, pos

def visualize_osmnx_graph(graph, pos=None, w_edge_label=True, figsize=(20,20), node_size=1000, font_size=12, edge_font_size=12, node_color="#FFE3E3", title=None, show_plot=True):
    mapped_labels = {}
    for n in graph.nodes:
        node = graph.nodes.get(n)
        label = node["label"]
        mapped_labels[n] = label

    if pos is None:
        pos = nx.spring_layout(graph)  # Define layout for visualization
    plt.figure(figsize=figsize)

    # Draw nodes and edges
    nx.draw(graph, pos, with_labels=True, node_color=node_color, node_size=node_size, font_size=font_size, font_weight='bold', edge_color='black', labels=mapped_labels)

    if w_edge_label:
        # Draw edge labels (relationships)
        edge_labels = {(u, v): d['edge_relations'] for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black', font_size=edge_font_size)

    if title is not None:
        plt.gcf().canvas.manager.set_window_title(title)

    if show_plot:
        # Show the plot
        plt.show()

if __name__ == "__main__":
    args = get_args()

    with open(os.path.join(f"{config.scene_graphs_path}/{config.cell_graphs_file_name}"), "rb") as f:
        cell_scene_graphs = pickle.load(f)
    cell_graph_keys = list(cell_scene_graphs.keys())

    with open(os.path.join(f"{config.scene_graphs_path}/{config.train_text_graphs_file_name}"), "rb") as f:
        train_scene_graphs = pickle.load(f)
    with open(os.path.join(f"{config.scene_graphs_path}/{config.test_text_graphs_file_name}"), "rb") as f:
        test_scene_graphs = pickle.load(f)
    with open(os.path.join(f"{config.scene_graphs_path}/{config.val_text_graphs_file_name}"), "rb") as f:
        val_scene_graphs = pickle.load(f)
    text_scene_graphs = train_scene_graphs | test_scene_graphs | val_scene_graphs

    text_scene_graphs_copy = copy.deepcopy(text_scene_graphs)
    for k,v in text_scene_graphs_copy.items():
        cell_id = "_".join(k.split("_")[:2])
        if cell_id not in cell_graph_keys:
            del text_scene_graphs[k]
    text_scene_graph_keys = list(text_scene_graphs.keys())

    if (args.visualization_graph_index < 0 or args.visualization_graph_index >= len(text_scene_graph_keys)):
        print(f"visualization_graph_index is invalid. It must be a value between 0 and {len(text_scene_graph_keys)-1}")
        exit()

    text_scene_graph_key = text_scene_graph_keys[args.visualization_graph_index]
    text_scene_graph = text_scene_graphs[text_scene_graph_key]
    text_scene_graph_dict = {text_scene_graph_key: text_scene_graph}

    model_name = config.model_name
    model_state_dict = torch.load(f'{config.model_checkpoints_path}/{model_name}.pt')
    model = BigGNN(config.N, config.heads).to('cuda')
    model.load_state_dict(model_state_dict)

    model.eval()

    accuracy, cos_sims_dict, sorted_top_k_cell_ids = eval(model, text_scene_graph_dict, cell_scene_graphs, cell_graph_keys, config.top_ks_list)
    recalls = [0] * len(config.top_ks_list)
    for k,v in accuracy.items():
        for acc_i, acc in enumerate(v):
            if acc:
                recalls[acc_i] += 1
    recalls = np.array(recalls, dtype=float)
    recalls /= len(accuracy)
    print("Recalls", recalls.tolist())

    graph_tmp, pos_tmp = scene_graph_to_osmnx_graph(text_scene_graph)
    visualize_osmnx_graph(graph_tmp, pos_tmp, figsize=(10,10), node_size=2500, font_size=18, edge_font_size=18, title="Text scene graph", show_plot=False)

    cell_scene_graph_keys = list(cell_scene_graphs.keys())
    top_one_cell_scene_graph_key = sorted_top_k_cell_ids[0]
    cell_scene_graph = cell_scene_graphs[top_one_cell_scene_graph_key]
    graph_tmp, pos_tmp = scene_graph_to_osmnx_graph(cell_scene_graph)
    visualize_osmnx_graph(graph_tmp, pos_tmp, figsize=(10,10), node_size=2500, font_size=18, edge_font_size=18, title="OSM scene graph")

    print(text_scene_graph_key, top_one_cell_scene_graph_key)