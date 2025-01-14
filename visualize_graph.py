import os
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import config

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

def visualize_osmnx_graph(graph, pos=None, w_edge_label=True, figsize=(20,20), node_size=1000, font_size=12, edge_font_size=12, node_color="#FFE3E3"):
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

    # Show the plot
    plt.show()

if __name__ == "__main__":
    with open(os.path.join(f"{config.scene_graphs_path}/{config.visualization_graphs_file_name}"), "rb") as f:
        scene_graphs = pickle.load(f)
    scene_graph_key = list(scene_graphs.keys())[config.visualization_graph_index]
    scene_graph = scene_graphs[scene_graph_key]
    graph_tmp, pos_tmp = scene_graph_to_osmnx_graph(scene_graph)
    visualize_osmnx_graph(graph_tmp, pos_tmp, figsize=(10,10), node_size=2500, font_size=18, edge_font_size=18)