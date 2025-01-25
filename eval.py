import pickle
import copy
import random

import numpy as np

import torch
import torch.cuda

import config
from eval_utils import eval
from models import BigGNN

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)

if __name__ == '__main__':
    model_name = config.model_name
    model_state_dict = torch.load(f'{config.model_checkpoints_path}/{model_name}.pt')
    model = BigGNN(config.N, config.heads).to('cuda')
    model.load_state_dict(model_state_dict)

    model.eval()

    with open(f"{config.scene_graphs_path}/{config.cell_graphs_file_name}", "rb") as f:
        cell_graphs = pickle.load(f)
    cell_graph_keys = list(cell_graphs.keys())

    with open(f"{config.scene_graphs_path}/{config.val_text_graphs_file_name}", "rb") as f:
        val_text_graphs = pickle.load(f)

    val_graph_keys = val_text_graphs.keys()
    val_graph_keys = ["_".join(k.split("_")[:2]) for k in val_graph_keys]
    val_graph_keys = list(set(val_graph_keys))
    val_scene_ids = list(set([k.split("_")[0] for k in val_graph_keys]))

    cell_graphs_copy = copy.deepcopy(cell_graphs)
    for k,v in cell_graphs.items():
        seq_name, seq_cell_id = k.split("_")
        if seq_name not in val_scene_ids:
            del cell_graphs_copy[k]
    cell_graph_keys = list(cell_graphs_copy.keys())
    accuracy, cos_sims_dict, _ = eval(model, val_text_graphs, cell_graphs_copy, cell_graph_keys, config.top_ks_list)
    recalls = [0] * len(config.top_ks_list)
    for k,v in accuracy.items():
        for acc_i, acc in enumerate(v):
            if acc:
                recalls[acc_i] += 1
    recalls = np.array(recalls, dtype=float)
    recalls /= len(accuracy)
    print("Val recalls", recalls.tolist())

    with open(f"{config.scene_graphs_path}/{config.test_text_graphs_file_name}", "rb") as f:
        test_text_graphs = pickle.load(f)

    test_graph_keys = test_text_graphs.keys()
    test_graph_keys = ["_".join(k.split("_")[:2]) for k in test_graph_keys]
    test_graph_keys = list(set(test_graph_keys))
    test_scene_ids = list(set([k.split("_")[0] for k in test_graph_keys]))

    cell_graphs_copy = copy.deepcopy(cell_graphs)
    for k,v in cell_graphs.items():
        seq_name, seq_cell_id = k.split("_")
        if seq_name not in test_scene_ids:
            del cell_graphs_copy[k]
    cell_graph_keys = list(cell_graphs_copy.keys())
    accuracy, cos_sims_dict, _ = eval(model, test_text_graphs, cell_graphs_copy, cell_graph_keys, config.top_ks_list)
    recalls = [0] * len(config.top_ks_list)
    for k,v in accuracy.items():
        for acc_i, acc in enumerate(v):
            if acc:
                recalls[acc_i] += 1
    recalls = np.array(recalls, dtype=float)
    recalls /= len(accuracy)
    print("Test recalls", recalls.tolist())