import random

from tqdm import tqdm
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F

import config
from scene_graph_candidates_extraction import proceed_candidates_extraction, setup_db

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)

def cal_cossim(model, db_subgraph, x_node_ft, x_edge_idx, x_edge_ft):
    p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()

    x_p, p_p, _ = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                            torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                            torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))

    cos_sim = (1 - F.cosine_similarity(x_p, p_p, dim=0)).item()
    return cos_sim

def eval(model, text_graphs, cell_graphs, cell_graph_keys, top_ks_list):
    accuracy = {}
    cos_sims_dict = {}

    if config.use_candidates_extraction:
        client = setup_db(model, cell_graphs)
    else:
        client = None

    for ttsg_i, (text_graph_scene_id,test_text_scene_graph) in tqdm(enumerate(text_graphs.items())):
        accuracy[ttsg_i] = [False] * len(top_ks_list)
        scene_name, cell_id, txt_id = text_graph_scene_id.split("_")
        scene_id = f"{scene_name}_{cell_id}"

        query = test_text_scene_graph

        query_subgraph = query

        cos_sims = []

        if client is not None:
            sorted_top_k_cell_ids, x_node_ft, x_edge_idx, x_edge_ft = proceed_candidates_extraction(model, cell_graphs, query_subgraph, client)
            for cell_graph_key in sorted_top_k_cell_ids:
                db_subgraph = cell_graphs[cell_graph_key]
                cos_sim = cal_cossim(model, db_subgraph, x_node_ft, x_edge_idx, x_edge_ft)
                cos_sims.append(cos_sim)
            cos_sims = np.array(cos_sims)
            sorted_indices = np.argsort(cos_sims)

            cos_sims_dict[ttsg_i] = cos_sims

            for k_i, k in enumerate(top_ks_list):
                top_k_indices = sorted_indices[:k]
                for top_k_index in top_k_indices:
                    if sorted_top_k_cell_ids[top_k_index] == scene_id:
                        accuracy[ttsg_i][k_i] = True
                        break
        else:
            x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
        
            for db in cell_graphs.values():
                db_subgraph = db
                cos_sim = cal_cossim(model, db_subgraph, x_node_ft, x_edge_idx, x_edge_ft)
                cos_sims.append(cos_sim)
            cos_sims = np.array(cos_sims)
            sorted_indices = np.argsort(cos_sims)
            sorted_top_k_cell_ids = np.array(cell_graph_keys)[sorted_indices]

            cos_sims_dict[ttsg_i] = cos_sims

            for k_i, k in enumerate(top_ks_list):
                top_k_indices = sorted_indices[:k]
                for top_k_index in top_k_indices:
                    if cell_graph_keys[top_k_index] == scene_id:
                        accuracy[ttsg_i][k_i] = True
                        break

        if (ttsg_i + 1) % config.result_save_epoch == 0 or ttsg_i == len(text_graphs) - 1:
            recalls = [0] * len(top_ks_list)
            for k,v in accuracy.items():
                for acc_i, acc in enumerate(v):
                    if acc:
                        recalls[acc_i] += 1
            recalls = np.array(recalls, dtype=float)
            recalls /= len(accuracy)
    return accuracy, cos_sims_dict, sorted_top_k_cell_ids
