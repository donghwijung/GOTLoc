import pickle
import copy
import random

import torch
import torch.cuda
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

import config
from models import BigGNN
from scene_graph_candidates_extraction import proceed_candidates_extraction

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.current_device())

random.seed(42)

def eval(model, text_graphs, cell_graphs, cell_graph_keys, top_ks_list):
    accuracy = {}
    cos_sims_dict = {}
    for ttsg_i, (text_graph_scene_id,test_text_scene_graph) in tqdm(enumerate(text_graphs.items())):
        accuracy[ttsg_i] = [False] * len(top_ks_list)
        scene_name, cell_id, txt_id = text_graph_scene_id.split("_")
        scene_id = f"{scene_name}_{cell_id}"

        query = test_text_scene_graph

        query_subgraph = query

        cos_sims = []

        if config.CANDIDATES_EXTRACTION:
            sorted_top_k_cell_ids, x_node_ft, x_edge_idx, x_edge_ft = proceed_candidates_extraction(model, cell_graphs, query_subgraph)
            for cell_graph_key in sorted_top_k_cell_ids:
                db_subgraph = cell_graphs[cell_graph_key]

                p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()

                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))

                cos_sim = (1 - F.cosine_similarity(x_p, p_p, dim=0)).item()
                cos_sims.append(cos_sim)
            cos_sims = np.array(cos_sims)
            sorted_indices = np.argsort(cos_sims)

            cos_sims_dict[ttsg_i] = cos_sims

            for k_i, k in enumerate(top_ks_list):
                top_k_indices = sorted_indices[:k]
                for top_k_index in top_k_indices:
                    # global_cell_id = sorted_top_k_cell_ids[top_k_index]
                    if sorted_top_k_cell_ids[top_k_index] == scene_id:
                        accuracy[ttsg_i][k_i] = True
                        break
        else:
            x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
        
            for db in cell_graphs.values():
                db_subgraph = db
                p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()

                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))

                cos_sim = (1 - F.cosine_similarity(x_p, p_p, dim=0)).item()
                cos_sims.append(cos_sim)
            cos_sims = np.array(cos_sims)
            sorted_indices = np.argsort(cos_sims)

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
    return accuracy, cos_sims_dict

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
    accuracy, cos_sims_dict = eval(model, val_text_graphs, cell_graphs_copy, cell_graph_keys, config.top_ks_list)
    # with open(f"{current_path}/result_osm_transformer_val.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             "accuracy": accuracy,
    #             "cos_sims_dict": cos_sims_dict,
    #             "cell_graph_keys": cell_graph_keys,
    #             "config.top_ks_list": config.top_ks_list
    #         }, f
    #     )

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
    # print(len(test_text_graphs))
    accuracy, cos_sims_dict = eval(model, test_text_graphs, cell_graphs_copy, cell_graph_keys, config.top_ks_list)
    print(accuracy)
    # with open(f"{current_path}/result_osm_transformer_test.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             "accuracy": accuracy,
    #             "cos_sims_dict": cos_sims_dict,
    #             "cell_graph_keys": cell_graph_keys,
    #             "config.top_ks_list": config.top_ks_list
    #         }, f
    #     )