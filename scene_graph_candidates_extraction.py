import numpy as np

import torch
from torch.functional import F

from pymilvus import MilvusClient

def setup_db(model, cell_graphs):
    client = MilvusClient("GOTPR_milvus.db")

    cell_embedding_data_list = []
    data = []
    random_cell_graph_keys = list(cell_graphs.keys())
    for rcgk_i, rcgk in enumerate(random_cell_graph_keys):
        p_node_ft, p_edge_idx, p_edge_ft = cell_graphs[rcgk].to_pyg()

        cell_embedding = model.TSALayers[0](torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                    torch.tensor(np.array(p_edge_idx), dtype=torch.int64).to('cuda'),
                    edge_attr=torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
        cell_embedding_pooled = torch.mean(cell_embedding, dim=0)

        cell_embedding_data_list.append(cell_embedding_pooled)
        data.append({
            "id": rcgk_i,
            "vector": cell_embedding_pooled 
        })

    if client.has_collection(collection_name="GOTPR_collection"):
        client.drop_collection(collection_name="GOTPR_collection")
    client.create_collection(
        collection_name="GOTPR_collection",
        dimension=list(cell_embedding_data_list[0].size())[0]
    )
    _ = client.insert(collection_name="GOTPR_collection", data=data)

    return client

def proceed_candidates_extraction(model, cell_graphs, text_graph, client, candidates_cnt=10):
    random_cell_graph_keys = list(cell_graphs.keys())

    x_node_ft, x_edge_idx, x_edge_ft = text_graph.to_pyg()
    test_text_embedding = model.TSALayers[0](torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'),
                    torch.tensor(np.array(x_edge_idx), dtype=torch.int64).to('cuda'),
                    edge_attr=torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'))
    test_text_embedding_pooled = torch.mean(test_text_embedding, dim=0)

    query_vectors = [test_text_embedding_pooled.cpu().detach().numpy()]

    res = client.search(
        collection_name="GOTPR_collection",  # Target collection
        data=query_vectors,  # Query vectors
        limit=candidates_cnt,  # Number of returned entities
        search_params={
            "metric_type": "COSINE"
        }
    )

    top_k_cell_ids = []
    cos_sims = []
    for r in res[0]:
        cell_id = random_cell_graph_keys[r["id"]]
        top_k_cell_ids.append(cell_id)
        db_subgraph = cell_graphs[cell_id]
        p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()

        x_p, p_p, _ = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                    torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                    torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
        
        cos_sim = (1 - F.cosine_similarity(x_p, p_p, dim=0)).item()
        cos_sims.append(cos_sim)
    cos_sims = np.array(cos_sims)
    sorted_indices = np.argsort(cos_sims)
    top_k_cell_ids = np.array(top_k_cell_ids)
    sorted_top_k_cell_ids = top_k_cell_ids[sorted_indices]
    return sorted_top_k_cell_ids, x_node_ft, x_edge_idx, x_edge_ft