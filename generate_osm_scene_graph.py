import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append(f'{current_path}/../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')

from scene_graph import SceneGraph

from utils import point_to_elements, elements_to_graph, gps_to_utm

def gps_to_osm_graph(gps, search_range, dist_thres, is_cell_graph=True):
    gdf = point_to_elements(gps, search_range)
    if gdf is not None:
        osm_graph = elements_to_graph(gdf, gps, ego_yaw, dist_thres, theta_range, word2vec_dim, epsg, include_ego=False, rotate_centroid=False, is_cell_graph=is_cell_graph)
    else:
        osm_graph = None
    return osm_graph

if __name__ == "__main__":
    ## GPS to scene graph
    ### GPS value
    
    ### OSM request
    
    ### OSM scene graph generation
    random_gps_cnt = 1000
    save_iter = 50

    word2vec_dim = 300
    use_attributes = True

    cell_search_range = 50
    cell_dist_thres = cell_search_range
    text_search_range = 25
    text_dist_thres = text_search_range
    theta_range = np.deg2rad(180)
    sub_graph_size = 6

    ego_yaw = 0
    epsg = 26917
    random_range = 0.00009 * 1 ## degree (GPS)

    gps = np.array([])

    cell_graphs = {}
    cell_gps_keys = {}
    osm_graph = gps_to_osm_graph(gps, cell_search_range, cell_dist_thres, is_cell_graph=True)
    if osm_graph is not None and len(osm_graph["nodes"]) >= sub_graph_size:
        cell_graph = osm_graph
        cell_graph_scene_id = gps_id
        cell_scene_graph = SceneGraph(cell_graph_scene_id,
                                graph_type='scanscribe', 
                                graph = cell_graph,
                                embedding_type='word2vec',
                                use_attributes=use_attributes)
        # if len(cell_scene_graph.edge_idx[0]) > 0:
        cell_graphs[cell_graph_scene_id] = cell_scene_graph

        gps_key = f"{gps[0]},{gps[1]}"
        cell_gps_keys[cell_graph_scene_id] = gps_key

    if (gps_id + 1) % save_iter == 0 or gps_id == len(random_lat) - 1:
        with open(cell_graph_file_path, "wb") as f:
            pickle.dump(cell_graphs, f)
        
        with open(cell_gps_keys_file_path, "wb") as f:
            pickle.dump(cell_gps_keys, f)