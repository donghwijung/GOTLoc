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

if __name__ == "__main__":
    ## text description to scene graph
    ### text sentence input

    ### parsing texts

    ### text scene graph generation
    max_try_cnt = 5

    text_graphs = {}
    # cell_graphs = {}
    text_gps_keys = {}
    # for gps_id in tqdm(range(len(random_lat))):

    #     cell_lat = random_lat[gps_id]
    #     cell_lon = random_lon[gps_id]

        # gps_key = f"{cell_lat},{cell_lon}"
        # if gps_id in cell_graphs.keys():
            # gps = np.array([cell_lat, cell_lon])
            # osm_graph = gps_to_osm_graph(gps)
            # if osm_graph is None:
    text_osm_graph = None
    try_cnt = 0
    skip_current_cell = False
    while text_osm_graph is None:
        if try_cnt > max_try_cnt:
            skip_current_cell = True
            break
        # lat, lon = get_random_gps_within_city(
        #     paris_bbox['lat_min'], paris_bbox['lat_max'],
        #     paris_bbox['lon_min'], paris_bbox['lon_max'],
        #     1
        # )
        lat = cell_lat + np.random.uniform(-1,1) * random_range
        lon = cell_lon + np.random.uniform(-1,1) * random_range
        # gps_key = f"{lat[0]},{lon[0]}"
        # if gps_key not in cell_graphs:
        gps = np.array([lat, lon])
        text_osm_graph = gps_to_osm_graph(gps, text_search_range, text_dist_thres, is_cell_graph=False)
        try_cnt += 1
        # else:
        #     continue
    # if skip_current_cell:
    #     skip_current_cell = False
    #     print(f"skipped cell {gps_id}")
    #     continue
    # cell_graph = osm_graph
    # # cell_graph_scene_id = gps_key
    # cell_graph_scene_id = gps_id
    # cell_scene_graph = SceneGraph(cell_graph_scene_id,
    #                         graph_type='scanscribe', 
    #                         graph = cell_graph,
    #                         embedding_type='word2vec',
    #                         use_attributes=use_attributes)
    # # if len(cell_scene_graph.edge_idx[0]) > 0:
    # cell_graphs[cell_graph_scene_id] = cell_scene_graph

    # lat, lon = gps

    # ego_yaw = 0
    # gdf = point_to_elements(gps, search_range)
    # if gdf is not None:
    #     osm_graph = elements_to_graph(gdf, gps_to_utm(lon, lat), ego_yaw, dist_thres, theta_range, word2vec_dim, include_ego=False, rotate_centroid=False)
    # else:
    #     osm_graph = None
    #     continue
    
    # sub_osm_graph = osm_graph_to_sub_graph(cell_graph, sub_graph_size)
    if len(text_osm_graph["nodes"]) >= sub_graph_size:
        txt_id = 0
        cell_id = gps_id

        text_graph_scene_id = f"{cell_id}_{str(txt_id).zfill(5)}"
        text_scene_graph = SceneGraph(cell_id,
                                txt_id=txt_id,
                                graph_type='scanscribe', 
                                graph = text_osm_graph,
                                embedding_type='word2vec',
                                use_attributes=use_attributes)
        text_graphs[text_graph_scene_id] = text_scene_graph

        gps_key = f"{gps[0]},{gps[1]}"
        text_gps_keys[text_graph_scene_id] = gps_key

    if (gps_id + 1) % save_iter == 0 or gps_id == len(random_lat) - 1:               
        with open(text_graph_file_path, "wb") as f:
            pickle.dump(text_graphs, f)
            # with open(text_gps_keys_file_path, "wb") as f:
            #     pickle.dump(text_gps_keys, f)
    #         # print(gps_id, text_graphs.keys(), text_gps_keys.keys())
    # cell_ids_list = []
    # for text_graph_scene_id in text_graphs.keys():
    #     cell_id = int(text_graph_scene_id.split("_")[0])
    #     cell_ids_list.append(cell_id)
    # cell_graph_ids = list(cell_graphs.keys())
    # for cell_id in cell_graph_ids:
    #     if cell_id not in cell_ids_list:
    #         del cell_graphs[cell_id]
    #         del cell_gps_keys[cell_id]

    # with open(cell_graph_file_path, "wb") as f:
    #     pickle.dump(cell_graphs, f)
    
    # with open(cell_gps_keys_file_path, "wb") as f:
    #     pickle.dump(cell_gps_keys, f)