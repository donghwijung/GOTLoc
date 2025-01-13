import numpy as np

data_path = "./data"
# kitti360pose_path = f"{data_path}/kitti360pose/k360_30-10_scG_pd10_pc4_spY_all"
# scene_graphs_path = f"{data_path}/scene_graphs"
# model_checkpoints_path = f"{data_path}/model_checkpoints"
# kitti360pose_path = f"/media/donghwijung/t7_4tb/donghwi/data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all"
scene_graphs_path = f"{data_path}/scene_graphs"
model_checkpoints_path = f"{data_path}/model_checkpoints"

model_name = "model_osm_transformer_whole_data"

# SCENE_NAMES = [
#     "2013_05_28_drive_0000_sync",
#     "2013_05_28_drive_0002_sync",
#     "2013_05_28_drive_0003_sync",
#     "2013_05_28_drive_0004_sync",
#     "2013_05_28_drive_0005_sync",
#     "2013_05_28_drive_0006_sync",
#     "2013_05_28_drive_0007_sync",
#     "2013_05_28_drive_0009_sync",
#     "2013_05_28_drive_0010_sync",
# ]
# SCENE_NAMES_TRAIN = [
#     "2013_05_28_drive_0000_sync",
#     "2013_05_28_drive_0002_sync",
#     "2013_05_28_drive_0004_sync",
#     "2013_05_28_drive_0006_sync",
#     "2013_05_28_drive_0007_sync",
# ]
# SCENE_NAMES_VAL = [
#     "2013_05_28_drive_0010_sync",
# ]
# SCENE_NAMES_TEST = [
#     "2013_05_28_drive_0003_sync",
#     "2013_05_28_drive_0005_sync",
#     "2013_05_28_drive_0009_sync",
# ]

CANDIDATES_EXTRACTION = True ## Please set this value as False, if you didn't install vectorDB (Milvus).

cell_graphs_file_name = "osm_cell_graphs_0000_0002_0003_0004_0005_0006_0007_0009_0010_5225_300.pkl"
train_text_graphs_file_name = "osm_train_text_graphs_0000_0002_0004_0006_0007_20648_300.pkl"
val_text_graphs_file_name = "osm_val_text_graphs_0010_2671_300.pkl"
test_text_graphs_file_name = "osm_test_text_graphs_0003_0005_0009_9205_300.pkl"

top_ks_list = [1,3,5]

# tags = {"amenity": True, "railway": True, "public_transport": True, "building": True,\
#          "landuse": True, "leisure": True, "natural": True, "highway": True, "shop": True,\
#               "healthcare": True, "waterway": True, "barrier": True, "cycleway": True}

# er = 6378137. # Average earth radius at the equator

# Origin in OXTS coordinate
# origin_oxts = [48.9843445, 8.4295857] # lake in Karlsruhe

word2vec_dim = 300

# Train
search_range = 10
dist_thres = 50
theta_range = np.deg2rad(180)
sub_graph_size = 6
model_save_epoch = 5

epoch = 100
lr = 0.0001
weight_decay = 5e-5
N = 1
batch_size = 16

contrastive_loss = True
valid_top_k = [1, 2, 3, 5]
use_attributes = True
training_with_cross_val = True
folds = 10
skip_k_fold = True
subgraph_ablation = True
loss_ablation_m = True
loss_ablation_c = False
eval_only_c = False
eval_entire_dataset = False
continue_training = 0
continue_training_model = None
entire_training_set = False

eval_iters = 30
eval_iter_count = 10
out_of = 10

heads = 2

# eval
result_save_epoch = 50