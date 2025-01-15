# Paths
data_path = "./data"
scene_graphs_path = f"{data_path}/scene_graphs"
model_checkpoints_path = f"{data_path}/model_checkpoints"

cell_graphs_file_name = "osm_cell_graphs_5225.pkl"
train_text_graphs_file_name = "osm_train_text_graphs_20648.pkl"
val_text_graphs_file_name = "osm_val_text_graphs_2671.pkl"
test_text_graphs_file_name = "osm_test_text_graphs_9205.pkl"

model_name = "model_osm" ## The name of the model checkpoints
top_ks_list = [1,3,5]
word2vec_dim = 300

# Train
epoch = 100
model_save_epoch = 5
lr = 0.0001 ## Learning rate
weight_decay = 5e-5
N = 1
batch_size = 16

heads = 2 ## The number of multi-head-attentions for graph transformers

valid_top_k = top_ks_list
folds = 10
skip_k_fold = True
loss_ablation_m = True ## use the cosine similarity only
loss_ablation_c = False ## use the matching probability only
eval_only_c = False ## use the cosine similarity only
continue_training = 0 ## Whether continue the training from the pre-saved model
continue_training_model = None ## The name of saved model
entire_training_set = False

eval_iters = 30
eval_iter_count = 10
out_of = 10

use_wandb = True ## Whether to use the wandb or not during training

# Evaluation
result_save_epoch = 50
use_candidates_extraction = True ## Please set this value as False, if you didn't install vectorDB (Milvus).

# Visualization
visualization_graphs_file_name = cell_graphs_file_name
visualization_graph_index = 0