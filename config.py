# Paths
data_path = "./data"
scene_graphs_path = f"{data_path}/scene_graphs"
model_checkpoints_path = f"{data_path}/model_checkpoints"
results_path = "./results"

cell_graphs_file_name = "osm_cell_graphs_0000_0002_0003_0004_0005_0006_0007_0009_0010_5225_300.pkl"
train_text_graphs_file_name = "osm_train_text_graphs_0000_6625_300.pkl"
val_text_graphs_file_name = "osm_val_text_graphs_0010_2671_300.pkl"
test_text_graphs_file_name = "osm_test_text_graphs_0003_653_300.pkl"


model_name = "model_osm_transformer_whole_data" ## the name of the model checkpoints
top_ks_list = [1,3,5]
word2vec_dim = 300

# Train
epoch = 100
model_save_epoch = 5
lr = 0.0001 ## learning rate
weight_decay = 5e-5
N = 1
batch_size = 16

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

heads = 2 ## The number of multi-head-attentions for graph transformers

use_wandb = True ## Whether to use the wandb or not during training

# eval
result_save_epoch = 50
use_candidates_extraction = True ## Please set this value as False, if you didn't install vectorDB (Milvus).