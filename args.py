import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='online')
    # parser.add_argument('--epoch', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--weight_decay', type=float, default=5e-5)
    # parser.add_argument('--N', type=int, default=1)
    # parser.add_argument('--overlap_thr', type=float, default=0.8)
    # parser.add_argument('--cos_sim_thr', type=float, default=0.5)
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--training_set_size', type=int, default=2847)
    # parser.add_argument('--test_set_size', type=int, default=712)
    # parser.add_argument('--graph_size_min', type=int, default=4, help='minimum number of nodes in a graph')
    # parser.add_argument('--contrastive_loss', type=bool, default=True)
    # parser.add_argument('--valid_top_k', nargs='+', type=int, default=[1, 2, 3, 5])
    # parser.add_argument('--use_attributes', type=bool, default=True)
    # parser.add_argument('--training_with_cross_val', type=bool, default=True)
    # parser.add_argument('--folds', type=int, default=5)
    # parser.add_argument('--skip_k_fold', type=bool, default=False)
    # parser.add_argument('--entire_training_set', action='store_true')
    # parser.add_argument('--subgraph_ablation', action='store_true')

    # parser.add_argument('--eval_iters', type=int, default=100)
    # parser.add_argument('--eval_iter_count', type=int, default=10)
    # parser.add_argument('--out_of', type=int, default=10)
    
    # parser.add_argument('--model_name', type=str, default=None)
    # parser.add_argument('--loss_ablation_m', action='store_true')
    # parser.add_argument('--loss_ablation_c', action='store_true')
    # parser.add_argument('--eval_only_c', action='store_true')
    # parser.add_argument('--continue_training', type=int, default=0)
    # parser.add_argument('--continue_training_model', type=str, default=None)

    # parser.add_argument('--eval_entire_dataset', action='store_true')
    # parser.add_argument('--heads', type=int, default=2)

    # parser.add_argument('--scannet', action='store_true')
    # parser.add_argument('--scanscribe_auto_gen', action='store_true')

    # parser.add_argument('--word2vec_dim', type=int, default=300)
    args = parser.parse_args()
    return args