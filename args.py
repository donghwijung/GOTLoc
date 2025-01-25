import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualization_graph_index', type=int, default=0, help='The index for performing the scene retrieval and visualizing the result')
    args = parser.parse_args()
    return args