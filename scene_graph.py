import random

import numpy as np
random.seed(3)

def check_valid_graph(nodes, edge_idx):
    assert(type(nodes) == dict)
    assert(type(edge_idx) == list)
    nodeids = [int(node_id) for node_id in nodes]
    nodeids = set(nodeids)
    edgeids = set()
    for from_idx, to_idx in zip(edge_idx[0], edge_idx[1]):
        edgeids.add(int(from_idx))
        edgeids.add(int(to_idx))

    return edgeids.issubset(nodeids)

class Node:
    def __init__(self, idx, label_features, attribute_features, use_attributes, label=None, attributes=None, centroid=None):
        self.idx = idx
        self.label_features = label_features
        self.attribute_features = attribute_features
        self.label = label
        self.attributes = attributes
        assert(type(self.attribute_features) == list)
        assert(type(self.label_features) == list or type(self.label_features) == np.ndarray)
        self.features = self.set_features(label_features, attribute_features, use_attributes=use_attributes)
        self.centroid = centroid

    def set_features(self, labels, attributes, use_attributes):
        if use_attributes: 
            attribute_features = np.zeros(len(labels))
            if attributes is not None and len(attributes) > 0:
                attribute_features = np.mean(attributes, axis=0)
            l = labels + attribute_features
            assert(len(l) == len(labels))
            return l
        else: return labels
    
class Edge:
    def __init__(self, from_idx, to_idx, features):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.features = features

class SceneGraph:
    def __init__(self, scene_id, txt_id=None, graph_type=None, graph=None, max_dist=None, embedding_type='ada', use_attributes=True):
        self.scene_id = scene_id
        if graph_type == '3dssg':
            self.nodes = self.extract_nodes_3dssg(graph['objects'], use_attributes, embedding_type)
            self.edge_idx, self.edge_relations, self.edge_features = self.extract_edges_3dssg(graph['edge_lists'], max_dist, embedding_type)
            assert(len(self.edge_idx[0]) == len(self.edge_features))
            assert(check_valid_graph(self.nodes, self.edge_idx))
        elif graph_type == 'scanscribe' or graph_type == 'human':
            self.nodes = self.extract_nodes_scanscribe(graph['nodes'], use_attributes, embedding_type)
            self.edge_idx, self.edge_relations, self.edge_features = self.extract_edges_scanscribe(graph['edges'], embedding_type)
            assert(len(self.edge_idx[0]) == len(self.edge_features))
            assert(check_valid_graph(self.nodes, self.edge_idx))
            self.txt_id = txt_id
        elif graph_type == None:
            self.nodes = None
            self.edge_idx = None
            self.edge_features = None
            self.scene_id = scene_id
            self.txt_id = txt_id

    def extract_nodes_3dssg(self, objects, use_attributes, embedding_type='ada'):
        nodes = {}
        for objid in objects:
            obj = objects[objid]
            attributes_list = [a for attr in obj['attributes_' + embedding_type] for a in obj['attributes_' + embedding_type][attr]]
            if len(attributes_list): assert(len(attributes_list[0]) == len(obj['label_' + embedding_type]))
            node = Node(int(obj['id']), obj['label_' + embedding_type], attributes_list, use_attributes=use_attributes,
                        label=obj['label'], attributes=obj['attributes'])
            nodes[int(obj['id'])] = node
        return nodes
    
    def extract_nodes_scanscribe(self, objects, use_attributes, embedding_type='ada'):
        nodes = {}
        for obj in objects:
            attributes_list = obj['attributes_' + embedding_type]['all']
            if len(attributes_list): assert(len(attributes_list[0]) == len(obj['label_' + embedding_type]))
            if "centroid" in obj:
                node = Node(int(obj['id']), obj['label_' + embedding_type], attributes_list, use_attributes=use_attributes,
                            label=obj['label'], attributes=obj['attributes'], centroid=obj["centroid"])
            else:
                node = Node(int(obj['id']), obj['label_' + embedding_type], attributes_list, use_attributes=use_attributes,
                    label=obj['label'], attributes=obj['attributes'])
            nodes[int(obj['id'])] = node
        return nodes

    def extract_edges_3dssg(self, edge_lists, max_dist, embedding_type='ada'):
        edge_idx = []
        from_edge = []
        to_edge = []
        edge_attributes = []
        edge_attributes_embedding = []
        for idx, d in enumerate(edge_lists['distance']):
            if d <= max_dist:
                from_edge.append(int(edge_lists['from'][idx]))
                to_edge.append(int(edge_lists['to'][idx]))
                edge_attributes.append(edge_lists['relation'][idx])
                edge_attributes_embedding.append(edge_lists['relation_' + embedding_type][idx])
        assert(len(from_edge) == len(to_edge))
        edge_idx.append(from_edge)
        edge_idx.append(to_edge)            
        return edge_idx, edge_attributes, edge_attributes_embedding
    
    def extract_edges_scanscribe(self, edges, embedding_type='ada'):
        edge_idx = []
        from_edge = []
        to_edge = []
        edge_attributes = []
        edge_attributes_embedding = []
        for idx in range(len(edges)):
            from_edge.append(int(edges[idx]['source']))
            to_edge.append(int(edges[idx]['target']))
            edge_attributes.append(edges[idx]['relationship'])
            edge_attributes_embedding.append(edges[idx]['relation_' + embedding_type])
        assert(len(from_edge) == len(to_edge))
        edge_idx.append(from_edge)
        edge_idx.append(to_edge)            
        return edge_idx, edge_attributes, edge_attributes_embedding

    def get_subgraph(self, node_ids, return_graph=False):
        assert(all([id in [int(n.idx) for n in self.nodes.values()] for id in node_ids]))
        if len(node_ids) == 0: return None
        subgraph_nodes = {}
        subgraph_node_features = []
        subgraph_edge_ids_from = []
        subgraph_edge_ids_to = []
        subgraph_edge_features = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if int(node.idx) in node_ids:
                subgraph_nodes[int(node.idx)] = node
                subgraph_node_features.append(node.features)
        for i, (from_idx, to_idx) in enumerate(zip(self.edge_idx[0], self.edge_idx[1])):
            if int(from_idx) in node_ids and int(to_idx) in node_ids:
                subgraph_edge_ids_from.append(int(from_idx))
                subgraph_edge_ids_to.append(int(to_idx))
                subgraph_edge_features.append(self.edge_features[i])
        subgraph_edge_ids = []
        subgraph_edge_ids.append(subgraph_edge_ids_from)
        subgraph_edge_ids.append(subgraph_edge_ids_to)
        assert(len(subgraph_edge_ids[0]) == len(subgraph_edge_ids[1]))
        assert(len(subgraph_edge_ids[0]) == len(subgraph_edge_features))
        assert(check_valid_graph(subgraph_nodes, subgraph_edge_ids))
        if return_graph: # Return a graph instead, used for data_distribution_analysis
            new_graph = SceneGraph(self.scene_id,
                                   graph_type=None,
                                   graph=None)
            new_graph.nodes = subgraph_nodes
            new_graph.edge_idx = subgraph_edge_ids
            new_graph.edge_features = subgraph_edge_features
            return new_graph
        return subgraph_nodes, subgraph_node_features, subgraph_edge_ids, subgraph_edge_features # nodes, node_features, edge_idx, edge_feature
    
    def to_pyg(self):
        assert(len(self.nodes) > 0)
        node_ids = [int(self.nodes[node_id].idx) for node_id in self.nodes]
        edge_ids = self.edge_idx

        nodeid_map = {}
        for idx, nodeid in enumerate(node_ids):
            nodeid_map[int(nodeid)] = idx
        
        edge_ids_remap = []
        edge_ids_from = []
        edge_ids_to = []
        for (from_idx, to_idx) in zip(edge_ids[0], edge_ids[1]):
            edge_ids_from.append(int(nodeid_map[int(from_idx)]))
            edge_ids_to.append(int(nodeid_map[int(to_idx)]))
        edge_ids_remap.append(edge_ids_from)
        edge_ids_remap.append(edge_ids_to)

        node_features = [self.nodes[node_id].features for node_id in self.nodes]
        return node_features, edge_ids_remap, self.edge_features # node_features, edge_idx, edge_features

    def get_node_features(self):
        node_features = [self.nodes[node_id].features for node_id in self.nodes]
        return node_features