import numpy as np
# import networkx as nx # To be removed
import random
import torch 

from sklearn.model_selection import train_test_split

np.random.seed(123)

# Helper function to convert NumPy adjacency matrix to edge_index
# This is a common operation when loading from formats that provide full matrices.
def numpy_adj_to_edge_index(adj_matrix):
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    # For unweighted graphs, non-zero entries indicate edges.
    # For weighted, ensure adj_matrix > 0 or some threshold if needed.
    rows, cols = np.where(adj_matrix > 0)
    edge_index = torch.from_numpy(np.vstack((rows, cols))).long()
    return edge_index

def load_graphs(dataset_str, time_step, device='cpu'):
    data_path = f"data/{dataset_str}/{dataset_str}.npz" # Assuming npz is named like dataset
    # Example: data_path=("data/{}/{}".format(dataset_str, "DBLP3.npz"))
    # If dataset_str is DBLP3, then DBLP3.npz. So f"data/{dataset_str}/{dataset_str}.npz"
    if dataset_str == "DBLP3": # Specific fix for DBLP3 filename from original
        data_path = f"data/{dataset_str}/DBLP3.npz"
    elif dataset_str == "DBLP5":
         data_path = f"data/{dataset_str}/DBLP5.npz"


    data_content = np.load(data_path, allow_pickle=True)
    
    adj_matrices_np = data_content['adjs']
    attribute_matrices_np = data_content['attmats']
    # labels_np = data_content['labels'] # Loaded in train.py

    # Store graphs as a list of dictionaries or simple objects
    # Each element: {'edge_index': tensor, 'num_nodes': int}
    # Adjacency matrices (for random walk perhaps) could be sparse tensors or (edge_index, edge_weight)
    graphs_repr = [] # List of (edge_index, num_nodes)
    adjs_edge_indices = [] # List of edge_indices
    
    # Assuming features are per-node, per-timestep
    # Or a single feature matrix if static: attribute_matrices_np[0]
    # The original code implies attribute_matrix[:,j,:] is for timestep j
    # So, attribute_matrices_np is likely (num_nodes, num_timesteps, num_features)
    # Or (num_timesteps, num_nodes, num_features) if loaded like that.
    # Original: attribute_matrix[:,j,:] implies (nodes, timesteps, features)
    # So feats_list[t] would be attribute_matrices_np[:, t, :]
    
    # Let's assume attribute_matrices_np is (num_timesteps, num_nodes, num_features) based on train.py usage
    # Data_dblp.append([torch.from_numpy(attribute_matrix[:,i,:])...]) seems to imply (nodes, timesteps, features)
    # If it's (nodes, timesteps, features), then attribute_matrices_np[:, t, :]
    # If it's (timesteps, nodes, features), then attribute_matrices_np[t, :, :]
    # From train.py: attribute_matrix = data_dblp['attmats']
    # Data_dblp.append([torch.from_numpy(attribute_matrix[:,i,:]).float().to(device) for i in range(args.time_steps)])
    # This confirms attribute_matrices_np is (num_nodes, num_timesteps, num_features)
    
    num_nodes_overall = attribute_matrices_np.shape[0]
    features_list = [] # List of feature tensors per timestep

    for i in range(time_step):
        adj_t = adj_matrices_np[i] # This is a NumPy adjacency matrix for timestep i
        edge_index_t = numpy_adj_to_edge_index(adj_t).to(device)
        num_nodes_t = adj_t.shape[0] # Or infer from max(edge_index_t) + 1
                                     # Safer to use adj_t.shape[0] if nodes can be isolated

        # Ensure num_nodes_t is consistent or use a global num_nodes
        # For DBLP like datasets, num_nodes is usually fixed.
        # If graph size changes, num_nodes_t is important.
        # Using num_nodes_overall from features if node set is consistent.
        
        graphs_repr.append({'edge_index': edge_index_t, 'num_nodes': num_nodes_overall})
        adjs_edge_indices.append(edge_index_t) # Storing edge_index for easier access if needed
                                            # The original 'adjs' were SciPy sparse matrices
                                            # This now stores the edge_index part.
                                            # If weights are needed, store (edge_index, edge_weight)
        
        # Extract features for timestep i
        # features_t = torch.from_numpy(attribute_matrices_np[:, i, :]).float().to(device)
        # features_list.append(features_t)
    
    # The original 'feats' in train.py was passed to MyDataset, which then preprocessed it.
    # MyDataset expects a list of feature matrices (one per snapshot, raw).
    # So, load_graphs should return this list of raw feature matrices.
    for i in range(time_step):
        # Assuming attribute_matrices_np is (num_nodes, num_timesteps, feature_dim)
        features_list.append(attribute_matrices_np[:, i, :]) # Keep as numpy for MyDataset to process

    # The original `load_graphs` returned G (nx graphs), adjs (scipy sparse), feats (raw np), data_dblp (raw npz content)
    # New return:
    # graphs_repr: list of {'edge_index': tensor, 'num_nodes': int}
    # adjs_for_minibatch: list of (edge_index_t, num_nodes_t) or similar for MyDataset's GCN norm
    #                      MyDataset uses self.adjs = [_normalize_graph_gcn(a) for a in adjs]
    #                      So, it needs something that _normalize_graph_gcn can process.
    #                      The original `adjs` were SciPy sparse matrices.
    #                      Let's pass the raw NumPy adjacency matrices for now, and MyDataset will convert.
    # features_list: list of raw NumPy feature arrays for MyDataset
    # data_content: raw npz content
    
    raw_adjs_np = [adj_matrices_np[i] for i in range(time_step)]

    return graphs_repr, raw_adjs_np, features_list, data_content


def get_context_pairs(graphs_repr, raw_adjs_np, time_step, num_walks=10, walk_len=20):
    # This function signature needs to align with what run_random_walks_n2v will expect.
    # run_random_walks_n2v will need edge_index, num_nodes, and optionally edge_weights.
    # raw_adjs_np[i] can be used to derive edge_index and possibly weights if present.
    print("Computing training pairs ...")
    context_pairs_train = []
    
    # This import is here because run_random_walks_n2v itself is in utils.utilities
    # and utils.utilities might import from utils.preprocess, creating a circular dependency risk
    # if imported at the top level. Best to refactor utilities.py first.
    # For now, assume it's available.
    from utils.utilities import run_random_walks_n2v

    for i in range(time_step):
        # graph_t_repr = graphs_repr[i] # Contains {'edge_index', 'num_nodes'}
        # adj_t_np = raw_adjs_np[i] # The NumPy adjacency matrix
        
        # run_random_walks_n2v needs to be adapted to take this.
        # It originally took an nx.Graph and a SciPy sparse adj.
        # New signature could be: run_random_walks_n2v(edge_index, num_nodes, edge_weights_np_array_or_None, num_walks, walk_len)
        
        # For now, we pass what Graph_RandomWalk would need if it were refactored:
        # edge_index from graphs_repr[i]['edge_index']
        # num_nodes from graphs_repr[i]['num_nodes']
        # and the raw_adjs_np[i] from which Graph_RandomWalk can extract weights if it needs to.
        
        current_graph_edge_index = graphs_repr[i]['edge_index']
        current_graph_num_nodes = graphs_repr[i]['num_nodes']
        
        # If run_random_walks_n2v expects weights, they can be extracted from raw_adjs_np[i]
        # For now, assume it can handle the raw adj matrix to get weights if needed,
        # or it's modified to take edge_index and optional edge_weights.
        # The original run_random_walks_n2v created an nx.Graph and set weights from the SciPy adj.
        # Let's pass the raw adj matrix for now, and random_walk.py will handle it.
        
        context_pairs_train.append(
            run_random_walks_n2v(
                graphs_repr[i], # Contains edge_index, num_nodes
                raw_adjs_np[i], # Raw numpy adj for weight extraction or direct use by a potentially new RW
                num_walks=num_walks,
                walk_len=walk_len
            )
        )
    return context_pairs_train

def get_evaluation_data(graphs_repr, time_step):
    eval_idx = time_step - 2
    eval_graph_repr = graphs_repr[eval_idx]    # {'edge_index', 'num_nodes'}
    next_graph_repr = graphs_repr[eval_idx+1]  # {'edge_index', 'num_nodes'}
    
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph_repr, next_graph_repr, val_mask_fraction=0.2,  test_mask_fraction=0.6)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(current_graph_repr, next_graph_repr, val_mask_fraction=0.2, test_mask_fraction=0.6):
    # current_graph_repr: {'edge_index', 'num_nodes'}
    # next_graph_repr: {'edge_index', 'num_nodes'}

    next_graph_edge_index = next_graph_repr['edge_index'].cpu().numpy().T # Shape (num_edges, 2)
    current_graph_num_nodes = current_graph_repr['num_nodes']

    edges_positive = []
    for edge in next_graph_edge_index:
        u, v = edge[0], edge[1]
        # Check if nodes exist in the current graph (nodes are 0 to num_nodes-1)
        if u < current_graph_num_nodes and v < current_graph_num_nodes:
            edges_positive.append([u, v]) # Store as list of pairs
    
    edges_positive_np = np.array(edges_positive)
    if edges_positive_np.shape[0] == 0: # No positive edges found
        print("Warning: No positive edges found for evaluation splits.")
        empty_array = np.empty((0,2), dtype=int)
        return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array

    edges_negative_np = negative_sample(edges_positive_np, next_graph_repr['num_nodes'], next_graph_repr['edge_index'])
    
    # Ensure consistent splitting even if one set is small
    test_val_size = val_mask_fraction + test_mask_fraction
    if len(edges_positive_np) * test_val_size < 1 or len(edges_negative_np) * test_val_size < 1 :
        print("Warning: Not enough positive or negative edges for requested validation/test split size. Adjusting.")
        # Fallback: use a very small test set or adjust fractions. For now, let train_test_split handle it.
        # Or, if critically small, return mostly training data.
        if len(edges_positive_np) <= 1 or len(edges_negative_np) <=1: # cannot split
             return edges_positive_np, edges_negative_np, np.empty((0,2)), np.empty((0,2)), np.empty((0,2)), np.empty((0,2))


    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(
        edges_positive_np, edges_negative_np, 
        test_size=min(test_val_size, 0.999), # Ensure test_size is < 1.0
        random_state=123
    )
    
    if len(test_pos) == 0 or len(test_neg) == 0 : # No data for val/test after first split
        return train_edges_pos, train_edges_neg, np.empty((0,2)), np.empty((0,2)), np.empty((0,2)), np.empty((0,2))


    # Calculate split fraction for the second split carefully
    if (test_mask_fraction + val_mask_fraction) == 0:
        val_test_split_ratio = 0.0 # Avoid division by zero
    elif len(test_pos) * (test_mask_fraction / (test_mask_fraction + val_mask_fraction)) < 1:
        val_test_split_ratio = 0.999 # if too small, give most to val
    else:
        val_test_split_ratio = test_mask_fraction / (test_mask_fraction + val_mask_fraction)
        val_test_split_ratio = min(val_test_split_ratio, 0.999) # ensure < 1.0


    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(
        test_pos, test_neg, 
        test_size=val_test_split_ratio,
        random_state=123
    )

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
            
def negative_sample(edges_pos_np, num_nodes_in_next_graph, next_graph_edge_index):
    # edges_pos_np: NumPy array of positive edges, shape (num_pos_edges, 2)
    # next_graph_edge_index: PyTorch tensor, shape (2, num_next_edges)
    
    edges_neg = []
    # For efficient lookup of existing edges in next_graph_edge_index
    # Create a set of tuples: {(u,v), (v,u)}
    existing_edges_set = set()
    next_graph_edges_np_T = next_graph_edge_index.cpu().numpy().T # (num_edges, 2)
    for u, v in next_graph_edges_np_T:
        existing_edges_set.add(tuple(sorted((u, v)))) # Store sorted to count (u,v) and (v,u) once

    num_positive_edges = len(edges_pos_np)
    max_attempts = num_positive_edges * 20 # Limit attempts to avoid infinite loops in dense graphs

    current_attempts = 0
    while len(edges_neg) < num_positive_edges and current_attempts < max_attempts:
        idx_i = np.random.randint(0, num_nodes_in_next_graph)
        idx_j = np.random.randint(0, num_nodes_in_next_graph)
        current_attempts +=1

        if idx_i == idx_j:
            continue
        
        # Check if (idx_i, idx_j) is an edge in the next graph
        if tuple(sorted((idx_i, idx_j))) in existing_edges_set:
            continue
        
        # Check for duplicates in edges_neg (less critical if sampling with replacement, but good for distinct negs)
        # This check can be slow if edges_neg is large. For performance, might skip if truly random sampling is primary.
        # if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg: # This is slow
        # A faster check for small number of neg samples:
        is_dup = False
        for neg_edge in edges_neg:
            if (neg_edge[0] == idx_i and neg_edge[1] == idx_j) or \
               (neg_edge[0] == idx_j and neg_edge[1] == idx_i):
                is_dup = True
                break
        if is_dup:
            continue
            
        edges_neg.append([idx_i, idx_j])
    
    if len(edges_neg) < num_positive_edges:
        print(f"Warning: Could only generate {len(edges_neg)} negative samples out of {num_positive_edges} requested.")
        # If still not enough, could try to fill with more random pairs, possibly allowing duplicates if desperate
        # or simply return what was found.

    return np.array(edges_neg)


def get_evaluation_classification_data(dataset_str, num_nodes, num_time_steps):
    # This function primarily deals with NumPy arrays and Python lists for indices.
    # No direct graph structure manipulation that needs networkx or scipy.sparse.
    # So, it should largely remain the same.
    
    # eval_idx = num_time_steps - 2 # Original: used for filename
    # eval_path = f'data/{dataset_str}/eval_nodeclassification_{eval_idx}.npz'

    train_ratios = [0.3, 0.5, 0.7]
    datas_splits_all_ratios = [] # list of (list of (list of indices))
                                # outer list: per ratio
                                # middle list: per timestep
                                # inner list: [train_idx, val_idx, test_idx]

    for ratio in train_ratios:
        splits_for_current_ratio = []
        for _ in range(num_time_steps): # Generate splits for each timestep
            all_node_indices = np.arange(num_nodes)
            
            # Ensure enough nodes for val split
            num_val_nodes = int(num_nodes * 0.25)
            if num_val_nodes == 0 and num_nodes > 0: num_val_nodes = 1 # At least 1 if possible
            if num_val_nodes > num_nodes: num_val_nodes = num_nodes
            
            idx_val = np.array(random.sample(list(all_node_indices), num_val_nodes), dtype=int)
            
            remaining_indices = np.setdiff1d(all_node_indices, idx_val)
            
            # Ensure enough remaining nodes for train split
            num_train_nodes = int(len(remaining_indices) * (ratio / (1-0.25) if (1-0.25) > 0 else ratio)) # scale ratio to remaining
            num_train_nodes = int(num_nodes * ratio) # Original was num_nodes*ratio, this might be more intuitive
            if num_train_nodes == 0 and len(remaining_indices) > 0: num_train_nodes = 1
            if num_train_nodes > len(remaining_indices): num_train_nodes = len(remaining_indices)
            
            idx_train = np.array(random.sample(list(remaining_indices), num_train_nodes), dtype=int)
            idx_test = np.setdiff1d(remaining_indices, idx_train)
            
            splits_for_current_ratio.append([idx_train, idx_val, idx_test])
        datas_splits_all_ratios.append(splits_for_current_ratio)
    
    # Original code saved this with np.savez(eval_path, data=np.array(datas_splits_all_ratios, dtype=object))
    # We will just return it. The train.py can decide to save or use it directly.
    return datas_splits_all_ratios