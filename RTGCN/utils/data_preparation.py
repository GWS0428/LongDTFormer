import os
import pandas as pd
import numpy as np
import pickle
import random
from scipy.sparse import coo_matrix

# --- Configuration ---
ORIGINAL_DATASET_NAME = 'CollegeMsg' 
TARGET_DATASET_NAME = 'CollegeMsg_RTGCN'
DEFAULT_NUM_SNAPSHOTS = 29 

# Path for the original role data PKL file.
# This is typically 'merged_snapshot_factorized_roles_.pkl' or similar.
ORIGINAL_ROLE_PKL_FILENAME = 'merged_snapshot_factorized_roles.pkl'

# --- Paths setup (mimicking the grandparent_dir structure from the first script) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir)) 

# --- Data Loading and Processing ---
def convert_link_prediction_data_to_project_format(
    original_dataset_name: str,
    target_dataset_name: str,
    num_snapshots: int,
    original_role_pkl_filename: str,
    grandparent_dir: str
):
    """
    Converts data from the original link prediction format to the target project's format.

    Args:
        original_dataset_name (str): The base name of the original dataset files (e.g., 'uci').
        target_dataset_name (str): The desired base name for the converted output files (e.g., 'UCI_Converted').
        num_snapshots (int): The number of time snapshots to discretize the data into.
                              This should ideally match what was used when generating the original role data.
        original_role_pkl_filename (str): The filename of the original role data pickle file.
        grandparent_dir (str): The base directory containing 'processed_data' and 'output'.
    """

    print(f"Starting conversion for dataset: {original_dataset_name}...")
    print(f"Target format name: {target_dataset_name}")
    print(f"Using {num_snapshots} snapshots for discretization.")

    # 1. Load original data files
    graph_df_path = os.path.join(grandparent_dir, 'data', original_dataset_name, f'ml_{original_dataset_name}.csv')
    edge_raw_features_path = os.path.join(grandparent_dir, 'data', original_dataset_name, f'ml_{original_dataset_name}.npy')
    node_raw_features_path = os.path.join(grandparent_dir, 'data', original_dataset_name, f'ml_{original_dataset_name}_node.npy')
    node_role_data_path = os.path.join(grandparent_dir, 'data', original_dataset_name, original_role_pkl_filename)

    if not os.path.exists(graph_df_path):
        print(f"Error: CSV file not found at {graph_df_path}")
        return
    if not os.path.exists(edge_raw_features_path):
        print(f"Error: Edge features NPY not found at {edge_raw_features_path}")
        return
    if not os.path.exists(node_raw_features_path):
        print(f"Error: Node features NPY not found at {node_raw_features_path}")
        return
    if not os.path.exists(node_role_data_path):
        print(f"Error: Node role PKL not found at {node_role_data_path}")
        print("Please ensure the original role data has been generated and saved.")
        return

    graph_df = pd.read_csv(graph_df_path)
    edge_raw_features = np.load(edge_raw_features_path)
    node_raw_features = np.load(node_raw_features_path)

    with open(node_role_data_path, "rb") as f:
        node_role_data_original = pickle.load(f)

    print("Original data loaded.")

    # 2. Preprocessing from the first script (feature padding and snapshot creation)
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172 # Dimensions from the original script
    
    # Pad node features if necessary
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    
    # Pad edge features if necessary (though not directly used in the .npz output for 'adjs')
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    # Discretize timestamps into snapshots, matching the original script's logic
    min_ts = graph_df['ts'].min()
    max_ts = graph_df['ts'].max()
    if num_snapshots == 0:
        raise ValueError("num_snapshots cannot be zero.")
    range_size = (max_ts - min_ts) / num_snapshots if num_snapshots > 0 else 1 # Avoid div by zero if only one snapshot

    graph_df['snapshots'] = ((graph_df['ts'] - min_ts) / range_size).astype(np.int16) + 1
    # Cap snapshots at num_snapshots, as per original script
    graph_df.loc[graph_df['snapshots'] > num_snapshots, 'snapshots'] = num_snapshots
    graph_df.loc[graph_df['snapshots'] < 1, 'snapshots'] = 1 # Ensure minimum snapshot is 1
    print(f"Snapshots created from timestamps (min_ts={min_ts}, max_ts={max_ts}, num_snapshots={num_snapshots}).")

    # Determine overall number of nodes
    all_unique_nodes = pd.concat([graph_df['u'], graph_df['i']]).unique()
    num_nodes_overall = int(all_unique_nodes.max()) + 1 # +1 because node IDs are 0-indexed or start from 1
    print(f"Detected total unique nodes: {num_nodes_overall}")

    # 3. Generate 'adjs' for .npz (list of adjacency matrices)
    adjs_list = []
    for t_idx in range(1, num_snapshots + 1): # Snapshots are 1-indexed in the original script
        snapshot_edges_df = graph_df[graph_df['snapshots'] == t_idx]
        
        # Create a COO matrix for memory efficiency, then convert to dense if required by target
        # Ensure node IDs are within bounds for array indexing
        u_nodes = snapshot_edges_df.u.values.astype(np.longlong)
        i_nodes = snapshot_edges_df.i.values.astype(np.longlong)
        
        # Create a sparse adjacency matrix first
        data = np.ones(len(u_nodes), dtype=np.int8)
        adj_t_coo = coo_matrix((data, (u_nodes, i_nodes)), shape=(num_nodes_overall, num_nodes_overall), dtype=np.int8)
        
        # Make it symmetric for an undirected graph 
        adj_t_coo = adj_t_coo + adj_t_coo.transpose()
        adj_t_coo.data[adj_t_coo.data > 0] = 1 # Ensure binary (0 or 1)

        adjs_list.append(adj_t_coo.toarray()) # Convert to dense array
    
    adjs_np = np.array(adjs_list) # Shape: (num_snapshots, num_nodes, num_nodes)
    print(f"Generated adjacency matrices: {adjs_np.shape}")

    # 4. Generate 'attmats' for .npz (node features over time)
    # `node_raw_features` shape: (num_nodes_in_source_file, NODE_FEAT_DIM)
    # `attmats` required shape: (num_nodes_overall, num_snapshots, NODE_FEAT_DIM)
    
    node_feat_dim = node_raw_features.shape[1]
    attribute_matrices_np = np.zeros((num_nodes_overall, num_snapshots, node_feat_dim), dtype=node_raw_features.dtype)
    
    # Fill in features for nodes that exist in `node_raw_features`
    existing_node_count_in_feat_file = node_raw_features.shape[0]
    
    if existing_node_count_in_feat_file > num_nodes_overall:
        print(f"Warning: Node feature file has more nodes ({existing_node_count_in_feat_file}) than detected unique nodes in graph ({num_nodes_overall}). Truncating features.")
        attribute_matrices_np[:, :, :] = np.expand_dims(node_raw_features[:num_nodes_overall, :], axis=1).repeat(num_snapshots, axis=1)
    else:
        attribute_matrices_np[:existing_node_count_in_feat_file, :, :] = np.expand_dims(node_raw_features, axis=1).repeat(num_snapshots, axis=1)
    
    print(f"Generated attribute matrices: {attribute_matrices_np.shape}")

    # 5. Generate 'labels' for .npz (placeholder)
    # The original `load_graphs` function doesn't seem to use `labels_np` for graph construction,
    # but it is part of the `data_content` npz.
    # NOTE(wsgwak): Check if node label exists in the original dataset.
    labels_np = np.zeros(num_nodes_overall, dtype=np.int16) 
    print(f"Generated placeholder labels: {labels_np.shape}")

    # 6. Convert role data from original format to target project format
    # Original: Dict[time_step: Dict[node_id: role_id]]
    # Target: Dict[time_step: Dict[role_id: List[node_id]]]
    converted_roles_data = {}
    for time_step, node_roles_dict in node_role_data_original.items():
        if time_step not in converted_roles_data:
            converted_roles_data[time_step] = {}
        for node_id, role_id in node_roles_dict.items():
            # Ensure role_id and node_id are suitable types (e.g., int)
            # The role_id might be a string in the original pkl, convert to int if needed.
            # The node_id is int already from node_role_data_pd processing.
            role_id = int(role_id) 
            node_id = int(node_id)
            if role_id not in converted_roles_data[time_step]:
                converted_roles_data[time_step][role_id] = []
            converted_roles_data[time_step][role_id].append(node_id)
    print("Converted role data format.")

    # 7. Save converted data
    output_dir = os.path.join(grandparent_dir, 'data', target_dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    npz_output_path = os.path.join(output_dir, f'{target_dataset_name}.npz')
    role_pkl_output_path = os.path.join(output_dir, f'{target_dataset_name}_wl_nc.pkl')

    np.savez_compressed(
        npz_output_path,
        adjs=adjs_np,
        attmats=attribute_matrices_np,
        labels=labels_np # Include labels for completeness, even if unused by load_graphs
    )
    print(f"Saved main graph data to: {npz_output_path}")

    with open(role_pkl_output_path, 'wb') as f:
        pickle.dump(converted_roles_data, f)
    print(f"Saved converted role data to: {role_pkl_output_path}")

    print(f"Conversion complete for {original_dataset_name} to {target_dataset_name} format!")

# --- Main execution ---
if __name__ == "__main__":
    # Ensure this script is placed appropriately relative to 'processed_data' and 'output'
    # For example, if your project structure is:
    # project_root/
    # ├── processed_data/
    # │   └── uci/
    # ├── output/
    # │   └── uci/
    # ├── data/ (will be created for new data)
    # └── scripts/
    #     └── this_conversion_script.py
    # Then `grandparent_dir` as calculated above should be `project_root`.

    # Example call:
    try:
        convert_link_prediction_data_to_project_format(
            original_dataset_name=ORIGINAL_DATASET_NAME,
            target_dataset_name=TARGET_DATASET_NAME,
            num_snapshots=DEFAULT_NUM_SNAPSHOTS,
            original_role_pkl_filename=ORIGINAL_ROLE_PKL_FILENAME,
            grandparent_dir=grandparent_dir
        )
    except Exception as e:
        print(f"An error occurred during conversion: {e}")