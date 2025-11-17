'''Global weighted combination approach for MoE selection'''

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from utils1 import *

BN = True

class GatingNetwork(nn.Module): 
    def __init__(self, input_dim, num_experts, top_k, coef): 
        super().__init__() 
        self.input_dim = input_dim 
        self.num_experts = num_experts 
        self.top_k = top_k 
        self.coef = coef # Load balancing loss coefficient 
        self.fc_gate = nn.Linear(input_dim, num_experts) 
        self.softmax = nn.Softmax(dim=1) 
        
        # For noisy top-k, similar to GMoE's MoE layer 
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts)) 
        nn.init.zeros_(self.w_noise) 
        self.softplus = nn.Softplus() 


    def noisy_top_k(self, x_for_gating, is_training, noise_epsilon=1e-2): 
        clean_logits = self.fc_gate(x_for_gating) # [N, num_experts] 

        if is_training and self.num_experts > self.top_k: 
            raw_noise_stddev = x_for_gating @ self.w_noise 
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon 
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev) 
            logits = noisy_logits 
        else: 
            logits = clean_logits 

        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.num_experts), dim=1) 
        top_k_logits = top_logits[:, :self.top_k] 
        top_k_indices = top_indices[:, :self.top_k] 
        top_k_gates_sm = self.softmax(top_k_logits) # Softmax over top-k logits [N, k] 

        # Scatter these k gates back to [N, num_experts]
        gates = torch.zeros_like(logits) 
        gates.scatter_(1, top_k_indices, top_k_gates_sm) 
        
        # Load balancing loss (CV squared)
        importance = gates.sum(0) # [num_experts], sum of gate values for each expert 
        # load = (gates > 0).sum(0) # [num_experts], how many examples routed to each expert 
                                    # For soft gating, load could be sum of gate values too, or based on top-k selection
        actual_k_for_load = min(self.top_k, self.num_experts) ### updated
        top_k_indices_for_load = logits.topk(actual_k_for_load, dim=1)[1] # Shape: [N, actual_k_for_load] ### updated

        load = torch.zeros(self.num_experts, device=logits.device, dtype=torch.float32) ### updated
        # Iterate through the k choices and count occurrences of each expert index
        for k_slot in range(top_k_indices_for_load.shape[1]): ### updated
            selected_expert_indices_in_this_slot = top_k_indices_for_load[:, k_slot] ### updated
            load.scatter_add_(0, selected_expert_indices_in_this_slot, 
                              torch.ones_like(selected_expert_indices_in_this_slot, dtype=torch.float)) ### updated
            
        eps = 1e-10
        load_balance_loss = torch.tensor(0.0, device=x_for_gating.device)
        if self.num_experts > 1:
            # Importance loss
            if importance.numel() > 1:
                mean_importance = importance.float().mean()
                var_importance = importance.float().var()
                if mean_importance.abs() > eps:
                    cv_importance = var_importance / (mean_importance**2 + eps)
                else:
                    cv_importance = torch.tensor(0.0, device=x_for_gating.device)
            else:
                cv_importance = torch.tensor(0.0, device=x_for_gating.device)

            # Load loss
            if load.numel() > 1:
                mean_load = load.float().mean()
                var_load = load.float().var()
                if mean_load.abs() > eps:
                    cv_load = var_load / (mean_load**2 + eps)
                else:
                    cv_load = torch.tensor(0.0, device=x_for_gating.device)
            else:
                cv_load = torch.tensor(0.0, device=x_for_gating.device)
            
            load_balance_loss = self.coef * (cv_importance + cv_load)
            
        return gates, load_balance_loss

    def forward(self, x_for_gating, expert_weights_list, is_training): 
        # x_for_gating: node features [N, F_in_gate]
        # expert_weights_list: list of K weight matrices [K, F_in, F_out]
        # Returns: combined_weight [F_in, F_out] and load_balance_loss

        gates, load_balance_loss = self.noisy_top_k(x_for_gating, is_training) # gates: [N, num_experts] 
        
        # Global combination: average gates over nodes, then combine expert weights
        # This results in one combined weight matrix for the whole batch
        avg_gates_over_nodes = gates.mean(dim=0) # [num_experts] 
        
        # Combine expert weights: Sum_j (avg_gate_j * ExpertWeight_j)
        # Expert weights are [F_in, F_out]. Need to stack them or iterate.
        # stacked_expert_weights = torch.stack(expert_weights_list, dim=0) # [num_experts, F_in, F_out]
        # combined_weight = torch.einsum('k,kio->io', avg_gates_over_nodes, stacked_expert_weights) # [F_in, F_out]
        
        # Simpler loop for combining based on avg_gates_over_nodes
        combined_weight = 0
        for i in range(self.num_experts):
            combined_weight += avg_gates_over_nodes[i] * expert_weights_list[i]
            
        return combined_weight, load_balance_loss 


class RTGCNSMoE(nn.Module):
    """
    act: activation function for Structural Role-based GRU
    n_node: number of nodes on the network
    output_dim: output embed size of node embedding
    seq_len: number of graphs
    attn_drop: attention/coefficient matrix dropout rate
    residual: if using short cut or not for GRU network
    neg_weight : the negative sampling ratio
    loss_weight: the hyper-parameter to balance the connective proximity and structural role proximity
    role_num : the number of role sets
    cross_role_num: the number of cross_role sets
    """

    def __init__(self,
                 act,
                 n_node,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 time_step,
                 neg_weight,
                 loss_weight,
                 attn_drop,
                 role_num,
                 cross_role_num,
                 residual=True,
                 dropout_rate=0.5, ### added
                 num_experts=2, ### added
                 moe_k=1,       ### added
                 moe_coef=1e-2,  ### added
                 ):
        super(RTGCNSMoE, self).__init__()

        self.act = act
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_time_steps = time_step
        self.neg_weight = neg_weight
        self.dropout_rate = dropout_rate ### updated
        self.residual = residual # For GCN skip connection logic
        self.bceloss = BCEWithLogitsLoss()
        
        self.num_experts = num_experts ### updated
        self.current_moe_load_balance_loss = 0.0 ### updated

        # Base parameters for expert weights (to be evolved)
        self.expert_weight1_bases = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, hidden_dim)) for _ in range(num_experts)]
        ) ### updated
        self.expert_weight2_bases = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden_dim, output_dim)) for _ in range(num_experts)]
        ) ### updated

        # GRU cells to evolve each expert weight
        self.evolve_expert_weights1 = nn.ModuleList(
            [MatGRUCell(n_node, input_dim, hidden_dim, cross_role_num) for _ in range(num_experts)]
        ) ### updated
        self.evolve_expert_weights2 = nn.ModuleList(
            [MatGRUCell(n_node, hidden_dim, output_dim, cross_role_num) for _ in range(num_experts)]
        ) ### updated
        
        # Gating Networks
        self.gate_network1 = GatingNetwork(input_dim, num_experts, top_k=moe_k, coef=moe_coef) ### updated
        self.gate_network2 = GatingNetwork(hidden_dim, num_experts, top_k=moe_k, coef=moe_coef) ### updated

        # GCN and HyperGNN layers (these will use the selected/combined weights)
        self.gcn_conv1 = GCNConv(input_dim, hidden_dim) ### updated
        self.gcn_conv2 = GCNConv(hidden_dim, output_dim) ### updated
        self.gcn_prej = nn.Linear(input_dim, output_dim, bias=False) # For GCN skip connection ### updated
        self.gcn_alpha_skip = nn.Parameter(torch.ones(1)) # For GCN skip connection ### updated
        
        self.hypergnn_layer1 = HyperGNN(input_dim, hidden_dim) ### updated
        self.hypergnn_layer2 = HyperGNN(hidden_dim, output_dim) ### updated

        self.loss_weight = loss_weight ### updated
        self.emb_weight_hyper = nn.Parameter(torch.ones(1)) # Alpha in Eq 7 (for current hypergraph) ### updated
        self.emb_weight_cross_role = nn.Parameter(torch.ones(1)) # Gamma in Eq 7 (for cross-role hypergraph) ### updated


    def forward(self, data, train_hypergraph,cross_role_hyper,cross_role_laplacian):
        """
        Forward pass of the RTGCN, processing input through both graph and hypergraph components, and combining outputs.
        """
        features_list = data[0] 
        edge_index_list = data[1] 
        adj_matrix_list = data[2] 

        current_hypergraph_laplacian_list = train_hypergraph 
        cross_role_hyper_list = cross_role_hyper # Combined H + H1 for GRU evolve input
        cross_role_laplacian_list = cross_role_laplacian # H1 Laplacian for cross-role HyperGNN layer

        embeds = []
        self.current_moe_load_balance_loss = torch.tensor(0.0, device=features_list[0].device) ### updated: ensure on correct device

        # Initialize lists of current (potentially evolved) expert weights
        current_expert_weights1 = list(self.expert_weight1_bases) ### updated
        current_expert_weights2 = list(self.expert_weight2_bases) ### updated

        # Time Step 0
        x_t0 = features_list[0]
        edge_index_t0 = edge_index_list[0]
        hypergraph_lap_t0 = current_hypergraph_laplacian_list[0]

        # Gating and weight selection for layer 1 at t=0
        selected_w1_t0, lb_loss1_t0 = self.gate_network1(x_t0, current_expert_weights1, self.training) ### updated
        self.current_moe_load_balance_loss += lb_loss1_t0 ### updated
        
        # GCN layer 1 at t=0
        h_gcn_l1_t0 = self.gcn_conv1(x_t0, edge_index_t0, selected_w1_t0) ### updated
        h_gcn_l1_t0 = F.relu(h_gcn_l1_t0) ### updated
        h_gcn_l1_t0_dropout = F.dropout(h_gcn_l1_t0, p=self.dropout_rate, training=self.training) ### updated

        # HyperGNN layer 1 at t=0
        h_hyper_l1_t0 = self.hypergnn_layer1(x_t0, hypergraph_lap_t0, selected_w1_t0) ### updated
        # Optional ReLU/dropout for hypergraph path
        # h_hyper_l1_t0 = F.relu(h_hyper_l1_t0)
        # h_hyper_l1_t0_dropout = F.dropout(h_hyper_l1_t0, p=self.dropout_rate, training=self.training)

        # Gating and weight selection for layer 2 at t=0
        # Input to gate_network2 is output of GCN path (h_gcn_l1_t0)
        # Or average of GCN and HyperGNN path? For now, GCN path output.
        selected_w2_t0, lb_loss2_t0 = self.gate_network2(h_gcn_l1_t0, current_expert_weights2, self.training) ### updated
        self.current_moe_load_balance_loss += lb_loss2_t0 ### updated
        
        # GCN layer 2 at t=0
        gnn_output_t0 = self.gcn_conv2(h_gcn_l1_t0_dropout, edge_index_t0, selected_w2_t0) ### updated
        if self.residual: ### GCN skip connection logic ### updated
            x0_gcn_t0 = self.gcn_prej(x_t0) ### updated
            gnn_output_t0 = gnn_output_t0 * self.gcn_alpha_skip + x0_gcn_t0 * (1 - self.gcn_alpha_skip) ### updated
        
        # HyperGNN layer 2 at t=0 (using h_hyper_l1_t0 as input)
        hyper_output_t0 = self.hypergnn_layer2(h_hyper_l1_t0, hypergraph_lap_t0, selected_w2_t0) ### updated
        hyper_output_t0 = F.log_softmax(hyper_output_t0, dim=1) ### updated

        output_t0 = gnn_output_t0 + self.emb_weight_hyper * hyper_output_t0 ### updated
        embeds.append(output_t0)

        # Subsequent Time Steps
        for i in range(1, self.num_time_steps):
            adj_matrix_ti = adj_matrix_list[i]
            cross_hyper_for_gru_ti = cross_role_hyper_list[i-1]
            
            # Evolve all expert weights
            evolved_expert_weights1_list_ti = [] ### updated
            for expert_idx in range(self.num_experts): ### updated
                evolved_w = self.evolve_expert_weights1[expert_idx](
                    adj_matrix_ti, current_expert_weights1[expert_idx], cross_hyper_for_gru_ti
                ) ### updated
                evolved_expert_weights1_list_ti.append(evolved_w) ### updated
            current_expert_weights1 = evolved_expert_weights1_list_ti # Update for next iteration or if used by L2 gate ### updated
            
            evolved_expert_weights2_list_ti = [] ### updated
            for expert_idx in range(self.num_experts): ### updated
                evolved_w = self.evolve_expert_weights2[expert_idx](
                    adj_matrix_ti, current_expert_weights2[expert_idx], cross_hyper_for_gru_ti
                ) ### updated
                evolved_expert_weights2_list_ti.append(evolved_w) ### updated
            current_expert_weights2 = evolved_expert_weights2_list_ti ### updated

            # Inputs for time i
            x_ti = features_list[i]
            edge_index_ti = edge_index_list[i]
            hypergraph_lap_ti = current_hypergraph_laplacian_list[i]
            cross_role_lap_ti = cross_role_laplacian_list[i-1] # H1 Laplacian from t-1->t

            # Gating and weight selection for layer 1 at time i
            selected_w1_ti, lb_loss1_ti = self.gate_network1(x_ti, current_expert_weights1, self.training) ### updated
            self.current_moe_load_balance_loss += lb_loss1_ti ### updated

            # GCN layer 1 at time i
            h_gcn_l1_ti = self.gcn_conv1(x_ti, edge_index_ti, selected_w1_ti) ### updated
            h_gcn_l1_ti = F.relu(h_gcn_l1_ti) ### updated
            h_gcn_l1_ti_dropout = F.dropout(h_gcn_l1_ti, p=self.dropout_rate, training=self.training) ### updated

            # HyperGNN (current roles) layer 1 at time i
            h_hyper_l1_ti = self.hypergnn_layer1(x_ti, hypergraph_lap_ti, selected_w1_ti) ### updated
            # h_hyper_l1_ti = F.relu(h_hyper_l1_ti) # Optional
            # h_hyper_l1_ti_dropout = F.dropout(h_hyper_l1_ti, p=self.dropout_rate, training=self.training)

            # Cross-role HyperGNN layer 1 at time i (uses features from t-1)
            h_cross_hyper_l1_ti = self.hypergnn_layer1(features_list[i-1], cross_role_lap_ti, selected_w1_ti) ### updated
            # h_cross_hyper_l1_ti = F.relu(h_cross_hyper_l1_ti) # Optional
            # h_cross_hyper_l1_ti_dropout = F.dropout(h_cross_hyper_l1_ti, p=self.dropout_rate, training=self.training)


            # Gating and weight selection for layer 2 at time i
            # Input to gate_network2 is from GCN path
            selected_w2_ti, lb_loss2_ti = self.gate_network2(h_gcn_l1_ti, current_expert_weights2, self.training) ### updated
            self.current_moe_load_balance_loss += lb_loss2_ti ### updated

            # GCN layer 2 at time i
            gnn_output_ti = self.gcn_conv2(h_gcn_l1_ti_dropout, edge_index_ti, selected_w2_ti) ### updated
            if self.residual: ### GCN skip connection logic ### updated
                x0_gcn_ti = self.gcn_prej(x_ti) ### updated
                gnn_output_ti = gnn_output_ti * self.gcn_alpha_skip + x0_gcn_ti * (1 - self.gcn_alpha_skip) ### updated
            
            # HyperGNN (current roles) layer 2 at time i
            hyper_output_ti = self.hypergnn_layer2(h_hyper_l1_ti, hypergraph_lap_ti, selected_w2_ti) ### updated
            hyper_output_ti = F.log_softmax(hyper_output_ti, dim=1) ### updated

            # Cross-role HyperGNN layer 2 at time i
            cross_hyper_output_ti = self.hypergnn_layer2(h_cross_hyper_l1_ti, cross_role_lap_ti, selected_w2_ti) ### updated
            cross_hyper_output_ti = F.log_softmax(cross_hyper_output_ti, dim=1) ### updated
            
            output_ti = gnn_output_ti + self.emb_weight_hyper * hyper_output_ti + self.emb_weight_cross_role * cross_hyper_output_ti ### updated
            embeds.append(output_ti)

        return embeds

    def get_loss(self, feed_dict ,data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian,list_loss_role):
        """
        Compute the loss for the model based on the predictions and true data.
        """
        node_1, node_2, node_2_negative = feed_dict.values()

        # Obtain a list of node embeddings through forward propagation.
        final_emb = self.forward(data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian) # [N, T, F]

        #Calculate cumulative loss across time steps [0, T-1]
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            emb_t = final_emb[t]  # [N, F]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]

            # Calculate scores for positive and negative node pairs
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten()

            # Binary cross-entropy loss for positive and negative pairs
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))

            #Calculate Connective Proximity loss
            graphloss = pos_loss + self.neg_weight*neg_loss
            self.graph_loss += graphloss

            #Calculate Structural Role Proximity
            role_loss=0
            calculate_loss=list_loss_role[t]
            for l in calculate_loss:
                node_role_emb=emb_t[l]
                a = node_role_emb/torch.norm(node_role_emb,dim=1,keepdim=True)
                similarity = torch.mm(a,a.T)
                I_mat=torch.ones_like(similarity)

                # Frobenius norm for Structural Role Proximity
                role_loss+=torch.norm(similarity-I_mat)**2/2
                del similarity,node_role_emb
            self.graph_loss+=self.loss_weight*role_loss
            
        return self.graph_loss


    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GCNConv(MessagePassing):
    """ Initialize the GCN convolution layer.
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        dropout (float): Dropout rate.
        concat (bool): Whether to concatenate results.
    """
    def __init__(self,in_channels,out_channels,dropout=0.0,concat=False):
        super(GCNConv,self).__init__(aggr='add')

    def forward(self,x,edge_index,W2):
        """ Forward pass of GCN convolution.
        Args:
            x (Tensor): Node feature matrix (N, in_channels).
            edge_index (Tensor): Edge indices.
            W2 (Tensor): Weight matrix for this layer.
        Returns:
            Tensor: Output after convolution.
        """
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(0))
        x=torch.matmul(x,W2)
        row,col=edge_index

        #Calculate the degree matrix
        deg=degree(col,x.size(0),dtype=x.dtype)

        #Calculate the negative one-half power of the degree matrix
        deg_inv_sqrt=deg.pow(-0.5)
        norm=deg_inv_sqrt[row]*deg_inv_sqrt[col]
        return self.propagate(edge_index,x=x,norm=norm)
    def message(self,x_j,norm):
        """ Messages passed between nodes.
        Args:
            x_j (Tensor): Feature matrix of neighboring nodes.
            norm (Tensor): Normalized degree matrix.
        """
        return norm.view(-1,1)*x_j


        



class GCN(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout,concat):
        """ Initialize the GCN model.
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden layer.
            output_dim (int): Dimension of output features.
            dropout (float): Dropout rate.
            concat (bool): Whether to concatenate results.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim,dropout,concat)
        self.conv2 = GCNConv(hidden_dim, output_dim,dropout,concat)
        self.prej =nn.Linear(input_dim,output_dim,bias=False)
        self.alpha1 = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index,gnn_weight1,gnn_weight2):
        """ Forward pass of the GCN model.
        Args:
            x (Tensor): Input feature matrix.
            edge_index (Tensor): Edge indices.
            gnn_weight1 (Tensor): Weight matrix for first GCN layer.
            gnn_weight2 (Tensor): Weight matrix for second GCN layer.
        Returns:
            Tensor: Output of the model.
        """
        x0 = self.prej(x)
        x = self.conv1(x, edge_index,gnn_weight1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index,gnn_weight2)
        X = x*self.alpha1+x0*(1-self.alpha1)
        return F.log_softmax(X, dim=1)



#HyperGNN
class HyperGNN(nn.Module):
    def __init__(self, input_dim, output_dim, negative_slope=0.2):
        """ Initialize the HyperGNN model.
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            negative_slope (float): Negative slope for leaky ReLU.
        """
        super(HyperGNN, self).__init__()
        self.negative_slope = negative_slope
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, node_initial_emb, hyp_graph,W):
        """ Forward pass of the HyperGNN model.
        Args:
            node_initial_emb (Tensor): Initial node embeddings.
            hyp_graph (Tensor): Hypergraph incidence matrix.
            W (Tensor): Weight matrix for this layer.
        Returns:
            Tensor: Updated node embeddings.
        """
        # rs = hyp_graph @ torch.matmul( node_initial_emb,W)
        hyp_graph_dense = hyp_graph.to_dense()
        node_emb_transformed = torch.matmul(node_initial_emb, W)
        rs = torch.matmul(hyp_graph_dense, node_emb_transformed)
        
        rs = (1-self.alpha)*self.proj(node_initial_emb)+rs*self.alpha
        return rs





class MatGRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.2 of the paper for the formula.
    """

    def __init__(self, n_node,input_dim, output_dim,cross_role_num):
        """Initialize the GRU cell with matrix-specific gates.
        Args:
            n_node (int): Number of nodes in the graph.
            input_dim (int): Dimensionality of input features per node.
            output_dim (int): Dimensionality of output features per node.
            cross_role_num (int): Number of cross-role interactions to consider.
        """
        super().__init__()

        #Update gate
        self.update = MatGRUGate(n_node,input_dim,
                                 output_dim,
                                 torch.nn.Sigmoid(),cross_role_num=cross_role_num)
        #Reset gate.
        self.reset = MatGRUGate(n_node,input_dim,
                                output_dim,
                                torch.nn.Sigmoid(),cross_role_num=cross_role_num)

        # Candidate memory content uses tanh to ensure the state values remain between -1 and 1.
        self.htilda = MatGRUGate(n_node,input_dim,
                                 output_dim,
                                 torch.nn.Tanh(),cross_role_num=cross_role_num)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the GRU parameters
        """
        reset_parameters(self.named_parameters)

    def forward(self, adj,weight_vars,H):
        """Perform a forward pass of the GRU cell.
        Args:
            adj (Tensor): The adjacency matrix.
            weight_vars (Tensor): The weight variables (hidden states).
            H (Tensor): Incident matrix or matrix representing cross-role interactions.
        Returns:
            Tensor: Updated node embeddings.
        """
        update = self.update(adj, weight_vars,H)
        reset = self.reset(adj, weight_vars,H)
        h_cap = reset * weight_vars
        h_cap = self.htilda(adj, h_cap,H)
        new_Q = (1 - update) * weight_vars + update * h_cap

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    For datasets with initial node features, if the dimension of the initial
    features does not equal the number of nodes, dimension matching is required.
    """
    def __init__(self, n_node,rows, cols, activation,cross_role_num):
        """Initialize the matrix GRU gate.
        Args:
            n_node (int): Number of nodes.
            rows (int): Number of rows in the matrix (usually corresponds to the number of nodes).
            cols (int): Number of columns in the matrix (dimensionality of the embeddings).
            activation (Activation): Activation function to apply at the gate.
            cross_role_num (int): Number of cross-role types considered in the gate.
        """
        super().__init__()
        self.activation = activation
        self.W = nn.Parameter(torch.Tensor(n_node, cols))
        self.W1=nn.Parameter(torch.Tensor(cross_role_num,cols))
        self.U = nn.Parameter(torch.Tensor(cols, cols))
        self.bias =nn.Parameter(torch.Tensor(rows, cols))

        #Dimensional transformation.
        self.transform = False
        if n_node != rows:
            self.P = nn.Parameter(torch.Tensor(rows, n_node))
            self.transform = True
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, adj, hidden,incident_matrix):
        """Forward pass through the gate.
        Args:
            adj (Tensor): Adjacency matrix of the graph.
            hidden (Tensor): Current hidden state.
            incident_matrix (Tensor): Incident matrix representing cross-role interactions.
        Returns:
            Tensor: Output of the gate after applying the activation function.
        """
        # temp = adj.matmul(self.W)
        # temp1=incident_matrix.matmul(self.W1)
        adj_dense = adj.to_dense()
        temp = torch.matmul(adj_dense, self.W)

        incident_matrix_dense = incident_matrix.to_dense()
        temp1 = torch.matmul(incident_matrix_dense, self.W1)
        
        if self.transform == True:
            out = self.activation(self.P.matmul(temp) +self.P.matmul(temp1)+ hidden.matmul(self.U) + self.bias)
        else:
            out = self.activation(temp + temp1 + hidden.matmul(self.U) + self.bias)

        return out



