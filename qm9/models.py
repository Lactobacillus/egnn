from models.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, 
                attention=False, update_coords=False, use_rinv=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)
        
        self.update_coords = update_coords
        self.use_rinv = use_rinv
        if not self.update_coords:
            del self.coord_mlp
        self.act_fn = act_fn
        self.coords_weight = coords_weight

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index     # row: source node, col: target node
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask # edge_feat: edge feature, coord_diff: relative distance between source and target node
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0)) # aggregate the edge feature to the source node
        coord = coord + agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None,
                radial=None, coord_diff=None):
        row, col = edge_index
        
        if (radial is None) or (coord_diff is None):
            if self.use_rinv:
                radial = 1.0 / (radial + 0.3)
            radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask
        coord_new = None
        if self.update_coords:
            coord_new = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord_new, edge_attr


                equation:bool = False, update_coords=False, use_rinv=False):
class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1,
                equation:bool = False, update_coords=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.use_rinv = use_rinv


        self.update_coords = update_coords
        
        self.norm_constant = 1e-6
        
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention, update_coords=update_coords, use_rinv=use_rinv))
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention, update_coords=update_coords))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

        if self.use_rinv:
            radial = 1.0 / (radial + 0.3)
    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        
        radial, coord_diff = self.coord2radial(edges, x) # [n_edges, 1], [n_edges, 3] call once and reuse
        
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, coord_new, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes,
                                                              radial=radial, coord_diff=coord_diff)
            else:
                h, coord_new, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes,
                                                      radial=radial, coord_diff=coord_diff)
            
            if self.update_coords:
                x = coord_new
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

                
        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        
        pred = self.graph_dec(h)
        return pred.squeeze(1), x
    
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff/(norm + self.norm_constant)

        return radial, coord_diff



