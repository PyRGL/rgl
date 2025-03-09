import torch
import numpy as np
import dgl
from pcst_fast import pcst_fast
from rgl.graph_retrieval import libretrieval

# C++ pybind
retrieve = libretrieval.retrieve
batch_retrieve = libretrieval.batch_retrieve
steiner_batch_retrieve = libretrieval.steiner_batch_retrieve
dense_batch_retrieve = libretrieval.dense_batch_retrieve

# # C
# import numpy as np
# import ctypes
# import os

# def load_c_retrieve():
#     lib_name = "libretrieval.dll" if os.name == "nt" else "libretrieval.so"
#     lib_path = os.path.join(os.path.dirname(__file__), lib_name)
#     lib = ctypes.CDLL(lib_path)
#     c_retrieve = lib.retrieve

#     lib.retrieve.argtypes = [
#         ctypes.POINTER(ctypes.c_int),  # src
#         ctypes.POINTER(ctypes.c_int),  # dst
#         ctypes.POINTER(ctypes.c_int),  # seed
#         ctypes.c_int,  # num_edge
#         ctypes.c_int,  # num_seed
#         ctypes.POINTER(ctypes.c_int),  # num_retrieved
#     ]
#     lib.retrieve.restype = ctypes.POINTER(ctypes.c_int)

#     def ndcg_score(src, dst, seeds):

#         src = list(src)
#         dst = list(dst)
#         num_edges = len(src)
#         seeds = list(seeds)
#         num_seeds = len(seeds)
#         num_retrieved = ctypes.c_int(0)
#         rel_type = ctypes.c_int * num_edges

#         result_ptr = c_retrieve(
#             rel_type(*src),
#             rel_type(*dst),
#             rel_type(*seeds),
#             ctypes.c_int(num_edges),
#             ctypes.c_int(num_seeds),
#             ctypes.byref(num_retrieved),
#         )
#         result = [result_ptr[i] for i in range(num_retrieved.value)]
#         return result

#     return ndcg_score
# retrieve = load_c_retrieve()


def retrieval_pcst(graph, q_emb, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    has_edge_attr = "edge_attr" in graph.edata
    if not has_edge_attr:
        topk_e = 0

    # Compute node prizes using cosine similarity.
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.ndata["feat"])
        topk = min(topk, graph.num_nodes())
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)
        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes())

    # Compute edge prizes only if edge attributes exist.
    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edata["edge_attr"])
        topk_e = min(topk_e, e_prizes.unique().size(0))
        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            # Avoid division by zero.
            value = min((topk_e - k) / indices.sum().item(), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - c)
        # Adjust cost_e so that at least one edge is selected.
        cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
    else:
        e_prizes = torch.zeros(graph.num_edges())

    # Prepare to build the list of candidate edges and virtual nodes.
    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    mapping_n = {}
    mapping_e = {}

    # Get all edges in the DGL graph in order.
    src_all, dst_all = graph.edges(order="eid")
    src_all_np = src_all.cpu().numpy() if src_all.is_cuda else src_all.numpy()
    dst_all_np = dst_all.cpu().numpy() if dst_all.is_cuda else dst_all.numpy()
    edge_index_np = np.stack((src_all_np, dst_all_np), axis=1)

    # Process each edge.
    if topk_e == 0:
        mapping_e = {i: i for i in range(graph.num_edges())}
        edges = edge_index_np.copy()
        costs = np.zeros(len(edges)) + cost_e
    else:
        for i, (src, dst) in enumerate(edge_index_np):
            prize_e = e_prizes[i]
            if prize_e <= cost_e:
                mapping_e[len(edges)] = i
                edges.append((src, dst))
                costs.append(cost_e - prize_e)
            else:
                virtual_node_id = graph.num_nodes() + len(virtual_n_prizes)
                mapping_n[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                costs.extend([0, 0])
                virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes.cpu().numpy(), np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_edges) > 0:
        costs = np.array(costs)
        edges = np.array(list(edges) + list(virtual_edges))
    else:
        costs = np.array(costs)
        edges = np.array(edges)

    # Run the prize-collecting Steiner tree algorithm.
    root = -1  # unrooted
    num_clusters = 1
    pruning = "gw"
    verbosity_level = 0
    vertices, selected_edge_indices = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    # Separate selected nodes and edges.
    selected_nodes = vertices[vertices < graph.num_nodes()]
    selected_edges = [mapping_e[e] for e in selected_edge_indices if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes()]
    if len(virtual_vertices) > 0:
        virtual_edges_selected = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges_selected)

    # Make sure all nodes in the selected edges are included.
    selected_src = src_all_np[selected_edges]
    selected_dst = dst_all_np[selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, selected_src, selected_dst]))
    # selected_edges can be fewer than the induced edges.
    return selected_nodes, selected_edges


def batch_retrieval_pcst(graph, q_emb_batch, topk=3, topk_e=3, cost_e=0.5):
    """
    Batched version of retrieval_pcst where q_emb_batch is a tensor of shape (B, d).
    Returns a tuple of two lists:
      - batch_selected_nodes: a list of numpy arrays of selected node IDs per query.
      - batch_selected_edges: a list of numpy arrays of selected original edge indices per query.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    batch_size = q_emb_batch.shape[0]
    batch_selected_nodes = []
    batch_selected_edges = []

    c = 0.01
    has_edge_attr = "edge_attr" in graph.edata
    # If no edge attributes, force topk_e to 0.
    topk_e_effective = topk_e if has_edge_attr else 0

    # Precompute the graph edge index (order="eid") only once.
    src_all, dst_all = graph.edges(order="eid")
    src_all_np = src_all.cpu().numpy() if src_all.is_cuda else src_all.numpy()
    dst_all_np = dst_all.cpu().numpy() if dst_all.is_cuda else dst_all.numpy()
    edge_index_np = np.stack((src_all_np, dst_all_np), axis=1)

    # Precompute cosine similarities for nodes (shape: [B, num_nodes])
    node_cos_sim = F.cosine_similarity(q_emb_batch.unsqueeze(1), graph.ndata["feat"].unsqueeze(0), dim=-1)
    # Precompute cosine similarities for edges if needed (shape: [B, num_edges])
    if has_edge_attr and topk_e_effective > 0:
        edge_cos_sim = F.cosine_similarity(q_emb_batch.unsqueeze(1), graph.edata["edge_attr"].unsqueeze(0), dim=-1)

    # Process each query in the batch.
    for i in range(batch_size):
        # ----- Node prizes -----
        if topk > 0:
            n_sim = node_cos_sim[i]  # (num_nodes,)
            effective_topk = min(topk, graph.num_nodes())
            _, topk_n_indices = torch.topk(n_sim, effective_topk, largest=True)
            # Initialize node prizes as zeros and assign decreasing prize values.
            node_prizes = torch.zeros_like(n_sim)
            node_prizes[topk_n_indices] = torch.arange(effective_topk, 0, -1).float()
        else:
            node_prizes = torch.zeros(graph.num_nodes())

        # ----- Edge prizes -----
        if has_edge_attr and topk_e_effective > 0:
            # Clone so we can modify without affecting the precomputed tensor.
            e_prizes = edge_cos_sim[i].clone()
            unique_vals = e_prizes.unique()
            effective_topk_e = min(topk_e_effective, unique_vals.size(0))
            # Get the top unique cosine values.
            topk_e_values, _ = torch.topk(unique_vals, effective_topk_e, largest=True)
            # Zero-out entries below the smallest top-k value.
            e_prizes[e_prizes < topk_e_values[-1]] = 0.0
            last_topk_e_value = float(effective_topk_e)
            for k in range(effective_topk_e):
                indices = e_prizes == topk_e_values[k]
                count = indices.sum().item()
                if count > 0:
                    value = min((effective_topk_e - k) / count, last_topk_e_value)
                    e_prizes[indices] = value
                    last_topk_e_value = value * (1 - c)
            # Adjust cost so that at least one edge is selected.
            cost_e_adj = min(cost_e, e_prizes.max().item() * (1 - c / 2))
        else:
            e_prizes = torch.zeros(graph.num_edges())
            cost_e_adj = cost_e

        # ----- Build candidate edges and virtual nodes -----
        costs = []
        edges = []
        virtual_edges = []
        virtual_n_prizes = []
        mapping_n = {}  # For virtual nodes: virtual node id -> original edge index.
        mapping_e = {}  # For real candidate edges: candidate index -> original edge index.

        if topk_e_effective == 0:
            # If no edge prizes to consider, simply use all edges.
            mapping_e = {j: j for j in range(graph.num_edges())}
            edges = edge_index_np.copy().tolist()
            costs = [cost_e_adj] * len(edges)
        else:
            for idx, (src, dst) in enumerate(edge_index_np):
                prize_e = e_prizes[idx].item()
                if prize_e <= cost_e_adj:
                    mapping_e[len(edges)] = idx
                    edges.append((src, dst))
                    costs.append(cost_e_adj - prize_e)
                else:
                    # Insert a virtual node for this edge.
                    virtual_node_id = graph.num_nodes() + len(virtual_n_prizes)
                    mapping_n[virtual_node_id] = idx
                    # Create two virtual edges.
                    virtual_edges.append((src, virtual_node_id))
                    virtual_edges.append((virtual_node_id, dst))
                    costs.extend([0, 0])
                    virtual_n_prizes.append(prize_e - cost_e_adj)

        # Concatenate node prizes with prizes for virtual nodes.
        prizes = np.concatenate([node_prizes.cpu().numpy(), np.array(virtual_n_prizes)])
        num_candidate_edges = len(edges)
        if len(virtual_edges) > 0:
            # Combine candidate edges with virtual edges.
            edges_combined = np.array(edges + virtual_edges)
            costs = np.array(costs)
        else:
            edges_combined = np.array(edges)
            costs = np.array(costs)

        # ----- Run the Prize-Collecting Steiner Tree algorithm -----
        # (Assumes pcst_fast is available; adjust parameters as needed.)
        root = -1  # unrooted
        num_clusters = 1
        pruning = "gw"
        verbosity_level = 0
        vertices, selected_edge_indices = pcst_fast(
            edges_combined, prizes, costs, root, num_clusters, pruning, verbosity_level
        )

        # Separate selected nodes (only those originally in the graph).
        selected_nodes = vertices[vertices < graph.num_nodes()]
        # Map candidate edge indices back to original edge indices.
        selected_edges = [mapping_e[e] for e in selected_edge_indices if e < num_candidate_edges]
        # If any virtual nodes were selected, map them back.
        virtual_vertices = vertices[vertices >= graph.num_nodes()]
        if len(virtual_vertices) > 0:
            virtual_edges_selected = [mapping_n[i] for i in virtual_vertices]
            selected_edges = list(selected_edges) + virtual_edges_selected
            selected_edges = np.array(selected_edges)
        else:
            selected_edges = np.array(selected_edges)

        # Ensure that all nodes from the selected edges are included.
        selected_src = src_all_np[selected_edges]
        selected_dst = dst_all_np[selected_edges]
        all_selected_nodes = np.unique(np.concatenate([selected_nodes, selected_src, selected_dst]))

        batch_selected_nodes.append(all_selected_nodes)
        batch_selected_edges.append(selected_edges)

    return batch_selected_nodes, batch_selected_edges
