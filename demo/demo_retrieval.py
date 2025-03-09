from rgl.datasets.ogb import OGBRGLDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve, batch_retrieve, steiner_batch_retrieve, dense_batch_retrieve

dataset = OGBRGLDataset("ogbn-arxiv", "../dataset/ogbn-arxiv")
query_node_indices = [0, 1]
query_vectors = dataset.feat[query_node_indices]  # TODO multi query node; query text
engine = VectorSearchEngine(dataset.feat)
batch_seeds, _ = engine.search(query_vectors, k=3)
batch_seeds = [batch_seeds[qid].tolist() for qid in query_node_indices]

src = dataset.graph.edges()[0].numpy().tolist()
dst = dataset.graph.edges()[1].numpy().tolist()

print("retrieve")
for qid in query_node_indices:
    seeds = batch_seeds[qid]
    subg_nodes = retrieve(src, dst, seeds)
    print("query node:", qid)
    print("retrieved seed nodes", seeds)
    print("retrieved subgraph", subg_nodes)
    print()

print("-" * 80)

print("batch_retrieve")
subg_nodes = batch_retrieve(src, dst, batch_seeds)
print("batched query nodes:", query_node_indices)
print("batched retrieved nodes", batch_seeds)
print("batched retrieved subgraphs", subg_nodes)

print("-" * 80)

print("steiner_batch_retrieve")
subg_nodes = steiner_batch_retrieve(src, dst, batch_seeds)
print("batched query nodes:", query_node_indices)
print("batched retrieved nodes", batch_seeds)
print("batched retrieved subgraphs", subg_nodes)

print("-" * 80)

print("dense_batch_retrieve")
subg_nodes = dense_batch_retrieve(src, dst, batch_seeds)
print("batched query nodes:", query_node_indices)
print("batched retrieved nodes", batch_seeds)
print("batched retrieved subgraphs", subg_nodes)
