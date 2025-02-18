from rgl.datasets.ogb import OGBRGLDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve, batch_retrieve

dataset = OGBRGLDataset("ogbn-arxiv")
query_node_indices = [0, 1, 2]
query_vectors = dataset.feat[query_node_indices]  # TODO multi query node; query text
engine = VectorSearchEngine(dataset.feat)
topK_indices, topK_distances = engine.search(query_vectors, k=3)

src = dataset.graph.edges()[0].numpy().tolist()
dst = dataset.graph.edges()[1].numpy().tolist()

print("single retrieve:")
for qid in query_node_indices:
    seeds = topK_indices[qid].tolist()
    subg_nodes = retrieve(src, dst, seeds)
    print("query node:", qid)
    print("retrieved seed nodes", seeds)
    print("retrieved subgraph", subg_nodes)
    print()

print("-" * 80)
print("batch retrieve")
seeds = [topK_indices[qid].tolist() for qid in query_node_indices]
subg_nodes = batch_retrieve(src, dst, seeds)
print(query_node_indices)
print(seeds)
print(subg_nodes)
