from rgl.datasets.ogb import OGBRGLDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve

dataset = OGBRGLDataset("ogbn-arxiv")
query_node_indices = [0, 1, 2]
query_vectors = dataset.feat[query_node_indices]  # TODO multi query node; query text
engine = VectorSearchEngine(dataset.feat)
topK_indices, topK_distances = engine.search(query_vectors, k=3)

src = dataset.graph.edges()[0].numpy().tolist()
dst = dataset.graph.edges()[1].numpy().tolist()
for qid in query_node_indices:
    seeds = topK_indices[qid].tolist()
    res = retrieve(src, dst, seeds)
    print(qid)
    print(seeds)
    print(res)
    print()
