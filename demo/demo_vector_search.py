from rgl.datasets.ogb import OGBRGLDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine

dataset = OGBRGLDataset("ogbn-arxiv")
query_node_indices = [0, 1, 2]
query_vectors = dataset.feat[query_node_indices]
engine = VectorSearchEngine(dataset.feat)
topK_indices, topK_distances = engine.search(query_vectors, k=5)
print(topK_indices)
print(topK_distances)
