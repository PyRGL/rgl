# coding = utf8

import numpy as np
import faiss


class VectorSearchEngine(object):
    def __init__(self, vectors, index=None):
        super().__init__()
        if not isinstance(vectors, np.ndarray):
            vectors = vectors.numpy()
        self.vectors = np.array(vectors)
        self.dim = self.vectors.shape[1]

        if index is not None:
            self.index = index
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.vectors)

    def search(self, query_vectors, k=10):
        query_vectors = np.asarray(query_vectors)
        topK_distances, topK_indices = self.index.search(query_vectors, k)

        return topK_indices, topK_distances
