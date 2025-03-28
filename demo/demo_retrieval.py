from rgl.datasets.ogb import OGBRGLDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve, batch_retrieve, steiner_batch_retrieve, dense_batch_retrieve
from rgl.utils.utils import get_logger

logger = get_logger()

dataset = OGBRGLDataset("ogbn-arxiv", "../dataset/ogbn-arxiv")
query_node_indices = [0, 1]
query_vectors = dataset.feat[query_node_indices]  # TODO multi query node; query text
engine = VectorSearchEngine(dataset.feat)
batch_seeds, _ = engine.search(query_vectors, k=3)
batch_seeds = [batch_seeds[qid].tolist() for qid in query_node_indices]

src = dataset.graph.edges()[0].numpy().tolist()
dst = dataset.graph.edges()[1].numpy().tolist()

logger.info("retrieve")
for qid in query_node_indices:
    seeds = batch_seeds[qid]
    subg_nodes = retrieve(src, dst, seeds)
    logger.info(f"query node: {qid}")
    logger.info(f"retrieved seed nodes: {seeds}")
    logger.info(f"retrieved subgraph: {subg_nodes}")
    logger.info("")

logger.info("-" * 80)

logger.info("batch_retrieve")
subg_nodes = batch_retrieve(src, dst, batch_seeds)
logger.info(f"batched query nodes: {query_node_indices}")
logger.info(f"batched retrieved nodes: {batch_seeds}")
logger.info(f"batched retrieved subgraphs: {subg_nodes}")
logger.info("-" * 80)

logger.info("steiner_batch_retrieve")
subg_nodes = steiner_batch_retrieve(src, dst, batch_seeds)
logger.info(f"batched query nodes: {query_node_indices}")
logger.info(f"batched retrieved nodes: {batch_seeds}")
logger.info(f"batched retrieved subgraphs: {subg_nodes}")
logger.info("-" * 80)

logger.info("dense_batch_retrieve")
subg_nodes = dense_batch_retrieve(src, dst, batch_seeds)
logger.info(f"batched query nodes: {query_node_indices}")
logger.info(f"batched retrieved nodes: {batch_seeds}")
logger.info(f"batched retrieved subgraphs: {subg_nodes}")
logger.info("-" * 80)

logger.info("baseline networkx steiner_tree")
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

G = nx.Graph()
G.add_edges_from(zip(src, dst))
subg_nodes = []
for seeds in batch_seeds:
    subg_nodes.append(steiner_tree(G, seeds).nodes())
logger.info(f"batched query nodes: {query_node_indices}")
logger.info(f"batched retrieved nodes: {batch_seeds}")
logger.info(f"batched retrieved subgraphs: {subg_nodes}")
logger.info("-" * 80)
