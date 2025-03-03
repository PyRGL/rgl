from rgl.datasets.ogb import OGBRGLDataset
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve
from rgl.utils import llm_utils
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"

dataset = OGBRGLDataset("ogbn-arxiv")
titles = dataset.raw_ndata["title"]
abstracts = dataset.raw_ndata["abstract"]
src = dataset.graph.edges()[0].numpy().tolist()
dst = dataset.graph.edges()[1].numpy().tolist()

query_node_indices = [0]
query_vectors = dataset.feat[query_node_indices]  # TODO multi query node; query text
engine = VectorSearchEngine(dataset.feat)
topK_indices, topK_distances = engine.search(query_vectors, k=3)

for qid, qnid in enumerate(query_node_indices):
    seeds = topK_indices[qid].tolist()
    subg_nodes = retrieve(src, dst, seeds)
    print("query node:", qnid)
    print("retrieved seed nodes", seeds)
    print("retrieved subgraph", subg_nodes)

    query_title = titles[qnid]
    query_abstract_gt = abstracts[qnid]

    prompt = f"""
You are an expert academic writer. Given the title of a query research paper and a set of related research papers (with their titles and abstracts), generate a concise, informative, and coherent abstract for the query paper. The generated abstract should reflect the ideas present in the related papers.
Query Paper Title: {query_title}

Related Papers:"""
    prompt_print = prompt

    for node in subg_nodes:
        related_title = titles[node]
        related_abstract = abstracts[node]
        prompt += f"\n\nTitle: {related_title}\nAbstract: {related_abstract}"
        prompt_print += f"\nTitle: {related_title}\nAbstract: {related_abstract[:40]}..."

    prompt += "\n\nGenerate the abstract for the query paper below:"
    prompt_print += "\n\nGenerate the abstract for the query paper below:"

    print(f"=== Prompt Sent to {model} ===")
    print(prompt_print)
    print("=============================\n")

    sys_msg = "You are an expert in academic writing."
    generated_abstract = llm_utils.chat_openai(prompt, model=model, sys_prompt=sys_msg)
    print("=== Generated Abstract ===")
    print(generated_abstract)
    print("\n=== Ground Truth Abstract ===")
    print(query_abstract_gt)
    print("\n" + "=" * 80 + "\n")

    # TODO compare with no RA; compare with other retrieval methods
