import os
import openai
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from rgl.utils import llm_utils
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve


# Task: Given a paper title, determine its most relevant classes based on similar papers.

# below are example paper data with titles and classes
node_data = [
    {"title": "Deep Learning for Graph Data", "classes": "AI, ML"},
    {"title": "Introduction to Neural Networks", "classes": "ML"},
    {"title": "Graph Neural Networks in Practice", "classes": "ML, Graph"},
    {"title": "Advanced Topics in Machine Learning", "classes": "ML"},
    {"title": "Reinforcement Learning in Robotics", "classes": "Robotics, ML"},
    {"title": "Large Language Models and Reasoning", "classes": "AI, NLP"},
    {"title": "Bayesian Methods in Deep Learning", "classes": "ML, Bayesian"},
    {"title": "Computer Vision for Autonomous Vehicles", "classes": "CV, AI, Robotics"},
]
edges = [
    [1, 1, 1, 1, 0, 0, 2, 2, 3, 3, 4, 5, 6],
    [0, 2, 3, 6, 2, 3, 3, 4, 6, 7, 7, 7, 0],
]
src, dst = edges
src, dst = src + dst, dst + src  # to undirected graph

# bag-of-words representation of paper titles
titles = [paper["title"] for paper in node_data]

# bag-of-words representation of paper titles
vectorizer = CountVectorizer()
paper_feats = vectorizer.fit_transform(titles).toarray()

# create a vector search engine
vector_search_engine = VectorSearchEngine(paper_feats)


def retrieve_seed_nodes_by_query(query_paper_title, k=3):
    # vectorize the query paper title
    query_vector = vectorizer.transform([query_paper_title]).toarray()

    # retrieve the most similar papers
    retrived_indices = vector_search_engine.search(query_vector, k=k)[0][0]
    relevant_papers = [node_data[idx] for idx in retrived_indices]

    return relevant_papers


def retrieve_subgraph_by_seed_nodes(seed_nodes):
    seeds = [node_data.index(paper) for paper in seed_nodes]
    subgraph_nodes = retrieve(src, dst, seeds)

    return subgraph_nodes


def generate_paper_classes(query_paper_title, subgraph_nodes):
    # concatenate the relevant paper information into a string
    relevant_paper_str = "\n".join(
        [f"Title: {node_data[node]['title']}, Classes: {node_data[node]['classes']}" for node in subgraph_nodes]
    )

    # prmopt construction
    prompt = (
        "Given the paper title: '{}'\n\n"
        "And relevant paper information:\n{}\n\n"
        "List the classes that are most relevant to the query paper."
    ).format(query_paper_title, relevant_paper_str)

    print("\n=== Prompt Sent to Model ===\n{}".format(prompt))
    # feeding the prompt to LLM for paper classification
    output = llm_utils.chat_openai(prompt, model="gpt-4o-mini")

    return output


query_paper_title = "Vision Transformers for Traffic Sign Recognition"
seeds = retrieve_seed_nodes_by_query(query_paper_title)
subgraph = retrieve_subgraph_by_seed_nodes(seeds)
output = generate_paper_classes(query_paper_title, subgraph)
print("\n=== RAG Paper Classification Output ===\n {}".format(output))
