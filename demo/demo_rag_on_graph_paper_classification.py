from sklearn.feature_extraction.text import CountVectorizer
from rgl.utils import llm_utils
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve

# Define nodes.
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

# Define edges. The two lists represent connections: the first list is the source nodes and the second list is the destination nodes.
edges = [
    [1, 1, 1, 1, 0, 0, 2, 2, 3, 3, 4, 5, 6],
    [0, 2, 3, 6, 2, 3, 3, 4, 6, 7, 7, 7, 0],
]

# Convert to an undirected graph by adding reversed edges.
src, dst = edges
src, dst = src + dst, dst + src

# Prepare title features and initialize vector search engine.
titles = [paper["title"] for paper in node_data]
vectorizer = CountVectorizer()
paper_feats = vectorizer.fit_transform(titles).toarray()
vector_search_engine = VectorSearchEngine(paper_feats)

# Input query paper title and retrieve similar papers (anchors).
query_paper_title = "Vision Transformers for Traffic Sign Recognition"
query_vector = vectorizer.transform([query_paper_title]).toarray()
retrieved_indices = vector_search_engine.search(query_vector, k=3)[0][0]
anchors = [node_data[idx] for idx in retrieved_indices]

# Retrieve subgraph from anchors.
anchor_indices = [node_data.index(paper) for paper in anchors]
subgraph_nodes = retrieve(src, dst, anchor_indices)

# Construct prompt with the query title and relevant paper information.
relevant_paper_str = "\n".join(
    [f"Title: {node_data[node]['title']}, Classes: {node_data[node]['classes']}" for node in subgraph_nodes]
)
prompt = (
    "Given the paper title: '{}'\n\n"
    "And relevant paper information:\n{}\n\n"
    "List the classes that are most relevant to the query paper."
).format(query_paper_title, relevant_paper_str)
print("\n=== Prompt Sent to Model ===\n{}".format(prompt))

# Query the LLM to obtain paper classification output.
output = llm_utils.chat_openai(prompt, model="gpt-4o-mini")
print("\n=== RAG Paper Classification Output ===\n {}".format(output))
