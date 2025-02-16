import os
import openai
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from rgl.utils import llm_utils

# Q&A on the toy dataset (find relevant authors)
titles = [
    "Deep Learning for Graph Data",
    "Introduction to Neural Networks",
    "Graph Neural Networks in Practice",
    "Advanced Topics in Machine Learning",
]
authors = ["Alice, Bob", "Charlie", "Diana, Eve", "Frank, Grace"]

# bag-of-words representation of paper titles
vectorizer = CountVectorizer()
paper_bow_vectors = vectorizer.fit_transform(titles)


def retrieve_relevant_papers(query, paper_bow_vectors, vectorizer, k=3):
    # Transform the query into the same bag-of-words space.
    query_vec = vectorizer.transform([query])

    # Compute dot product similarity between the query and each paper title.
    # The result is a (n_documents, 1) matrix.
    similarities = paper_bow_vectors.dot(query_vec.T).toarray().flatten()

    # (paper_index, similarity_score) tuples
    scores = list(enumerate(similarities))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    relevant_indices = [i for i, score in scores_sorted][:k]
    return relevant_indices


def main():
    query_topic = "Graph Neural Networks"
    relevant_indices = retrieve_relevant_papers(query_topic, paper_bow_vectors, vectorizer, k=3)
    print("Query Topic:", query_topic)
    print("Retrieved paper indices:", relevant_indices)

    prompt = f"""
You are an expert in academic research. Given a topic and a list of research papers (with their titles and authors),
please identify the authors whose work is most relevant to the topic.

Topic: {query_topic}

Research Papers:"""

    for idx in relevant_indices:
        prompt += f"\n\nTitle: {titles[idx]}\nAuthors: {authors[idx]}"

    prompt += "\n\nBased on the above papers, list the names of the authors whose work is most relevant to the topic."

    print("\n=== Prompt Sent to Model ===")
    print(prompt)
    print("=" * 30 + "\n")

    system_msg = "You are an expert in academic research."
    answer = llm_utils.chat_openai(prompt, model="gpt-4o-mini", system_message=system_msg)

    print("=== Generated Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
