import gradio as gr
from sklearn.feature_extraction.text import CountVectorizer
from rgl.utils import llm_utils
from rgl.node_retrieval.vector_search import VectorSearchEngine
from rgl.graph_retrieval.retrieve import retrieve
from openai import OpenAI

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

def run_classification(api_key, api_provider, model_name, query_paper_title, num_anchors):
    try:
        # Setup API
        base_url = "https://api.openai.com/v1"
        if api_provider == "DeepSeek":
            base_url = "https://api.deepseek.com"
            
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Input query paper title and retrieve similar papers (anchors).
        query_vector = vectorizer.transform([query_paper_title]).toarray()
        k = min(int(num_anchors), len(node_data))
        retrieved_indices = vector_search_engine.search(query_vector, k=k)[0][0]
        anchors = [node_data[idx] for idx in retrieved_indices]
        
        # Show retrieved anchor nodes
        anchors_info = "\n".join([f"- {paper['title']} (Classes: {paper['classes']})" for paper in anchors])
        
        # Retrieve subgraph from anchors.
        anchor_indices = [node_data.index(paper) for paper in anchors]
        subgraph_nodes = retrieve(src, dst, anchor_indices)
        
        # Show subgraph nodes
        subgraph_info = "\n".join([f"- {node_data[node]['title']} (Classes: {node_data[node]['classes']})" for node in subgraph_nodes])
        
        # Construct prompt with the query title and relevant paper information.
        relevant_paper_str = "\n".join(
            [f"Title: {node_data[node]['title']}, Classes: {node_data[node]['classes']}" for node in subgraph_nodes]
        )
        prompt = (
            "Given the paper title: '{}'\n\n"
            "And relevant paper information:\n{}\n\n"
            "List the classes that are most relevant to the query paper."
        ).format(query_paper_title, relevant_paper_str)
        
        # Query the LLM to obtain paper classification output.
        output = llm_utils.chat_openai(prompt, model=model_name, client=client)
        
        return prompt, anchors_info, subgraph_info, output
    
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

# Create Gradio interface
with gr.Blocks(title="Demo of RGL - Paper Node Classification") as demo:
    gr.Markdown("# Demo of RGL - Paper Node Classification")
    gr.Markdown("""
    This demo showcases **RoG (Retrieval-Augmented Generation on Graphs)** to classify academic paper nodes by leveraging graph-structured knowledge.
    
    Based on the paper: **"RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs"**
    """)
    
    with gr.Row():
        with gr.Column():
            # Input components
            api_key = gr.Textbox(
                label="API Key", 
                placeholder="Enter your API key here",
                type="password"
            )
            
            api_provider = gr.Radio(
                ["OpenAI", "DeepSeek"], 
                label="API Provider", 
                value="OpenAI"
            )
            
            model_name = gr.Dropdown(
                label="Model Name",
                choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "deepseek-chat"],
                value="gpt-4o-mini"
            )
            
            query_title = gr.Textbox(
                label="Query Paper Title",
                value="Vision Transformers for Traffic Sign Recognition",
                placeholder="Enter paper title to classify"
            )
            
            num_anchors = gr.Slider(
                label="Number of Anchor Nodes",
                minimum=1,
                maximum=5,
                value=3,
                step=1
            )
            
            classify_btn = gr.Button("Classify Paper", variant="primary")
        
        with gr.Column():
            # Output components
            prompt_output = gr.Textbox(label="Generated Prompt", lines=10)
            anchor_nodes = gr.Textbox(label="Retrieved Anchor Nodes", lines=5)
            subgraph_nodes = gr.Textbox(label="Subgraph Nodes", lines=8)
            classification_output = gr.Textbox(label="Classification Result", lines=5)
    
    # Function to update model options based on selected provider
    def update_model_options(provider):
        if provider == "OpenAI":
            return gr.Dropdown(choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], value="gpt-4o-mini")
        else:  # DeepSeek
            return gr.Dropdown(choices=["deepseek-chat"], value="deepseek-chat")
    
    # Connect components
    api_provider.change(
        update_model_options,
        inputs=[api_provider],
        outputs=[model_name]
    )
    
    classify_btn.click(
        run_classification,
        inputs=[api_key, api_provider, model_name, query_title, num_anchors],
        outputs=[prompt_output, anchor_nodes, subgraph_nodes, classification_output]
    )
    
    gr.Markdown("## Instructions")
    gr.Markdown("""
    1. Enter your API key for OpenAI or DeepSeek
    2. Input a paper title you want to classify
    3. Adjust the number of anchor nodes to retrieve (more nodes = broader context)
    4. Click "Classify Paper" to see the results
    
    The **RoG (Retrieval-Augmented Generation on Graphs)** process will:
    - Find similar papers based on title similarity
    - Extract a relevant subgraph from the knowledge graph
    - Use graph-structured context to generate accurate paper node classifications
    """)

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()