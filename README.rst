.. role:: raw-html-m2r(raw)
   :format: html



.. raw:: html

   <!-- include logo svg in this markdown -->
   <!-- <p align="center">
       <img src="rgl-logo.png" width="400"/>
   </p> -->



RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs
==============================================================================================

RGL is a **friendly and efficient Retrieval-Augmented Generation on Graphs (RAG on Graph) library** for AI researchers, providing seamless integration with **DGL** and **PyG** while offering high-performance graph retrieval algorithms, many of which are optimized in **C++** for efficiency. 

Features
--------

✅ **Seamless Integration** – Works smoothly with **DGL** and **PyG**\ :raw-html-m2r:`<br>`
⚡ **Optimized Performance** – C++-backed retrieval algorithms for speed\ :raw-html-m2r:`<br>`
🧠 **AI-Focused** – Tailored for **GraphRAG** research and applications\ :raw-html-m2r:`<br>`
🔗 **Scalability** – Handles large-scale graphs with ease  

Homepage, Documentation and Paper
---------------------------------


* Homepage: https://github.com/PyRGL/rgl
* Documentation: https://rgl.readthedocs.io
* Paper Access: `\ *RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs* <https://arxiv.org/abs/2503.19314>`_

Requirements
------------


* DGL: https://www.dgl.ai/pages/start.html
* PyG (Optional): https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Installation
------------

.. code-block:: bash

   pip install rgl

Compile C++ libraries (Optional)
--------------------------------

.. code-block:: bash

   cd clibs
   ./build_linux.sh

Toy Example
-----------

.. code-block:: python

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

Output:

.. code-block::

   === Prompt Sent to Model ===
   Given the paper title: 'Vision Transformers for Traffic Sign Recognition'

   And relevant paper information:
   Title: Graph Neural Networks in Practice, Classes: ML, Graph
   Title: Deep Learning for Graph Data, Classes: AI, ML
   Title: Advanced Topics in Machine Learning, Classes: ML
   Title: Computer Vision for Autonomous Vehicles, Classes: CV, AI, Robotics

   List the classes that are most relevant to the query paper.

   === RAG Paper Classification Output ===
    The classes that are most relevant to the query paper 'Vision Transformers for Traffic Sign Recognition' are:

   1. Computer Vision (CV)
   2. Artificial Intelligence (AI)
   3. Machine Learning (ML)

   These classes are particularly relevant because traffic sign recognition is a task within the domain of computer vision and typically involves machine learning techniques, including advanced models like Vision Transformers.

Demo
----

We recommend you to get started with some demo.

Basic Operations
^^^^^^^^^^^^^^^^


* `RGL Dataset Loading <demo/demo_load_rgl_dataset.py>`_
* `RGL Graph Retrieval <demo/demo_retrieval.py>`_

Applications
^^^^^^^^^^^^


* `Abstract generation via RGL <demo/demo_rag_on_graph_abstract_generation.py>`_
* `Paper classification via RGL <demo/demo_rag_on_graph_paper_classification.py>`_

Cite
----

.. code-block::

   @misc{li2025rglgraphcentricmodularframework,
         title={RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs}, 
         author={Yuan Li and Jun Hu and Jiaxin Jiang and Zemin Liu and Bryan Hooi and Bingsheng He},
         year={2025},
         eprint={2503.19314},
         archivePrefix={arXiv},
         primaryClass={cs.IR},
         url={https://arxiv.org/abs/2503.19314}, 
   }
