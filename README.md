<!-- include logo svg in this markdown -->
<!-- <p align="center">
    <img src="rgl-logo.png" width="400"/>
</p> -->

# RGL - RAG-on-Graphs Library  

RGL is a **friendly and efficient Graph Retrieval-Augmented Generation (GraphRAG) library** for AI researchers, providing seamless integration with **DGL** and **PyG** while offering high-performance graph retrieval algorithms, many of which are optimized in **C++** for efficiency. 

## Features  
✅ **Seamless Integration** – Works smoothly with **DGL** and **PyG**  
⚡ **Optimized Performance** – C++-backed retrieval algorithms for speed  
🧠 **AI-Focused** – Tailored for **GraphRAG** research and applications  
🔗 **Scalability** – Handles large-scale graphs with ease  

## Homepage and Documentation

- Homepage: https://github.com/PyRGL/rgl
- Documentation: https://rgl.readthedocs.io

## Requirements

- DGL: https://www.dgl.ai/pages/start.html
- PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Installation

```bash
pip install rgl
```

## Compile C++ libraries

```bash
cd clibs
./build_linux.sh
```

## Run demos

```bash
cd demo
python demo_x.py
```