import setuptools
import io
from setuptools import setup, find_packages

packages = find_packages(exclude=["benchmarks", "data", "demo", "dist", "doc", "docs", "logs", "models", "test"])
print("packages:", packages)

setup(
    name="rgl",
    python_requires=">3.5.0",
    version="0.0.1",
    author="anthonynus",
    author_email="e0403849@u.nus.edu",
    packages=packages,
    install_requires=["ogb", "patool", "faiss-cpu"],
    extras_require={"torch-2.4": ["dgl @ https://data.dgl.ai/wheels/torch-2.4/dgl-0.9.0-cp38-cp38-linux_x86_64.whl"]},
    package_data={"rgl": ["graph_retrieval/libretrieval.cpython-310-x86_64-linux-gnu.so"]},
    description="RGL - RAG-on-Graphs Library",
    license="GNU General Public License v3.0 (See LICENSE)",
    long_description="RGL - RAG-on-Graphs Library",
    url="https://github.com/PyRGL/rgl",
)
