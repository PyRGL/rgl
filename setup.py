import setuptools
import io
from setuptools import setup, find_packages

setup(
    name="rgl",
    python_requires=">3.5.0",
    version="0.0.1",
    author="anthonynus",
    author_email="e0403849@u.nus.edu",
    packages=find_packages(exclude=["benchmarks", "data", "demo", "dist", "doc", "docs", "logs", "models", "test"]),
    install_requires=[],
    extras_require={},
    package_data={},
    description="RGL - RAG-on-Graphs Library",
    license="GNU General Public License v3.0 (See LICENSE)",
    long_description="RGL - RAG-on-Graphs Library",
    url="https://github.com/PyRGL/rgl",
)
