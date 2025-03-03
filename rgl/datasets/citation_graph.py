from rgl.data.dataset import DownloadableRGLDataset
from ogb.nodeproppred import DglNodePropPredDataset
import torch
import os
import patoolib

import dgl
import dgl.data
import numpy as np
import networkx as nx
import scipy.sparse as sp
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools

from rgl.utils.data_utils import extract_archive
from rgl.utils.utils import get_logger

logger = get_logger()


class CoraRGLDataset(DownloadableRGLDataset):

    def __init__(self, dataset_root_path=None):
        # manually download dataset https://github.com/XiaoxinHe/TAPE to download/
        # manually unzip to raw/
        download_urls = []
        download_file_name = []

        super().__init__(
            dataset_name="cora",
            download_urls=download_urls,
            download_file_names=download_file_name,
            cache_name=None,
            dataset_root_path=dataset_root_path,
        )

    def download_graph(self, dataset_name, graph_root_path):
        dataset = dgl.data.CoraGraphDataset(raw_dir=graph_root_path)
        graph = dataset[0]
        self.backend_dataset = dataset
        self.graph = graph
        self.feat = graph.ndata["feat"]
        self.label = graph.ndata["label"].flatten()
        self.train_mask = graph.ndata["train_mask"]
        self.val_mask = graph.ndata["val_mask"]
        self.test_mask = graph.ndata["test_mask"]

    def process(self):
        dataset_lower = self.dataset_name.replace("-", "_")

        # inspired by https://github.com/XiaoxinHe/TAPE/blob/main/core/data_utils/load_cora.py

        edge_path = f"{self.raw_root_path}/cora_orig/cora.cites"
        text_path = f"{self.raw_root_path}/cora_orig/cora.content"
        papers_path = f"{self.raw_root_path}/cora_orig/mccallum/cora/papers"
        extractions_path = f"{self.raw_root_path}/cora_orig/mccallum/cora/extractions"

        dglg = self.graph
        dglg_feat = dglg.ndata["feat"].numpy()
        dglg_labels = dglg.ndata["label"].numpy()
        dglg_feat = (dglg_feat != 0).astype(np.int64)  # row-denormalize
        dglg_attr = np.concatenate([dglg_feat, dglg_labels[:, None]], axis=1).astype(np.int64)
        num_nodes = dglg.number_of_nodes()

        class_name_to_label = {  # follow dgl labels instead of raw labels
            "Case_Based": 5,
            "Genetic_Algorithms": 2,
            "Neural_Networks": 3,
            "Probabilistic_Methods": 4,
            "Reinforcement_Learning": 1,
            "Rule_Learning": 6,
            "Theory": 0,
        }
        label_to_class_name = {v: k for k, v in class_name_to_label.items()}

        edges = pd.read_csv(edge_path, sep="\t", header=None, names=["src", "dst"])
        attrs = pd.read_csv(text_path, sep="\t", header=None, index_col=0)

        raw_id_to_raw_idx = {raw_id: raw_idx for raw_idx, raw_id in enumerate(attrs.index)}
        raw_idx_to_raw_id = {raw_idx: raw_id for raw_id, raw_idx in raw_id_to_raw_idx.items()}

        rawg = dgl.graph((edges["src"].map(raw_id_to_raw_idx), edges["dst"].map(raw_id_to_raw_idx)))
        rawg = dgl.add_reverse_edges(rawg)
        rawg = dgl.to_simple(rawg)
        assert dglg.number_of_edges() == rawg.number_of_edges()
        rawg_degrees = rawg.in_degrees().numpy()

        attrs.iloc[:, -1] = attrs.iloc[:, -1].map(class_name_to_label)
        attrs = attrs.astype(np.int64)

        raw_feat = attrs.iloc[:, :-1].to_numpy()
        raw_labels = attrs.iloc[:, -1].to_numpy()
        raw_attr = attrs.to_numpy()

        # NOTE dgl node IDs are not aligned with raw node IDs. Here we align them.

        def build_attr_dict(attrs):
            # Build a mapping from attribute tuple to list of node indices.
            attr_dict = defaultdict(list)
            for i, row in enumerate(attrs):
                attr_dict[tuple(row)].append(i)
            return attr_dict

        # NOTE attrs are not unique
        raw_attr_dict = build_attr_dict(raw_attr)
        dgl_attr_dict = build_attr_dict(dglg_attr)

        print(f"raw attr {len(raw_attr_dict)} / {len(raw_attr)} unique")
        print(f"num of raw attr having more than one node: {sum(len(v) > 1 for v in raw_attr_dict.values())}")
        print(f"dgl attr {len(dgl_attr_dict)} / {len(dglg_attr)} unique")
        print(f"num of dgl attr having more than one node: {sum(len(v) > 1 for v in dgl_attr_dict.values())}")
        print(f"thus, exact mappings: {len(raw_attr_dict) - sum(len(v) > 1 for v in raw_attr_dict.values())}")

        # This will hold the mapping: raw_node_id -> dgl_node_id.
        raw_idx_to_dgl_idx = {}

        # First, directly map those attribute rows that appear uniquely.
        for attr, raw_indices in raw_attr_dict.items():
            dgl_indices = dgl_attr_dict.get(attr, [])
            if len(raw_indices) == 1 and len(dgl_indices) == 1:
                raw_idx_to_dgl_idx[raw_indices[0]] = dgl_indices[0]

        assert len(raw_attr_dict) - sum(len(v) > 1 for v in raw_attr_dict.values()) == len(raw_idx_to_dgl_idx)

        # Now, handle duplicates (which should be only a few).
        for attr, raw_indices in raw_attr_dict.items():
            dgl_indices = dgl_attr_dict.get(attr, [])
            if len(raw_indices) > 1 or len(dgl_indices) > 1:
                assert len(raw_indices) == len(dgl_indices)
                # For the duplicate group, try all permutations of raw indices and choose
                # the assignment that best agrees with the already determined neighbors.
                best_perm = None
                best_score = -1
                for perm in itertools.permutations(raw_indices):
                    score = 0
                    for raw_i, dgl_i in zip(perm, dgl_indices):
                        # Obtain neighbors from both graphs.
                        raw_neighbors = set(rawg.predecessors(raw_i).tolist() + rawg.successors(raw_i).tolist())
                        dgl_neighbors = set(dglg.predecessors(dgl_i).tolist() + dglg.successors(dgl_i).tolist())
                        # Increase score by the number of already-mapped neighbors that match.
                        for rn in raw_neighbors:
                            if rn in raw_idx_to_dgl_idx and raw_idx_to_dgl_idx[rn] in dgl_neighbors:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_perm = perm
                # Assign the best permutation found.
                for raw_i, dgl_i in zip(best_perm, dgl_indices):
                    raw_idx_to_dgl_idx[raw_i] = dgl_i

        dgl_idx_to_raw_idx = {dgl_idx: raw_idx for raw_idx, dgl_idx in raw_idx_to_dgl_idx.items()}
        dgl_idx_to_raw_id = {dgl_idx: raw_idx_to_raw_id[raw_idx] for dgl_idx, raw_idx in dgl_idx_to_raw_idx.items()}

        raw_feat_aligned = raw_feat[[dgl_idx_to_raw_idx[dgl_idx] for dgl_idx in range(num_nodes)]]
        raw_labels_aligned = raw_labels[[dgl_idx_to_raw_idx[dgl_idx] for dgl_idx in range(num_nodes)]]
        assert np.array_equal(raw_feat_aligned, dglg_feat)
        assert np.array_equal(raw_labels_aligned, dglg_labels)

        ######

        papers = pd.read_csv(papers_path, sep="\t", header=None, names=["raw_id", "filename", "_"])
        raw_id_to_filename = papers.groupby("raw_id")["filename"].apply(list).to_dict()
        dgl_idx_to_filename = [raw_id_to_filename[dgl_idx_to_raw_id[dgl_idx]] for dgl_idx in range(num_nodes)]
        titles = []
        abstracts = []
        missing_ti_fns = []
        missing_ab_fns = []
        for filenames in dgl_idx_to_filename:
            title = ""
            abstract = ""
            for filename in filenames:  # NOTE we use all filenames, while TAPE uses the last.
                with open(f"{extractions_path}/{filename}") as f:
                    for line in f:
                        # NOTE We match at line start, while TAPE matches in line (sometimes wrong)
                        # NOTE Abstract-found: 0 might have empty Abstract
                        if line.startswith("Title:"):
                            title = line[len("Title:") :].strip()
                        elif line.startswith("Abstract:"):
                            abstract = line[len("Abstract:") :].strip()
            if not title:
                missing_ti_fns.append(filename)
            if not abstract:
                missing_ab_fns.append(filename)
            titles.append(title)
            abstracts.append(abstract)

        print(f"papers with missing title: {len(missing_ti_fns)}")
        print(f"papers with missing abstract: {len(missing_ab_fns)}")
        self.raw_ndata["title"] = titles
        self.raw_ndata["abstract"] = abstracts
