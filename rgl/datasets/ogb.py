from rgl.data.dataset import DownloadableRGLDataset

# from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset
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


def idx_to_mask(idx, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[idx] = 1
    return mask


class OGBRGLDataset(DownloadableRGLDataset):

    def __init__(self, dataset_name, dataset_root_path=None):
        """
        OGB Node Property Prediction Datasets: https://ogb.stanford.edu/docs/nodeprop/

        :param dataset_name: "ogbn-arxiv" | "ogbn-products" | "ogbn-proteins" | "ogbn-papers100M" | "ogbn-mag"
        :param dataset_root_path:
        """

        if dataset_name == "ogbn-arxiv":
            download_urls = ["https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"]
            download_file_name = ["titleabs.tsv.gz"]
        elif dataset_name == "ogbn-products":
            # manually download: https://drive.google.com/file/d/1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN/view?usp=sharing
            download_urls = []
            download_file_name = []
        elif dataset_name == "ogbn-papers100M":
            # manually download (ogb easy to fail): http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
            download_urls = ["https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/paperinfo.zip"]  # TODO
            download_file_name = ["paperinfo.zip"]

        super().__init__(
            dataset_name=dataset_name,
            download_urls=download_urls,
            download_file_names=download_file_name,
            cache_name=None,
            dataset_root_path=dataset_root_path,
        )

    def download_graph(self, dataset_name, graph_root_path):
        dataset = DglNodePropPredDataset(name=dataset_name, root=graph_root_path)
        split_idx = dataset.get_idx_split()
        graph, label = dataset[0]
        n = graph.number_of_nodes()
        self.backend_dataset = dataset
        self.graph = graph
        self.feat = graph.ndata["feat"]
        self.label = label.flatten()
        self.train_mask = idx_to_mask(split_idx["train"], n)
        self.val_mask = idx_to_mask(split_idx["valid"], n)
        self.test_mask = idx_to_mask(split_idx["test"], n)
        self.graph.ndata["label"] = self.label

    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    def process(self):
        dataset_lower = self.dataset_name.replace("-", "_")
        if self.dataset_name == "ogbn-arxiv":
            mapping_path = f"{self.graph_root_path}/{dataset_lower}/mapping/nodeidx2paperid.csv.gz"
            nodeidx2paperid = pd.read_csv(mapping_path)

            cate_path = f"{self.graph_root_path}/{dataset_lower}/mapping/labelidx2arxivcategeory.csv.gz"
            cate = pd.read_csv(cate_path)
            cate = dict(zip(cate["label idx"], cate["arxiv category"]))
            # reformat: arxiv cs na to cs.NA
            cate = {k: f"{v.split()[1]}.{v.split()[2].upper()}" for k, v in cate.items()}

            titleabs_path = f"{self.raw_root_path}/titleabs.tsv"
            titleabs = pd.read_csv(titleabs_path, sep="\t", header=None)
            titleabs = titleabs.set_index(0)
            titleabs = titleabs.loc[nodeidx2paperid["paper id"]]
            total_missing = titleabs.isnull().sum().sum()
            assert total_missing == 0
            title = titleabs[1].values
            abstract = titleabs[2].values
            self.raw_ndata["title"] = title
            self.raw_ndata["abstract"] = abstract
            self.label2categeory = cate

        elif self.dataset_name == "ogbn-products":
            mapping_path = f"{self.graph_root_path}/{dataset_lower}/mapping/nodeidx2asin.csv"
            nodeidx2asin = pd.read_csv(mapping_path)

            # find title using trn/tst.json instead of Yf.txt
            trnjson_path = f"{self.raw_root_path}/Amazon-3M.raw/trn.json.gz"
            tstjson_path = f"{self.raw_root_path}/Amazon-3M.raw/tst.json.gz"
            trnjson_df = pd.read_json(trnjson_path, lines=True)
            tstjson_df = pd.read_json(tstjson_path, lines=True)
            title = pd.concat([trnjson_df, tstjson_df])
            title = title.set_index("uid")
            title = title.loc[nodeidx2asin["asin"]]
            total_missing = title.isnull().sum().sum()
            assert total_missing == 0, f"missing {total_missing} titles"
            title = title["title"].values
            self.raw_ndata["title"] = title

        elif self.dataset_name == "ogbn-papers100M":  # FIXME link fails
            mapping_path = f"{self.graph_root_path}/{dataset_lower}/mapping/nodeidx2paperid.csv.gz"
            nodeidx2paperid = pd.read_csv(mapping_path)
            map_pid_set = set(nodeidx2paperid["paper id"])

            title_path = f"{self.raw_root_path}/paperinfo/idx_title.tsv"
            title = pd.read_csv(title_path, sep="\t", header=None).set_index(0)
            title_pid_set = set(title.index)
            title = title.reindex(nodeidx2paperid["paper id"])
            total_missing = title.isnull().sum().sum()
            if total_missing > 0:
                # missing 108905842 titles, total 111059956 papers
                # idx_title.tsv lines: 110549007
                # idx_abs.tsv lines: 85566493
                # missing 109388360 abstracts, total 111059956 papers
                print(f"missing {total_missing} titles, total {len(nodeidx2paperid)} papers")
                diff = map_pid_set - title_pid_set
                print(f"diff titles: {len(diff)}")
                print(f"diff titles: {diff}")
            title = title[1].values
            self.raw_ndata["title"] = title

            abstract_path = f"{self.raw_root_path}/paperinfo/idx_abs.tsv"
            abstract = pd.read_csv(abstract_path, sep="\t", header=None).set_index(0)
            abstract = abstract.reindex(nodeidx2paperid["paper id"])
            total_missing = abstract.isnull().sum().sum()
            if total_missing > 0:
                print(f"missing {total_missing} abstracts, total {len(nodeidx2paperid)} papers")
            abstract = abstract[1].values
            self.raw_ndata["abstract"] = abstract


# class OGBPyGRGLDataset(DownloadableRGLDataset):
#     """
#     OGB Node Property Prediction Datasets using a PyG backend.
#     Datasets: "ogbn-arxiv" | "ogbn-products" | "ogbn-papers100M"
#     """

#     def __init__(self, dataset_name, dataset_root_path=None):
#         if dataset_name == "ogbn-arxiv":
#             download_urls = ["https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"]
#             download_file_name = ["titleabs.tsv.gz"]
#         elif dataset_name == "ogbn-products":
#             # manually download: https://drive.google.com/file/d/1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN/view?usp=sharing
#             download_urls = []
#             download_file_name = []
#         elif dataset_name == "ogbn-papers100M":
#             # manually download (ogb easy to fail): http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
#             download_urls = ["https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/paperinfo.zip"]  # TODO
#             download_file_name = ["paperinfo.zip"]
#         else:
#             raise ValueError(f"Dataset {dataset_name} not supported.")

#         super().__init__(
#             dataset_name=dataset_name,
#             download_urls=download_urls,
#             download_file_names=download_file_name,
#             cache_name=None,
#             dataset_root_path=dataset_root_path,
#         )

#     def download_graph(self, dataset_name, graph_root_path):
#         # Load the dataset via the PyG API.
#         dataset = PygNodePropPredDataset(name=dataset_name, root=graph_root_path)
#         split_idx = dataset.get_idx_split()
#         # The pyg dataset returns a Data object.
#         data = dataset[0]
#         # In PyG, features are typically stored in data.x and labels in data.y.
#         n = data.x.size(0)
#         self.backend_dataset = dataset
#         self.graph = data
#         self.feat = data.x
#         # Flatten the label tensor to match our expected shape.
#         self.label = data.y.view(-1)
#         self.train_mask = idx_to_mask(split_idx["train"], n)
#         self.val_mask = idx_to_mask(split_idx["valid"], n)
#         self.test_mask = idx_to_mask(split_idx["test"], n)

#     def process(self):
#         # Process raw data to attach titles and abstracts.
#         dataset_lower = self.dataset_name.replace("-", "_")
#         if self.dataset_name == "ogbn-arxiv":
#             mapping_path = f"{self.graph_root_path}/{dataset_lower}/mapping/nodeidx2paperid.csv.gz"
#             nodeidx2paperid = pd.read_csv(mapping_path)
#             titleabs_path = f"{self.raw_root_path}/titleabs.tsv"
#             titleabs = pd.read_csv(titleabs_path, sep="\t", header=None)
#             titleabs = titleabs.set_index(0)
#             titleabs = titleabs.loc[nodeidx2paperid["paper id"]]
#             total_missing = titleabs.isnull().sum().sum()
#             assert total_missing == 0, f"Found {total_missing} missing values in titleabs."
#             title = titleabs[1].values
#             abstract = titleabs[2].values
#             self.raw_ndata["title"] = title
#             self.raw_ndata["abstract"] = abstract

#         elif self.dataset_name == "ogbn-products":
#             mapping_path = f"{self.graph_root_path}/{dataset_lower}/mapping/nodeidx2asin.csv"
#             nodeidx2asin = pd.read_csv(mapping_path)
#             # Find titles using trn/tst.json instead of Yf.txt.
#             trnjson_path = f"{self.raw_root_path}/Amazon-3M.raw/trn.json.gz"
#             tstjson_path = f"{self.raw_root_path}/Amazon-3M.raw/tst.json.gz"
#             trnjson_df = pd.read_json(trnjson_path, lines=True)
#             tstjson_df = pd.read_json(tstjson_path, lines=True)
#             title_df = pd.concat([trnjson_df, tstjson_df])
#             title_df = title_df.set_index("uid")
#             title_df = title_df.loc[nodeidx2asin["asin"]]
#             total_missing = title_df.isnull().sum().sum()
#             assert total_missing == 0, f"Missing {total_missing} titles in products dataset."
#             title = title_df["title"].values
#             self.raw_ndata["title"] = title

#         elif self.dataset_name == "ogbn-papers100M":
#             mapping_path = f"{self.graph_root_path}/{dataset_lower}/mapping/nodeidx2paperid.csv.gz"
#             nodeidx2paperid = pd.read_csv(mapping_path)
#             map_pid_set = set(nodeidx2paperid["paper id"])
#             title_path = f"{self.raw_root_path}/paperinfo/idx_title.tsv"
#             title = pd.read_csv(title_path, sep="\t", header=None).set_index(0)
#             title_pid_set = set(title.index)
#             title = title.reindex(nodeidx2paperid["paper id"])
#             total_missing = title.isnull().sum().sum()
#             if total_missing > 0:
#                 print(f"Missing {total_missing} titles, total {len(nodeidx2paperid)} papers")
#                 diff = map_pid_set - title_pid_set
#                 print(f"Difference in titles: {len(diff)}")
#                 print(f"IDs missing: {diff}")
#             title = title[1].values
#             self.raw_ndata["title"] = title

#             abstract_path = f"{self.raw_root_path}/paperinfo/idx_abs.tsv"
#             abstract = pd.read_csv(abstract_path, sep="\t", header=None).set_index(0)
#             abstract = abstract.reindex(nodeidx2paperid["paper id"])
#             total_missing = abstract.isnull().sum().sum()
#             if total_missing > 0:
#                 print(f"Missing {total_missing} abstracts, total {len(nodeidx2paperid)} papers")
#             abstract = abstract[1].values
#             self.raw_ndata["abstract"] = abstract
