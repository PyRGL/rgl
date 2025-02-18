from rgl.data.dataset import DownloadableRGLDataset
from ogb.nodeproppred import DglNodePropPredDataset
import torch
import os
import patoolib

# from keras.utils.data_utils import _extract_archive
# from shutil import unpack_archive
from rgl.utils.data_utils import extract_archive
import pandas as pd


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
            # manually download:
            # https://drive.google.com/file/d/1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN/view?usp=sharing
            download_urls = []
            download_file_name = []

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
        self.graph = graph
        self.feat = graph.ndata["feat"]
        self.label = label.flatten()
        self.train_mask = idx_to_mask(split_idx["train"], n)
        self.val_mask = idx_to_mask(split_idx["valid"], n)
        self.test_mask = idx_to_mask(split_idx["test"], n)

    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    def process(self):
        dataset_lower = self.dataset_name.lower().replace("-", "_")
        if self.dataset_name == "ogbn-arxiv":
            mapping_path = f"{self.graph_root_path}/{dataset_lower}/mapping/nodeidx2paperid.csv.gz"
            nodeidx2paperid = pd.read_csv(mapping_path)

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
