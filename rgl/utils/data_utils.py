from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import pandas as pd
import tarfile
import zipfile
import gzip
import patoolib

import torch
import dgl
import dgl.data
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl import RowFeatNormalizer
from ogb.nodeproppred import DglNodePropPredDataset

from tqdm import tqdm

from rgl.utils import get_logger

logger = get_logger()


def extract_archive(file_path, output_dir):
    """
    Extracts various archive formats (.zip, .tar, .tar.gz, .tgz, .gz) using patoolib.

    Args:
        file_path (str): Path to the compressed file.
        output_dir (str): Destination folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    # If the output directory already contains files, skip extraction.
    if os.listdir(output_dir):
        return

    try:
        # patoolib will automatically detect the archive type and use the appropriate tool.
        patoolib.extract_archive(file_path, outdir=output_dir, interactive=False)
    except Exception as e:
        raise ValueError(f"Extraction failed for {file_path}: {str(e)}") from e

    print(f"Extraction complete: {file_path} → {output_dir}")


# def extract_archive(file_path, output_dir):
#     """
#     Extracts various archive formats (.zip, .tar, .tar.gz, .tgz, .gz) with a progress bar.

#     Args:
#         file_path (str): Path to the compressed file.
#         output_dir (str): Destination folder.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     if len(os.listdir(output_dir)) > 0:
#         return

#     # Handle ZIP files
#     if file_path.endswith(".zip"):
#         with zipfile.ZipFile(file_path, "r") as zip_ref:
#             file_list = zip_ref.namelist()
#             with tqdm(total=len(file_list), desc="Extracting ZIP", unit="file") as bar:
#                 for file in file_list:
#                     zip_ref.extract(file, output_dir)
#                     bar.update(1)

#     # Handle TAR, TAR.GZ, TGZ files
#     elif file_path.endswith((".tar", ".tar.gz", ".tgz")):
#         with tarfile.open(file_path, "r:*") as tar_ref:
#             members = tar_ref.getmembers()
#             with tqdm(total=len(members), desc="Extracting TAR", unit="file") as bar:
#                 for member in members:
#                     tar_ref.extract(member, path=output_dir)
#                     bar.update(1)

#     # Handle GZ files (single file compression) TODO sometimes need "tar -xzf" to be correct
#     elif file_path.endswith(".gz") and not file_path.endswith(".tar.gz"):
#         output_file = os.path.join(output_dir, os.path.basename(file_path)[:-3])
#         file_size = os.path.getsize(file_path)  # Get file size for progress tracking
#         chunk_size = 1024 * 1024  # 1MB

#         with gzip.open(file_path, "rb") as f_in, open(output_file, "wb") as f_out, tqdm(
#             total=file_size, unit="B", unit_scale=True, desc="Extracting GZ"
#         ) as bar:
#             while chunk := f_in.read(chunk_size):
#                 f_out.write(chunk)
#                 bar.update(len(chunk))

#     else:
#         raise ValueError(f"Unsupported file format: {file_path}")

#     print(f"Extraction complete: {file_path} → {output_dir}")


def download(url, path=None):
    if not os.path.exists(path):
        os.system(f"wget {url} -O {path}")
        if path.endswith(".gz"):
            os.system(f"tar -xzf {path}")
    path = path.replace(".gz", "")
    return path


def idx_to_mask(idx, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[idx] = 1
    return mask


def mask_to_idx(mask):
    return torch.where(mask)[0]


def to_bidir(g):
    print(f"num_edges (raw): {g.num_edges()}")
    g = dgl.remove_self_loop(g)
    print(f"num_edges (remove self loop): {g.num_edges()}")
    g = dgl.add_reverse_edges(g)
    print(f"num_edges (add reverse edges): {g.num_edges()}")
    g = dgl.to_simple(g)
    n_edge = g.num_edges() // 2
    print(f"num_edges (simple graph)(directed): {g.num_edges()}")
    print(f"num_edges (simple graph)(undirected): {n_edge}")
    return g


# NOTE cora, citeseer, pubmed: already row-normalized
def norm_feat(graph):
    transform = RowFeatNormalizer(subtract_min=True, node_feat_names=["feat"])
    graph = transform(graph)
    print(f"END normalize")
    return graph


def std_feat(graph):
    features = graph.ndata["feat"]
    mask_train = graph.ndata["train_mask"]
    features_train = features[mask_train]
    scaler = StandardScaler()
    scaler.fit(features_train)
    std_features = scaler.transform(features)
    graph.ndata["feat"] = torch.tensor(std_features, dtype=graph.ndata["feat"].dtype)
    logger.info(f"END standardize")
    return graph


def get_syn(path):
    graphs, _ = load_graphs(path)
    graph = graphs[0]
    graph = to_bidir(graph)
    graph.ndata["train_mask"] = torch.ones(graph.number_of_nodes(), dtype=torch.bool)

    # norm/std feat
    norm_feat_flag = False
    std_feat_flag = False
    if "ogbn-arxiv" in path:
        std_feat_flag = True

    if norm_feat_flag:
        graph = norm_feat(graph)
    if std_feat_flag:
        graph = std_feat(graph)

    return graph


class RGLDataset(object):
    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph


# TODO text-attributed/MM datasets
def get_dataset(name, args=None):
    norm_feat_flag = False
    std_feat_flag = False
    if args:
        norm_feat_flag = args.norm_feat
        std_feat_flag = args.std_feat

    if name == "ogbn-arxiv":
        std_feat_flag = True

    if name in ["cora"]:
        dataset = dgl.data.CoraGraphDataset()
        graph = dataset[0]
    elif name in ["citeseer"]:
        dataset = dgl.data.CiteseerGraphDataset()
        graph = dataset[0]
    elif name in ["pubmed"]:
        dataset = dgl.data.PubmedGraphDataset()
        graph = dataset[0]
    elif name in ["ogbn-arxiv", "ogbn-products"]:
        dataset = DglNodePropPredDataset(name=name)
        split_idx = dataset.get_idx_split()
        graph, label = dataset[0]
        n = graph.number_of_nodes()
        graph.ndata["label"] = label.flatten()
        graph.ndata["train_mask"] = idx_to_mask(split_idx["train"], n)
        graph.ndata["val_mask"] = idx_to_mask(split_idx["valid"], n)
        graph.ndata["test_mask"] = idx_to_mask(split_idx["test"], n)

        mapping_path = f"dataset/{name.replace('-', '_')}/mapping/nodeidx2paperid.csv"
        if not os.path.exists(mapping_path):
            os.system(f"gzip -dk {mapping_path}.gz")
        nodeidx2paperid = pd.read_csv(f"dataset/{name.replace('-', '_')}/mapping/nodeidx2paperid.csv")

        # add titleabs
        titleabs_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
        titleabs_path = f"dataset/{name.replace('-', '_')}/titleabs.tsv.gz"
        titleabs_path = download(titleabs_url, titleabs_path)
        titleabs = pd.read_csv(titleabs_path, sep="\t", header=None)

        # add titleabs to graph.ndate, using the mapping
        titleabs = titleabs.set_index(0)
        titleabs = titleabs.loc[nodeidx2paperid["paper id"]]
        total_missing = titleabs.isnull().sum().sum()
        assert total_missing == 0
        title = titleabs[1].values
        abstract = titleabs[2].values
        rgldata = RGLDataset(graph)
        rgldata.title = title
        rgldata.abstract = abstract

    elif name in ["flickr"]:
        dataset = dgl.data.FlickrDataset()
        graph = dataset[0]
        graph.ndata["train_mask"] = graph.ndata["train_mask"].bool()
        graph.ndata["val_mask"] = graph.ndata["val_mask"].bool()
        graph.ndata["test_mask"] = graph.ndata["test_mask"].bool()
    elif name in ["yelp"]:
        dataset = dgl.data.YelpDataset()
        graph = dataset[0]
    elif name in ["reddit"]:
        dataset = dgl.data.RedditDataset()
        graph = dataset[0]
    elif name in ["computer"]:
        dataset = dgl.data.AmazonCoBuyComputerDataset()
        graph = dataset[0]
    elif name in ["wikics"]:
        dataset = dgl.data.WikiCSDataset()
        graph = dataset[0]
        graph.ndata["train_mask"] = graph.ndata["train_mask"].bool()
        graph.ndata["val_mask"] = graph.ndata["val_mask"].bool()
        graph.ndata["test_mask"] = graph.ndata["test_mask"].bool()
    elif name in ["photo"]:
        dataset = dgl.data.AmazonCoBuyPhotoDataset()
        graph = dataset[0]
    elif name in ["amazon-ratings"]:
        dataset = dgl.data.AmazonRatingsDataset()
        graph = dataset[0]
    elif name in ["tfinance"]:
        graph = BwgnnDataset(name="tfinance").graph
    else:
        raise NotImplementedError

    # set splits
    if "train_mask" not in graph.ndata:
        n = graph.number_of_nodes()
        idx = torch.randperm(n)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        graph.ndata["train_mask"] = idx_to_mask(idx[:n_train], n)
        graph.ndata["val_mask"] = idx_to_mask(idx[n_train : n_train + n_val], n)
        graph.ndata["test_mask"] = idx_to_mask(idx[n_train + n_val :], n)
        logger.info(f"END random split for {name}")

    if graph.ndata["train_mask"].ndim > 1:
        graph.ndata["train_mask"] = graph.ndata["train_mask"][:, 0]
        logger.info(f"END pick first train mask for {name}")
    if graph.ndata["val_mask"].ndim > 1:
        graph.ndata["val_mask"] = graph.ndata["val_mask"][:, 0]
        logger.info(f"END pick first val mask for {name}")
    if graph.ndata["test_mask"].ndim > 1:
        graph.ndata["test_mask"] = graph.ndata["test_mask"][:, 0]
        logger.info(f"END pick first test mask for {name}")

    # stats
    logger.info(graph)
    logger.info(f"n_clses: {graph.ndata['label'].unique().shape[0]}")
    label_counts = graph.ndata["label"].bincount()
    label_counts = {k: v.item() for k, v in enumerate(label_counts)}
    logger.info(f"label_counts: {label_counts}")
    logger.info(
        f"train/val/test: {graph.ndata['train_mask'].sum().item()} / {graph.ndata['val_mask'].sum().item()} / {graph.ndata['test_mask'].sum().item()}"
    )

    # PREPROCESS GRAPH
    if name not in ["ogbn-products", "reddit"]:  # slow
        graph = to_bidir(graph)

    if norm_feat_flag:
        graph = norm_feat(graph)
    if std_feat_flag:
        graph = std_feat(graph)

    return graph


class BwgnnDataset:
    def __init__(self, name="tfinance", homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == "tfinance":
            graph, label_dict = load_graphs("dataset/tfinance")
            graph = graph[0]
            graph.ndata["label"] = graph.ndata["label"].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs("dataset/tfinance")
                graph = graph[0]
                feat = graph.ndata["feature"].numpy()
                anomaly_id = graph.ndata["label"][:, 1].nonzero().squeeze(1)
                feat = (feat - np.average(feat, 0)) / np.std(feat, 0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata["feature"] = torch.tensor(feat)
                graph.ndata["label"] = graph.ndata["label"].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs("dataset/tfinance")
                graph = graph[0]
                feat = graph.ndata["feature"].numpy()
                anomaly_id = list(graph.ndata["label"][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata["label"][:, 0].nonzero().squeeze(1))
                label = graph.ndata["label"].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random

                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0

        elif name == "tsocial":
            graph, label_dict = load_graphs("dataset/tsocial")
            graph = graph[0]

        elif name == "yelp":
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(
                    dataset[0], ndata=["feature", "label", "train_mask", "val_mask", "test_mask"]
                )
                graph = dgl.add_self_loop(graph)
        elif name == "amazon":
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(
                    dataset[0], ndata=["feature", "label", "train_mask", "val_mask", "test_mask"]
                )
                graph = dgl.add_self_loop(graph)
        else:
            print("no such dataset")
            exit(1)

        graph.ndata["label"] = graph.ndata["label"].long().squeeze(-1)
        graph.ndata["feature"] = graph.ndata["feature"].float()
        graph.ndata["feat"] = graph.ndata.pop("feature")

        self.graph = graph


if __name__ == "__main__":
    g = get_dataset("ogbn-arxiv")
