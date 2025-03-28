from rgl.data.dataset import DownloadableRGLDataset
from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset
import torch
import os
from os.path import join as opj
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

from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
import lmdb

from rgl.utils.data_utils import extract_archive
from rgl.utils.utils import get_logger

logger = get_logger()


class MMRGLDataset(DownloadableRGLDataset):
    def __init__(self, dataset_name, dataset_root_path=None):
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
        pass

    def process(self):
        graph_root_path = self.graph_root_path
        dataset_name = self.dataset_name

        dataset = RecDataset(self)
        logger.info(str(dataset))

        train_dataset, valid_dataset, test_dataset = dataset.split()
        logger.info("\n====Training====\n" + str(train_dataset))
        logger.info("\n====Validation====\n" + str(valid_dataset))
        logger.info("\n====Testing====\n" + str(test_dataset))

        num_users = dataset.user_num
        num_items = dataset.item_num

        train_user_item_edges, train_user_items_dict, train_mask_user_items_dict = convert_freedom_dataset_to_common(
            train_dataset, num_users, [valid_dataset, test_dataset]
        )
        valid_user_item_edges, valid_user_items_dict, valid_mask_user_items_dict = convert_freedom_dataset_to_common(
            valid_dataset, num_users, [train_dataset, test_dataset]
        )
        test_user_item_edges, test_user_items_dict, test_mask_user_items_dict = convert_freedom_dataset_to_common(
            test_dataset, num_users, [train_dataset, valid_dataset]
        )

        v_feat, t_feat = None, None
        v_feat_file_path = os.path.join(graph_root_path, dataset_name, "image_feat.npy")
        t_feat_file_path = os.path.join(graph_root_path, dataset_name, "text_feat.npy")
        if os.path.isfile(v_feat_file_path):
            v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)
        if os.path.isfile(t_feat_file_path):
            t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)

        assert v_feat is not None or t_feat is not None, "Features all NONE"

        print("v_feat:", v_feat.shape, "t_feat", t_feat.shape)

        total_user_item_edges = np.concatenate(
            [train_user_item_edges, valid_user_item_edges, test_user_item_edges], axis=0
        )
        num_total_user_item_edges = len(total_user_item_edges)
        g = build_sorted_homo_graph(total_user_item_edges, num_users=num_users, num_items=num_items)

        # Create edge masks
        num_homo_nodes = num_users + num_items
        train_edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        valid_edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        test_edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        # set self loop to 1
        train_edge_mask[-num_homo_nodes:] = 1
        valid_edge_mask[-num_homo_nodes:] = 1
        test_edge_mask[-num_homo_nodes:] = 1
        # set bidirections to 1
        train_edge_mask[: len(train_user_item_edges)] = 1
        train_edge_mask[num_total_user_item_edges : num_total_user_item_edges + len(train_user_item_edges)] = 1
        valid_edge_mask[: len(valid_user_item_edges)] = 1
        valid_edge_mask[num_total_user_item_edges : num_total_user_item_edges + len(valid_user_item_edges)] = 1
        test_edge_mask[: len(test_user_item_edges)] = 1
        test_edge_mask[num_total_user_item_edges : num_total_user_item_edges + len(test_user_item_edges)] = 1

        g.edata["train_mask"] = train_edge_mask
        g.edata["valid_mask"] = valid_edge_mask
        g.edata["test_mask"] = test_edge_mask
        assert g.num_edges() == num_total_user_item_edges * 2 + num_users + num_items

        self.graph = g
        self.item_v_feat = v_feat
        self.item_t_feat = t_feat

        asin_itemidx = f"{self.graph_root_path}/{self.dataset_name}/i_id_mapping.csv"
        uid_useridx = f"{self.graph_root_path}/{self.dataset_name}/u_id_mapping.csv"
        asin_itemidx = pd.read_csv(asin_itemidx, sep="\t").set_index("itemID")
        uid_useridx = pd.read_csv(uid_useridx, sep="\t").set_index("userID")
        asins = asin_itemidx["asin"]
        user_ids = uid_useridx["user_id"]
        df_meta = get_df(opj(self.raw_root_path, "bsc"))
        iids = list(asin_itemidx.index)
        uids = list(uid_useridx.index)
        df_meta = df_meta.set_index("asin").loc[asins]
        self.raw_ndata["title"] = df_meta["title"].values
        self.raw_ndata["price"] = df_meta["price"].values
        self.raw_ndata["salesRank"] = df_meta["salesRank"].values
        self.raw_ndata["imUrl"] = df_meta["imUrl"].values
        self.raw_ndata["brand"] = df_meta["brand"].values
        self.raw_ndata["categories"] = df_meta["categories"].values
        self.raw_ndata["description"] = df_meta["description"].values


def convert_freedom_dataset_to_common(split_dataset, num_users, mask_datasets):
    split_df = split_dataset.df

    user_field = "userID"
    item_field = "itemID"

    # group by user_field

    user_item_edges = np.array(split_df[[user_field, item_field]].values, dtype=np.int64)

    # convert to dict user=>items
    user_items_dict = split_df.groupby(user_field)[item_field].apply(list).to_dict()
    for user_index in range(num_users):
        if user_index not in user_items_dict:
            user_items_dict[user_index] = []

    mask_dfs = [mask_dataset.df for mask_dataset in mask_datasets]
    mask_df = pd.concat(mask_dfs)

    mask_user_items_dict = mask_df.groupby(user_field)[item_field].apply(list).to_dict()
    for user_index in range(num_users):
        if user_index not in mask_user_items_dict:
            mask_user_items_dict[user_index] = []

    return user_item_edges, user_items_dict, mask_user_items_dict


class RecDataset(object):
    def __init__(self, rgl_dataset, df=None):
        self.rgl_dataset = rgl_dataset
        dataset_name = rgl_dataset.dataset_name
        graph_root_path = rgl_dataset.graph_root_path

        self.dataset_name = dataset_name
        self.graph_root_path = graph_root_path

        # dataframe
        self.uid_field = "userID"
        self.iid_field = "itemID"
        self.splitting_label = "x_label"

        if df is not None:
            self.df = df
            return

        # if all files exists
        inter_file_name = f"{dataset_name}.inter"
        check_file_list = [inter_file_name]
        for i in check_file_list:
            file_path = os.path.join(self.graph_root_path, dataset_name, i)
            if not os.path.isfile(file_path):
                raise ValueError("File {} not exist".format(file_path))

        # load rating file from data path?
        self.load_inter_graph(inter_file_name)
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.graph_root_path, self.dataset_name, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep="\t")
        if not self.df.columns.isin(cols).all():
            raise ValueError("File {} lost some required columns.".format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)  # no use again
            dfs.append(temp_df)

        # filtering out new users in val/test sets
        train_u = set(dfs[0][self.uid_field].values)
        for i in [1, 2]:
            dropped_inter = pd.Series(True, index=dfs[i].index)
            dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
            dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_df``, and all the other attributes the same.

        Args:
            new_df (pandas.DataFrame): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = RecDataset(self.rgl_dataset, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace."""
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num / tmp_user_num
            info.extend(
                [
                    "The number of users: {}".format(tmp_user_num),
                    "Average actions of users: {}".format(avg_actions_of_users),
                ]
            )
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num / tmp_item_num
            info.extend(
                [
                    "The number of items: {}".format(tmp_item_num),
                    "Average actions of items: {}".format(avg_actions_of_items),
                ]
            )
        info.append("The number of inters: {}".format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append("The sparsity of the dataset: {}%".format(sparsity * 100))
        return "\n".join(info)


def build_sorted_homo_graph(user_item_edges, num_users=None, num_items=None):

    user_index, item_index = user_item_edges.T

    if num_users is None:
        num_users = np.max(user_index) + 1

    if num_items is None:
        num_items = np.max(item_index) + 1

    user_index = torch.tensor(user_index)
    item_index = torch.tensor(item_index)

    num_homo_nodes = num_users + num_items
    homo_item_index = item_index + num_users

    src = torch.concat([user_index, homo_item_index, torch.arange(num_homo_nodes)], dim=0)
    dst = torch.concat([homo_item_index, user_index, torch.arange(num_homo_nodes)], dim=0)

    g = dgl.graph((src, dst), num_nodes=num_homo_nodes)

    assert g.num_edges() == src.size(0)

    # g =  dgl.add_reverse_edges(g)
    # # Different from LightGCN, MGDCF considers self-loop
    # g = dgl.add_self_loop(g)
    # g = dgl.to_simple(g)

    return g


def parse_dict(path):
    # g = gzip.open(path, "rb")
    g = open(path, "r")
    for l in g:
        yield eval(l)


def get_df(name):
    df_path = name + ".pkl"
    try:
        return pd.read_pickle(df_path)
    except:
        pass

    path = name + ".json"
    i = 0
    df = {}
    for d in parse_dict(path):
        df[i] = d
        i += 1
    df = pd.DataFrame.from_dict(df, orient="index")
    df.to_pickle(df_path)
    return df
