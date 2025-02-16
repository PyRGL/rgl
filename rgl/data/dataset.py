# coding=utf-8
import os

# from keras.utils.file_utils import _extract_archive
# from shutil import unpack_archive
from rgl.utils import extract_archive
from torch.hub import download_url_to_file

# from tf_geometric.utils.data_utils import download_file, load_cache, save_cache
from shutil import copy
from os.path import join as opj
import pickle

DEFAULT_DATASETS_ROOT = "dataset"


def get_dataset_root_path(
    dataset_root_path=None, dataset_name=None, datasets_root_path=DEFAULT_DATASETS_ROOT, mkdir=False
):
    if dataset_root_path is None:
        dataset_root_path = os.path.join(datasets_root_path, dataset_name)
    dataset_root_path = os.path.abspath(dataset_root_path)

    if mkdir:
        os.makedirs(dataset_root_path, exist_ok=True)
    return dataset_root_path


# class RGLDataset(object):
#     def __init__(self, dataset_name, graph):
#         self.dataset_name = dataset_name
#         self.graph = graph


class DownloadableRGLDataset(object):

    def __init__(
        self, dataset_name, download_urls=None, download_file_names=None, cache_name="cache.p", dataset_root_path=None
    ):
        self.dataset_name = dataset_name
        self.dataset_root_path = get_dataset_root_path(dataset_root_path, dataset_name)
        self.download_urls = download_urls
        self.download_file_names = download_file_names

        self.graph_root_path = opj(self.dataset_root_path, "graph")

        self.download_root_path = os.path.join(self.dataset_root_path, "download")
        self.raw_root_path = os.path.join(self.dataset_root_path, "raw")
        self.processed_root_path = os.path.join(self.dataset_root_path, "processed")

        if download_urls is not None:
            self.download_file_paths = [
                os.path.join(self.download_root_path, download_file_name) for download_file_name in download_file_names
            ]
        else:
            self.download_file_path = None

        self.cache_path = None if cache_name is None else os.path.join(self.processed_root_path, cache_name)

        self.build_dirs()

        print(dataset_name, self.graph_root_path)
        self.raw_ndata = {}
        self.download_graph(dataset_name, self.graph_root_path)
        self.download()
        self.extract_raw()
        self.process()

    @property
    def cache_enabled(self):
        return self.cache_path is not None

    def build_dirs(self):
        os.makedirs(self.download_root_path, exist_ok=True)
        os.makedirs(self.raw_root_path, exist_ok=True)
        os.makedirs(self.processed_root_path, exist_ok=True)

    def download(self):
        print("download urls: ", self.download_urls)
        print("download file paths: ", self.download_file_paths)
        for url, path in zip(self.download_urls, self.download_file_paths):
            if os.path.exists(path):
                print("file exists: {}, ignore".format(path))
                continue
            download_url_to_file(url, path)

    def extract_raw(self):
        if len(os.listdir(self.raw_root_path)) == 0:
            if self.download_file_path.endswith(".npz"):
                copy(self.download_file_path, os.path.join(self.raw_root_path, self.download_file_names))
            else:
                extract_archive(self.download_file_path, self.raw_root_path)
        else:
            print("raw data exists: {}, ignore".format(self.raw_root_path))

    def process(self):
        pass

    def download_graph(self):
        pass
