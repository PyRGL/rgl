from dgl.data import PubmedGraphDataset
import torch
import dgl

class PUBMEDRGLDataset:
    def __init__(self, dataset_root_path=None):
        """
        Use DGL's PubMed dataset
        
        :param dataset_root_path: Path to store the dataset
        """
        self.dataset_root_path = dataset_root_path if dataset_root_path else "dataset/pubmed"
        self.raw_ndata = {}
        self.process()

    def idx_to_mask(self, idx, size):
        """
        Convert indices to a boolean mask
        
        :param idx: Indices to set as True
        :param size: Total size of the mask
        :return: Boolean mask tensor
        """
        mask = torch.zeros(size, dtype=torch.bool)
        mask[idx] = 1
        return mask

    def process(self):
        """
        Process the PubMed dataset
        """
        # Load PubMed dataset
        dataset = PubmedGraphDataset(self.dataset_root_path)
        
        # PubMed dataset contains only one graph
        graph = dataset[0]
        n = graph.number_of_nodes()
        
        # Store graph and related data
        self.graph = graph
        self.feat = graph.ndata['feat']  # Node features
        self.label = graph.ndata['label']  # Node labels
        
        # Get train/valid/test splits
        self.train_mask = graph.ndata['train_mask']
        self.val_mask = graph.ndata['val_mask']
        self.test_mask = graph.ndata['test_mask']
        
        # Store original dataset
        self.backend_dataset = dataset
