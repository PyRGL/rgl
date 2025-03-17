from dgl.data import CiteseerGraphDataset
import torch
import dgl

class CiteseerRGLDataset:
    def __init__(self, dataset_root_path=None):
        """
        Use DGL's CiteSeer dataset
        
        :param dataset_root_path: Path to store the dataset
        :param text_file_path: Optional path to a CSV file containing paper_id and text (title/abstract)
        """
        self.dataset_root_path = dataset_root_path if dataset_root_path else "dataset/citeseer"
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
        Process the CiteSeer dataset
        """
        # Load CiteSeer dataset
        dataset = CiteseerGraphDataset(raw_dir=self.dataset_root_path)
        
        # CiteSeer dataset contains only one graph
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

