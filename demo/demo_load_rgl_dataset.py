from rgl.datasets.ogb import OGBRGLDataset
from rgl.datasets.citation_graph import CoraRGLDataset
from rgl.datasets.pubmed import PubmedRGLDataset
from rgl.datasets.citeseer import CiteseerRGLDataset

dataset = CoraRGLDataset("./dataset/cora")
print("loading cora")
print(dataset.graph)
print(dataset.feat.shape)
print(dataset.label)
print(dataset.train_mask)
print(dataset.val_mask)
print(dataset.test_mask)
i = 1
print("Title:", dataset.raw_ndata["title"][i])
print("Abstract:", dataset.raw_ndata["abstract"][i])

print("-" * 50)

dataset = OGBRGLDataset("ogbn-arxiv", "./dataset/ogbn-arxiv")
print("loading ogbn-arxiv")
print(dataset.graph)
print(dataset.feat.shape)
print(dataset.label)
print(dataset.train_mask)
print(dataset.val_mask)
print(dataset.test_mask)
print(dataset.raw_ndata["title"][0])
print(dataset.raw_ndata["abstract"][0])

print("-" * 50)

dataset = OGBRGLDataset("ogbn-products", "./dataset/ogbn-products")
print("loading ogbn-products")
print(dataset.graph)
print(dataset.feat.shape)
print(dataset.label)
print(dataset.train_mask)
print(dataset.val_mask)
print(dataset.test_mask)
print(dataset.raw_ndata["title"][0])

print("-" * 50)

dataset = PubmedRGLDataset("./dataset/pubmed")
print("loading pubmed")
print(dataset.graph)
print(dataset.feat.shape)
print(dataset.label)
print(dataset.train_mask)
print(dataset.val_mask)
print(dataset.test_mask)

print("-" * 50)

dataset = CiteseerRGLDataset("./dataset/citeseer")
print("loading pubmed")
print(dataset.graph)
print(dataset.feat.shape)
print(dataset.label)
print(dataset.train_mask)
print(dataset.val_mask)
print(dataset.test_mask)

print("-" * 50)
