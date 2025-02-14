from rgl.datasets.ogb import OGBRGLDataset

dataset = OGBRGLDataset("ogbn-arxiv")

print(dataset.graph)
print(dataset.feat)
print(dataset.label)
print(dataset.train_mask)
print(dataset.val_mask)
print(dataset.test_mask)
print(dataset.raw_ndata["title"])
print(dataset.raw_ndata["abstract"])
