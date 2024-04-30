
import os
import numpy as np
import json

from check_files import NameParser

embeddings_folder = r'./extracted/embeddings'

# Read embeddings
embedding_files = [os.path.join(embeddings_folder, f) for f in os.listdir(embeddings_folder) if os.path.isfile(os.path.join(embeddings_folder, f))]

embeddings = []
periods = []
types = []
for embedding_file in embedding_files:
    file_name = os.path.splitext(os.path.basename(embedding_file))[0]
    attrs = NameParser(file_name)
    periods.append(attrs.period)
    types.append(attrs.type)

    with open(embedding_file, 'r') as f:
        embedding = json.load(f)
        embeddings.append(embedding)

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Apply UMAP to reduce dimensionality
import umap

umap_embeddings = umap.UMAP(n_neighbors=5, n_components=3, min_dist=0.3, metric='correlation').fit_transform(embeddings)

# Plot the clusters
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(umap_embeddings)):
    x, y, z = umap_embeddings[i]
    period:str = periods[i]
    is_wei = False
    is_qi = False
    for p in period:
        if 'wei' in p.lower():
            is_wei = True
            break
    if not is_wei:
        for p in period:
            if 'qi' in p.lower():
                is_qi = True
                break

    if is_wei:
        ax.scatter(x, y, z, c='r', marker='o')
    elif is_qi:
        ax.scatter(x, y, z, c='g', marker='o')
    else:
        ax.scatter(x, y, z, c='b', marker='o')

plt.show()
