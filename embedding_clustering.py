
import os
import numpy as np
import json

from check_files import NameParser

USE_GREYSCALE = True

embeddings_folder = r'./extracted/embeddings'

embeddings_greyscale_folder = r'./extracted/embeddings_greyscale'

extra_embeddings_folder = r'./extraneous_buddha_statues/extra_embeddings'

extra_embeddings_greyscale_folder = r'./extraneous_buddha_statues/extra_embeddings_greyscale'

if USE_GREYSCALE:
    embeddings_folder = embeddings_greyscale_folder
    extra_embeddings_folder = extra_embeddings_greyscale_folder

# Read embeddings
embedding_files = [os.path.join(embeddings_folder, f) for f in os.listdir(embeddings_folder) if os.path.isfile(os.path.join(embeddings_folder, f))]

extra_embeddings_files = [os.path.join(extra_embeddings_folder, f) for f in os.listdir(extra_embeddings_folder) if os.path.isfile(os.path.join(extra_embeddings_folder, f))]

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

extra_embeddings = []
for extra_embedding_file in extra_embeddings_files:
    file_name = os.path.splitext(os.path.basename(extra_embedding_file))[0]
    periods.append(['extra'])

    with open(extra_embedding_file, 'r') as f:
        embedding = json.load(f)
        extra_embeddings.append(embedding)

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

extra_embeddings = np.array(extra_embeddings)

all_embeddings = np.concatenate((embeddings, extra_embeddings))

# Apply UMAP to reduce dimensionality
import umap

_umap_embeddings = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.3, metric='correlation').fit_transform(all_embeddings)

# Plot the clusters
import matplotlib.pyplot as plt

def draw_embeddings_3d(umap_embeddings):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(umap_embeddings)):
        x, y, z = umap_embeddings[i]
        period:str = periods[i]
        is_wei = False
        is_qi = False
        is_extra = False
        for p in period:
            if 'wei' in p.lower():
                is_wei = True
                break
        if not is_wei:
            for p in period:
                if 'qi' in p.lower():
                    is_qi = True
                    break
        if not is_wei and not is_qi:
            for p in period:
                if 'extra' in p.lower():
                    is_extra = True
                    break

        if is_wei:
            ax.scatter(x, y, z, c='r', marker='o')
        elif is_qi:
            ax.scatter(x, y, z, c='g', marker='o')
        elif is_extra:
            ax.scatter(x, y, z, c='y', marker='o')
        else:
            #ax.scatter(x, y, z, c='b', marker='o')
            pass

    plt.show()

def draw_embeddings_2d(umap_embeddings):
    for i in range(len(umap_embeddings)):
        x, y = umap_embeddings[i]
        period:str = periods[i]
        is_wei = False
        is_qi = False
        is_extra = False
        for p in period:
            if 'wei' in p.lower():
                is_wei = True
                break
        if not is_wei:
            for p in period:
                if 'qi' in p.lower():
                    is_qi = True
                    break
        if not is_wei and not is_qi:
            for p in period:
                if 'extra' in p.lower():
                    is_extra = True
                    break

        if is_wei:
            plt.scatter(x, y, c='r', marker='o')
        elif is_qi:
            plt.scatter(x, y, c='g', marker='o')
        elif is_extra:
            plt.scatter(x, y, c='y', marker='o')
        else:
            #plt.scatter(x, y, c='b', marker='o')
            pass

    plt.show()


draw_embeddings_2d(_umap_embeddings)