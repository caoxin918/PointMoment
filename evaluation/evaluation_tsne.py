from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './')

from data.modelnet40 import ModelNet40SVM
from data.modelnet10 import Dataset

def map_label_to_category(x):
    mapping = {
        0: 'bathtub',
        1: 'bed',
        2: 'chair',
        3: 'desk',
        4: 'dresser',
        5: 'monitor',
        6: 'night_stand',
        7: 'sofa',
        8: 'table',
        9: 'toilet'
    }

    return mapping[x]

file_features = '../results/three_order.npy'

root = '../datasets'
dataset_name = 'modelnet10'
split = 'test'

datasets = Dataset(root=root, dataset_name=dataset_name, num_points=2048, split=split)

labels = datasets.label

features = np.load(file_features)

tsne = TSNE(n_components=2, random_state=2000, metric="cosine")

tsne_results1 = tsne.fit_transform(features)

df_subset = {}
df_subset['x'] = tsne_results1[:, 0]
df_subset['y'] = tsne_results1[:, 1]
df_subset['y1'] = np.array(labels).squeeze()

df_subset = pd.DataFrame({
    'x': df_subset['x'],
    'y': df_subset['y'],
    'label': df_subset['y1']
})

df_subset['Category'] = df_subset['label'].apply(map_label_to_category)

plt.figure(figsize=(10, 7), dpi=100)
sns.scatterplot(
    x="x", y="y",
    hue="Category",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend=False,
    alpha=0.99,
    s=80,
)

# plt.legend(loc='center right', fontsize=15, bbox_to_anchor=(1.3, 0.5), borderaxespad=0.2)
# plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.axis('on')
plt.xlabel('')
plt.ylabel('')
plt.title('Three-Order', fontsize=35)
plt.tight_layout()
plt.savefig('../result/fig/three_order.png', dpi=100)

plt.show()