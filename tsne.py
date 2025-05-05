import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os.path as osp


def plot_tsne(feats, labels, n_classes, save_path='', num_samples=1000):
    plt.rcParams['font.family'] = 'Times New Roman'
    feats = feats.cpu().detach().numpy()
    # labels = labels.cpu().detach().numpy()

    if num_samples:
        indices = np.random.choice(len(labels), size=num_samples, replace=False)
        feats = feats[indices]
        labels = labels[indices]

    num_class = n_classes + 1
    print(f'class number: {num_class}')

    X_tsne = TSNE(n_components=2, init='pca',
                  learning_rate='auto').fit_transform(feats)

    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    colormap = plt.cm.gist_ncar  #nipy_spectral, Set1,Paired
    colorst = [colormap(i) for i in np.linspace(0, 0.9, num_class)]
    category_list = ['dig', 'knock', 'shake', 'unknown']
    for i in range(num_class):
        inds = np.where(labels == i)[0]
        plt.scatter(X_tsne[inds, 0],
                    X_tsne[inds, 1],
                    label=category_list[i],
                    color=colorst[i],
                    s=30)
    plt.legend(fontsize=24, scatterpoints=1, handletextpad=0.3, handlelength=1, loc="lower left")
    plt.axis('tight')
    plt.xticks(())  # ignore xticks
    plt.yticks(())  # ignore yticks
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.savefig(osp.join(save_path, "tsne.pdf"),
                    format='pdf',
                    dpi=600)
    plt.close()
