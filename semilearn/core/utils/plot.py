import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os.path as osp


def plot_cm(cm, labels=None, save_path=''):
    plt.figure(figsize=(10, 8))
    if labels:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
   
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像以释放资源

def plot_tsne(feats, preds, n_classes, save_path='', num_samples=1000):
    feats = feats.cpu().detach().numpy()
    # preds = preds.cpu().detach().numpy()

    if num_samples:
        indices = np.random.choice(len(preds), size=num_samples, replace=False)
        feats = feats[indices]
        preds = preds[indices]

    num_class = n_classes + 1
    print(f'class number: {num_class}')

    X_tsne = TSNE(n_components=2, init='pca',
                  learning_rate='auto').fit_transform(feats)

    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    colormap = plt.cm.gist_ncar  #nipy_spectral, Set1,Paired
    colorst = [colormap(i) for i in np.linspace(0, 0.9, num_class)]
    for i in range(num_class):
        inds = np.where(preds == i)[0]
        plt.scatter(X_tsne[inds, 0],
                    X_tsne[inds, 1],
                    label=str(i),
                    color=colorst[i],
                    s=20)
    plt.legend(fontsize=14)
    plt.axis('tight')
    plt.xticks(())  # ignore xticks
    plt.yticks(())  # ignore yticks
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.savefig(osp.join(save_path, "tsne.pdf"),
                    format='pdf',
                    dpi=600)
    plt.close()