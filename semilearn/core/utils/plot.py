import numpy as np
import matplotlib.pyplot as plt
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