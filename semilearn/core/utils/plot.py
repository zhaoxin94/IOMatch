import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
from matplotlib.ticker import MaxNLocator  # 导入 MaxNLocator



# def plot_cm(cm, labels=None, save_path='', label_fontsize=16, annot_fontsize=20):
#     # 设置全局字体为 Times New Roman
#     plt.rcParams['font.family'] = 'Times New Roman'
#     # plt.rcParams['font.family'] = 'serif'

#     plt.figure(figsize=(5, 4))
#     if labels:
#         heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": annot_fontsize})
#     else:
#         heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": annot_fontsize})
#     # plt.title('Confusion Matrix', fontsize=title_fontsize)

#     # 调整图例（colorbar）的字体大小
#     cbar = heatmap.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=label_fontsize)

#     cbar.locator = MaxNLocator(nbins=6)  # 设置图例的刻度数量
#     cbar.update_ticks()

#     plt.xlabel('Predicted Label', fontsize=label_fontsize-2)
#     plt.ylabel('True Label', fontsize=label_fontsize-2)
#     plt.xticks(rotation=0)
#     plt.yticks(rotation=0)
#     plt.xticks(fontsize=label_fontsize)
#     plt.yticks(fontsize=label_fontsize)
#     plt.tight_layout()
   
#     # 保存图像
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()  # 关闭图像以释放资源


def plot_cm(cm, labels=None, save_path='', label_fontsize=15, annot_fontsize=18):
    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.family'] = 'serif'

    plt.figure(figsize=(5, 4))
    if labels:
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": annot_fontsize})
    else:
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": annot_fontsize})
    # plt.title('Confusion Matrix', fontsize=title_fontsize)

    # 调整图例（colorbar）的字体大小
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=label_fontsize)

    cbar.locator = MaxNLocator(nbins=6)  # 设置图例的刻度数量
    cbar.update_ticks()

    plt.xlabel('Predicted Label', fontsize=label_fontsize-2)
    plt.ylabel('True Label', fontsize=label_fontsize-2)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=label_fontsize)
    plt.tight_layout()
   
    # 保存图像
    plt.savefig(save_path+'.png', dpi=300, format='png', bbox_inches='tight')
    plt.savefig(save_path+'.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.close()  # 关闭图像以释放资源