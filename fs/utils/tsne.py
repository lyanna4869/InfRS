### bsf.c 2024.05.21 stable version 1.0

import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import pandas as pd
import os.path as osp
import pdb
import matplotlib.colors as mcolors
# COLORS = [
#     'black', 'firebrick', 'red', 'orangered', 'saddlebrown',
#     'darkorange', 'gold', 'yellow', 'greenyellow', 'forestgreen',
#     'lightseagreen', 'lime', 'cyan', 'deepskyblue', 'royalblue',
#     'darkblue', 'blueviolet','purple','fuchsia','deeppink',
#     'hotpink'
#     ]
COLORS = [
    '#87843b', 'firebrick', 'RoyalBlue','Turquoise','darkgreen','#f47920',
    'Goldenrod', 'lawngreen','darkviolet','palevioletred'
]
# COLORS = [
#     '黑色', '耐火磚紅', '皇家藍','天藍色','深綠色','棕色',
#     '金色', '草地綠色','紫羅蘭色','淺紫紅色'
# ]
# 1: ['ship', 'storage-tank', 'ground-track-field', 'basketball-court', 'harbor', 'bridge', 'vehicle',
#     'airplane', 'baseball-diamond', 'tennis-court',
#设置散点颜色

def get_data():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

class TSNEPloter():
    def __init__(self, num_class=15, n_iter=1000) -> None:
        self.num_class = num_class
        # 显示成 2 维图像
        self.tsne = TSNE(n_components=2,  random_state=0, n_iter=n_iter, init='pca')
# tsne = TSNE(n_components=10, init='pca', random_state=0)
    def start(self, detection, gt, title, **kwargs):
        print('Computing t-SNE embedding')
        t0 = time()
        result = self.tsne.fit_transform(detection) # T[1797, 10]
        t1 = time() - t0
        self._plot_embedding(result, gt, f'{title}', **kwargs)
        print(f"Use: {t1:.2f}s")


    def _plot_embedding(self, data, label, title, dst="checkpoints/results/tsne"):
        """label: T[]
        """
        dname = osp.dirname(dst)
        if len(dname) > 0:
            os.makedirs(dname, exist_ok=True)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / x_max - x_min
        label = label.reshape((-1, 1))
      #  pdb.set_trace()
        S_data=np.hstack((data, label))
        S_data=pd.DataFrame({'x': S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
        fig = plt.figure()
        ax = plt.subplot(111)
        mid = int(data.shape[0] / 2)
        for index in range(self.num_class):
            X = S_data.loc[S_data['label'] == index]['x']
            Y = S_data.loc[S_data['label'] == index]['y']
            plt.scatter(X, Y, cmap='brg', s=1, marker='+', c=COLORS[index], edgecolors=COLORS[index],alpha=1)

        plt.xticks([])  # 坐标轴设置
        plt.yticks([])
        plt.title(title)
        plt.savefig(f"{dst}.png")
        plt.savefig(f"{dst}.svg")

def main():
    data, label, n_samples, n_features = get_data() # data T[ 1797, 64] label T[1797, ]
    num_class = 10
    tsneploter = TSNEPloter(num_class)
    tsneploter._plot_embedding(data, label, "t-SNE embedding of the digits")


if __name__ == '__main__':
    main()