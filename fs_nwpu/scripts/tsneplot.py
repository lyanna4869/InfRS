from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import pandas as pd
colors = ['black','tomato','yellow','cyan','blue', 'lime', 'r', 'violet','m','peru','olivedrab','hotpink']#设置散点颜色

# tsne = TSNE(n_components=10, init='pca', random_state=0)
class_num = 10
def get_data():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    label = label.reshape((-1, 1))
    S_data=np.hstack((data, label))
    S_data=pd.DataFrame({'x': S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    fig = plt.figure()
    ax = plt.subplot(111)
    mid = int(data.shape[0] / 2)
    for index in range(class_num):
        X= S_data.loc[S_data['label'] == index]['x']
        Y= S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X,Y,cmap='brg', s=1,  c=colors[index], edgecolors=colors[index],alpha=0.65)

    plt.xticks([])  # 坐标轴设置
    plt.yticks([])
    plt.title(title)
    plt.savefig("checkpoints/results/tsne.png")

def main():
    data, label, n_samples, n_features = get_data() # data T[ 1797, 64] label T[1797, ]
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2,  random_state=0, n_iter=3000, init='pca')
    t0 = time()
    result = tsne.fit_transform(data) # T[1797, 10]

    plot_embedding(result, label,
                   't-SNE embedding of the digits (time %.2fs)'
                   % (time() - t0))


if __name__ == '__main__':
    main()