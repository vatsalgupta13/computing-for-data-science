import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings

data_df = pd.read_csv('./dataset.csv')

x_df = data_df[:][['Len1', 'Len2']]
y_df = data_df['Output']

train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=.3, random_state=0)

standardscaler = StandardScaler()

standardscaler.fit(train_x)

train_x_std = standardscaler.transform(train_x)
test_x_std = standardscaler.transform(test_x)

svm_model = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm_model.fit(train_x_std, train_y)

print('Accuray of model on training data is {:.2f}'.format(svm_model.score(train_x_std, train_y)))

print('Accuray of model on test data is {:.2f}'.format(svm_model.score(test_x_std, test_y)))

def tuple_v(v):
    return tuple(map(int, (v.split("."))))

def plot_func(x_df, y_df, model, idx_test=None, res=0.02):

    mark = ('s', 'x', 'o', '^', 'v')
    cols = ('#edf285', '#ec5858', '#fd8c04', '#93abd3', '#f9f7cf')
    map_c = ListedColormap(cols[:len(np.unique(y_df))])

    x_min_1, x_max_1 = x_df[:, 0].min() - 1, x_df[:, 0].max() + 1
    x_min_2, x_max_2 = x_df[:, 1].min() - 1, x_df[:, 1].max() + 1
    x1_np, x2_np = np.meshgrid(np.arange(x_min_1, x_max_1, res),
                           np.arange(x_min_2, x_max_2, res))
    z_np = model.predict(np.array([x1_np.ravel(), x2_np.ravel()]).T)
    z_np = z_np.reshape(x1_np.shape)
    plt.contourf(x1_np, x2_np, z_np, alpha=0.4, cmap=map_c)
    plt.xlim(x1_np.min(), x1_np.max())
    plt.ylim(x2_np.min(), x2_np.max())

    for idx, cl in enumerate(np.unique(y_df)):
        plt.scatter(x=x_df[y_df == cl, 0], y=x_df[y_df == cl, 1],
                    alpha=0.8, c=map_c(idx),
                    marker=mark[idx], label=cl)
    plt.show()

plot_func(test_x_std, test_y, svm_model)