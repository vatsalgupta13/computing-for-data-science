import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import datetime
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

output = train_df['output']
del train_df['output']

all_df = pd.concat([train_df,test_df],axis=0)
all_df['Start'] = pd.to_datetime(all_df["Start"])
all_df['Year'] = all_df['Start'].apply(lambda x:x.year)
all_df['Month'] = all_df['Start'].apply(lambda x:x.month)
all_df['Day'] = all_df['Start'].apply(lambda x:x.day)

labelencoder = LabelEncoder()
all_df['Place'] = labelencoder.fit_transform(all_df['Place'])
all_df['Group'] = all_df['Group'].map({'Other':0,'Big Cities':1})
all_df["Category"] = all_df["Category"].map({"FC":0, "IL":1, "DT":2, "MB":3})

train_df = all_df.iloc[:train_df.shape[0]]
test_df = all_df.iloc[train_df.shape[0]:]

train_cols_df = [col for col in train_df.columns if col not in ['Id','Start']]

standardscaler = StandardScaler()
minmaxscaler = MinMaxScaler()

train_sc_df = standardscaler.fit_transform(train_df[train_cols_df])
train_scms_df = minmaxscaler.fit_transform(train_sc_df)

def gen_cv():
    m_train = np.floor(len(output)*0.75).astype(int)
    train_idx = np.arange(m_train)
    test_idx = np.arange(m_train, len(output))
    yield (train_idx, test_idx)

cnt_param = 20
param = {"C":np.logspace(0,1,cnt_param), "epsilon":np.logspace(-1,1,cnt_param)}
gridsearch = GridSearchCV(SVR(kernel="linear"), param, cv=gen_cv(), scoring="r2", return_train_score=True)
gridsearch.fit(train_scms_df, output)
print('The best parameter = ',gridsearch.best_params_)
print('accuracy = ',gridsearch.best_score_)
print()

regr = SVR(kernel="linear", C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])

cnt_split = 5
kfold = KFold(n_splits=cnt_split,shuffle=True,random_state=0)

svr_models = []
for train_index,test_index in kfold.split(train_scms_df):
    train_x = train_df.iloc[train_index][train_cols_df]
    train_y = output.iloc[train_index]
    test_x = train_df.iloc[test_index][train_cols_df]
    test_y = output.iloc[test_index]
    svr_model = regr.fit(train_x, train_y)
    svr_models.append(svr_model)
    pred = regr.predict(test_x)
    plt.scatter(train_x['P11'], train_y)
    plt.plot(test_x['P11'], pred, color='red')
    plt.xlabel('P11')
    plt.ylabel('Output')
    plt.show()
