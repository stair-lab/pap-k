import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from random import sample 
from sklearn.utils import shuffle
from sklearn import manifold, datasets
from sklearn.datasets import load_digits
import umap
from time import time 
from copy import deepcopy
import random
import pickle

pd.set_option('display.max_rows', 500)
np.random.seed(5)

dim = 50

with open('../data/citation/citation/CitationDataset_Train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('../data/citation/citation/CitationDataset_Val.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open('../data/citation/citation/CitationDataset_Test.pkl', 'rb') as f:
    test_data = pickle.load(f)

dim = 50

new_name = {}

new_name['CoAuthorCount'] = 'f0'
new_name['UserAuthorCitationCount'] = 'f1'
new_name['AuthorUserCitationCount'] = 'f2'
new_name['UserCitedConference'] = 'f3'
new_name['UserPublishedInConferenceCount'] = 'f4'


for i in range(5, 2*dim+5):
    new_name[str(i-5)] = 'f' + str(i)

train_data['u_id'] = train_data['u_id'].astype(str) + '_' + train_data['paper_id_written'].astype(str)
train_data.drop(['paper_id_written', 'a_id', 'paper_id_referred', 'c_id'], axis = 1, inplace=True)
train_data = train_data.rename(columns={"u_id": "user"})
train_data = train_data.rename(columns=new_name)

val_data['u_id'] = val_data['u_id'].astype(str) + '_' + val_data['paper_id_written'].astype(str)
val_data.drop(['paper_id_written', 'a_id', 'paper_id_referred', 'c_id'], axis = 1, inplace=True)
val_data = val_data.rename(columns={"u_id": "user"})
val_data = val_data.rename(columns=new_name)

test_data['u_id'] = test_data['u_id'].astype(str) + '_' + test_data['paper_id_written'].astype(str)
test_data.drop(['paper_id_written', 'a_id', 'paper_id_referred', 'c_id'], axis = 1, inplace=True)
test_data = test_data.rename(columns={"u_id": "user"})
test_data = test_data.rename(columns=new_name)

train_users = train_data.user.tolist()
val_users = val_data.user.tolist()
test_users = test_data.user.tolist()

print("Number of users in train data are: ", len(train_data.user.unique()))
print("Number of users in val data are: ", len(val_data.user.unique()))
print("Number of users in test data are: ", len(test_data.user.unique()))
print("\n")

print("Number of datapoints in train data are: ", len(train_data))
print("Number of datapoints in val data are: ", len(val_data))
print("Number of datapoints in test data are: ", len(test_data))
print("\n")

print("Number of positive datapoints in train data are: ", sum(train_data.label))
print("Number of positive datapoints in val data are: ", sum(val_data.label))
print("Number of positive datapoints in test data are: ", sum(test_data.label))

def filterdata(data, pos_to_all_ratio_thresh = 0.1, num_pos_thresh = 3):
    
    dat = deepcopy(data)
    
    user_list = dat.user.unique().tolist()

    user_list_sel = []
    
    # The following is to remove users who have extraordinarily large number of negatives, yet it maintains
    # significant data imbalance.
    # Also, users are required to have at least three positives.
    
    for user in user_list:

        dat_user = dat[dat['user'] == user]
        if((sum(dat_user.label)/len(dat_user) >= pos_to_all_ratio_thresh) and sum(dat_user.label) >= num_pos_thresh):
            user_list_sel.append(user)

    print("Number of selected users are ", len(user_list_sel))

    dat = dat.loc[dat['user'].isin(user_list_sel)]
    print("Shape of filtered data is ", dat.shape)

    dat = dat.rename(columns={"user": "user_id"})

    new_feat = dat.iloc[:, 6:(dim+6)].values*dat.iloc[:, (dim+6):(2*dim+6)].values

    for i in range(dim):
        dat['f'+str(2*dim+5+i)] = new_feat[:, i]

    new_colnames = ['user_id', 'f0', 'f1', 'f2', 'f3', 'f4']

    for i in range(5, 3*dim + 5):
        new_colnames.append('f'+str(i))

    new_colnames.append('label')

    dat = dat[new_colnames]

    return dat

train_data_final = filterdata(train_data)
val_data_final = filterdata(val_data)
test_data_final = filterdata(test_data)

print("Number of users in train data are: ", len(train_data_final.user_id.unique()))
print("Number of users in val data are: ", len(val_data_final.user_id.unique()))
print("Number of users in test data are: ", len(test_data_final.user_id.unique()))
print("\n")

print("Number of datapoints in train data are: ", len(train_data_final))
print("Number of datapoints in val data are: ", len(val_data_final))
print("Number of datapoints in test data are: ", len(test_data_final))
print("\n")

print("Number of positive datapoints in train data are: ", sum(train_data_final.label))
print("Number of positive datapoints in val data are: ", sum(val_data_final.label))
print("Number of positive datapoints in test data are: ", sum(test_data_final.label))

train_data = train_data_final
val_data = val_data_final
test_data = test_data_final

for i in range(3*dim+5):
    f = 'f{}'.format(i)
    feature_range = train_data[f].max() - train_data[f].min()
    feature_mean = (train_data[f].max() + train_data[f].min()) / 2
    train_data[f] = (train_data[f] - feature_mean) / feature_range
    val_data[f] = (val_data[f] - feature_mean) / feature_range
    test_data[f] = (test_data[f] - feature_mean) / feature_range
    
train_data = shuffle(train_data)
train_data = train_data.reset_index(drop=True)
val_data = shuffle(val_data)
val_data = val_data.reset_index(drop=True)
test_data = shuffle(test_data)
test_data = test_data.reset_index(drop=True)

train_data.to_csv('../data/citation/train_data_d' + str(3*dim+5) + '.tsv', sep='\t', index=False)
val_data.to_csv('../data/citation/val_data_d' + str(3*dim+5) + '.tsv', sep='\t', index=False)
test_data.to_csv('../data/citation/test_data_d' + str(3*dim+5) + '.tsv', sep='\t', index=False)