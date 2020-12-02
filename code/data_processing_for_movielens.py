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
from tqdm import tqdm
import numpy as np
from scipy import linalg
from numpy import dot

pd.set_option('display.max_rows', 500)
np.random.seed(5)

dim = 30

dat = pd.read_csv('../data/movielens/u.data', names = ['user', 'item', 'label', 'timestamp'], sep = "\s+", dtype=str)
dat['label'] = dat['label'].astype('int')
dat = dat.sort_values(by=['timestamp'])
dat = dat.drop(['timestamp'], axis = 1)
user_list = dat.user.unique().tolist()

thresh_for_MF = 20
rows_for_MF = []

for user in user_list:
    
    dat_user = dat[dat['user'] == str(user)]
    rows_for_MF.extend(list(dat_user.index)[0:thresh_for_MF])

rows_rest = list(set(list(range(dat.shape[0]))) - set(rows_for_MF))

dat_for_MF = dat.loc[rows_for_MF, :]
item_list = dat_for_MF.item.unique()
item_list_dict = {}
for i in range(len(item_list)):
    item_list_dict[item_list[i]] = i
user_list = dat_for_MF.user.unique()
user_list_dict = {}
for i in range(len(user_list)):
    user_list_dict[user_list[i]] = i

matrix_for_MF = np.zeros((len(user_list), len(item_list)))
for i in range(dat_for_MF.shape[0]):
    matrix_for_MF[user_list_dict[dat_for_MF.user.iloc[i]], item_list_dict[dat_for_MF.item.iloc[i]]] = dat_for_MF.label.iloc[i]

def nmf(X, latent_features = dim, max_iter=1000, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    curReslist = []
    eps = 1e-5

    mask = np.sign(X)

    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)


        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            curReslist.append(curRes)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y, curReslist

U, V, loss = nmf(matrix_for_MF)
print("U shape is")
print(U.shape)
V = V.transpose()
print("V shape is")
print(V.shape)

user_feat = {}
item_feat = {}

for user in user_list_dict.keys():
    user_feat[user] = U[user_list_dict[user], :]
    
for item in item_list_dict.keys():
    item_feat[item] = V[item_list_dict[item], :]

dat_rest = dat.loc[rows_rest, :]
print("Number of rows in rest of the data are ", dat_rest.shape[0])
dat_rest = dat_rest[dat_rest['item'].isin(item_list)]
print("Number of rows in rest of the data after filtering are ", dat_rest.shape[0])
dat_rest['label'] = (dat_rest['label'] >= 5).astype(int)
print("Fraction of postives are: ")
print(sum(dat_rest.label)/len(dat_rest.label))

users_dist = pd.DataFrame(dat_rest.user.value_counts())
users_dist = users_dist.rename(columns = {'user' : 'datapoints'})
users_dist['user'] = users_dist.index
users_dist = users_dist.reset_index(drop=True)
users_dist = users_dist[['user', 'datapoints']]

thresh_below = 20

num_users = len(users_dist[(users_dist['datapoints'] >= thresh_below)])

print("Number of users in the desired frequency thresholds ", num_users)

users_dist_thresholded = users_dist[(users_dist['datapoints'] >= thresh_below)]

user_filter_list = list(users_dist_thresholded.user)
dat_filtered = dat_rest[dat_rest['user'].isin(user_filter_list)]
dat_filtered = dat_filtered.reset_index(drop=True)

pos_neg_ratio = []

for user in user_filter_list:
    dat_filtered_user = dat_filtered[dat_filtered['user'] == str(user)]
    pos_neg_ratio.append(sum(dat_filtered_user.label)/len(dat_filtered_user))

final_data_w_ids = dat_filtered
final_data_w_ids = shuffle(final_data_w_ids)
final_data_w_ids = final_data_w_ids.reset_index(drop=True)
final_data = deepcopy(final_data_w_ids)
final_data['user'] = final_data['user'].apply(lambda x: user_feat[x])
final_data['item'] = final_data['item'].apply(lambda x: item_feat[x])
final_data['user_id'] = final_data_w_ids['user']
final_data = final_data[['user_id', 'user', 'item', 'label']]

col_names = ['user_id']
for i in range(3*dim):
    col_names.append('f'+str(i))
col_names.append('label')

other_data = pd.DataFrame(columns=col_names)
other_data['user_id'] = final_data['user_id']
other_data['label'] = final_data['label']

other_data[col_names[1:(dim+1)]] = final_data.user.values.tolist()
other_data[col_names[(dim+1):(2*dim+1)]] = final_data.item.values.tolist()

new_feat = other_data.iloc[:, 1:(dim+1)].values*other_data.iloc[:, (dim+1):(2*dim+1)].values

for i in range(dim):
    other_data['f'+str(2*dim+i)] = new_feat[:, i]
    
train_df_list = []
val_df_list = []
test_df_list = []


for user in tqdm(user_filter_list):
    
    other_data_user = other_data[other_data['user_id'] == user]
    
    msk = np.random.rand(len(other_data_user)) <= 0.6
    train = other_data_user[msk]
    remaining = other_data_user[~msk]
    
    msk = np.random.rand(len(remaining)) <= 0.5
    val = remaining[msk]
    test = remaining[~msk]
    
    train_df_list.append(train)
    val_df_list.append(val)
    test_df_list.append(test)


train_data = pd.concat(train_df_list)
train_data = shuffle(train_data)
train_data = train_data.reset_index(drop=True)

val_data = pd.concat(val_df_list)
val_data = shuffle(val_data)
val_data = val_data.reset_index(drop=True)

test_data = pd.concat(test_df_list)
test_data = shuffle(test_data)
test_data = test_data.reset_index(drop=True)

for i in range(3*dim):
    f = 'f{}'.format(i)
    feature_range = train_data[f].max() - train_data[f].min()
    feature_mean = (train_data[f].max() + train_data[f].min()) / 2
    train_data[f] = (train_data[f] - feature_mean) / feature_range
    val_data[f] = (val_data[f] - feature_mean) / feature_range
    test_data[f] = (test_data[f] - feature_mean) / feature_range

train_data.to_csv('../data/movielens/train_data_d' + str(3*dim) + '.tsv', sep='\t', index=False)
val_data.to_csv('../data/movielens/val_data_d' + str(3*dim) + '.tsv', sep='\t', index=False)
test_data.to_csv('../data/movielens/test_data_d' + str(3*dim) + '.tsv', sep='\t', index=False)