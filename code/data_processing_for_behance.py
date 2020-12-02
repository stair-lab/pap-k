import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from random import sample 
from sklearn.utils import shuffle
from sklearn import manifold, datasets
import umap
from time import time 
from copy import deepcopy
import struct
import random
import pickle
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)

np.random.seed(5)

def readImageFeatures(path):
    f = open(path, 'rb')
    item_feat_dict = {}
    while True:
        itemId = f.read(8)
        if itemId == '': break
        try:
            feature = struct.unpack('f'*4096, f.read(4*4096))
            yield itemId, feature
        except: 
            continue

length = 178787
item_ids_to_row = {}

item_feat_dict = readImageFeatures('../data/behance/Behance_Image_Features.b')

k = 0
item_feat = np.zeros((length, 4096))
for i, el in enumerate(item_feat_dict):
    if(k%10000 == 0):
        print(k, " is done.")
    item_feat[i, :] = el[1]
    item_ids_to_row[el[0].decode("utf-8")] = i
    k+=1
    if(k == 178787):
        break
    
item_list = list(item_ids_to_row.keys())

# This process takes approximately 45 minutes

tic1 = time()
n_neighbors = 10
dim = 50

umap_emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.5, n_components=dim, metric='euclidean')
item_feat_red = umap_emb.fit_transform(item_feat)

tic2 = time()
print("time taken is ", tic2 - tic1)


np.save('../data/behance/item_feat_d' + str(dim), item_feat_red)

with open('../data/behance/item_ids_to_row_d' + str(dim) + '.pickle', 'wb') as handle:
    pickle.dump(item_ids_to_row, handle)
    
item_ids_to_feat_dict = {}

for k in item_ids_to_row.keys():
    item_ids_to_feat_dict[k] = item_feat_red[item_ids_to_row[k], :]
    
with open('../data/behance/item_ids_to_feat_dict_d' + str(dim) + '.pickle', 'wb') as handle:
    pickle.dump(item_ids_to_feat_dict, handle)


############ Beyond this point, only processing of data happens ##########

dim = 50

with open('../data/behance/item_ids_to_feat_dict_d' + str(dim) + '.pickle', 'rb') as handle:
    item_ids_to_feat_dict = pickle.load(handle)
    
item_list = list(item_ids_to_feat_dict.keys())

dat = pd.read_csv('../data/behance/Behance_appreciate_1M', names = ['user', 'item', 'timestamp'], sep = "\s+", dtype=str)
dat = dat.sort_values(by=['timestamp'])
dat = dat.drop(['timestamp'], axis = 1)
dat['label'] = 1.0

indexNames = dat[dat['item'] == '01398047'].index
dat.drop(indexNames , inplace=True)

users_dist = pd.DataFrame(dat.user.value_counts())
users_dist = users_dist.rename(columns = {'user' : 'datapoints'})
users_dist['user'] = users_dist.index
users_dist = users_dist.reset_index(drop=True)
users_dist = users_dist[['user', 'datapoints']]


# Using 50 images to define user features. Choose your thresholds accordingly

thresh_below = 60   # 50 will be used in constructing user features
thresh_above = 170  # Just to limit the number of users, otherwise data becomes too big

num_users = len(users_dist[(users_dist['datapoints'] <= thresh_above) & (users_dist['datapoints'] >= thresh_below)])

print("Number of users in the desired frequency thresholds ", num_users)

users_dist_thresholded = users_dist[(users_dist['datapoints'] <= thresh_above) & (users_dist['datapoints'] >= thresh_below)]

user_list = list(users_dist_thresholded.user)
dat_filtered = dat[dat['user'].isin(user_list)]
dat_filtered['user'] = dat_filtered['user'].astype(str)
dat_filtered['item'] = dat_filtered['item'].astype(str)

num_negatives_factor_list = [4,5,6]
user_dataframe_list = []

user_item_forfeat = {}

for user in user_list:
    
    dat_filtered_user = dat_filtered[dat_filtered['user'] == user]

    item_forfeat = list(dat_filtered_user[0:(thresh_below - 10)].item)
    user_item_forfeat[user] = item_forfeat

    dat_filtered_user = dat_filtered_user[(thresh_below - 10):]

    user_item_list = list(dat_filtered_user.item)
    user_item_remaining_list = list(set(item_list) - set(user_item_list))

    num_negative_factor = sample(num_negatives_factor_list, 1)[0]
    user_item_selected = sample(user_item_remaining_list, num_negative_factor*len(dat_filtered_user))
    user_selected = [user]*len(user_item_selected)
    label_selected = [0.0]*len(user_item_selected)
    
    dat_w_zeros = pd.DataFrame(list(zip(user_selected, user_item_selected, label_selected)), columns =['user', 'item', 'label'])
    dat_final_user = pd.concat([dat_w_zeros, dat_filtered_user]) 
    
    user_dataframe_list.append(dat_final_user)


final_data_w_ids = pd.concat(user_dataframe_list)
final_data_w_ids = shuffle(final_data_w_ids)
final_data_w_ids = final_data_w_ids.reset_index(drop=True)

user_feat_dict = {}
for user in user_list:
    user_feat = np.zeros(dim)   
    for i in user_item_forfeat[user]:
        user_feat += item_ids_to_feat_dict[i]
    user_feat = user_feat/float(len(user_item_forfeat[user]))
    user_feat_dict[user] = user_feat

final_data = deepcopy(final_data_w_ids)
final_data['user'] = final_data['user'].apply(lambda x: user_feat_dict[x])
final_data['item'] = final_data['item'].apply(lambda x: item_ids_to_feat_dict[x])
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
    
user_list[0]

train_df_list = []
val_df_list = []
test_df_list = []


for user in user_list:
    
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
val_data = pd.concat(val_df_list)
test_data = pd.concat(test_df_list)

for i in range(3*dim):
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

train_data.to_csv('../data/behance/train_data_d' + str(3*dim) + '.tsv', sep='\t', index=False)
val_data.to_csv('../data/behance/val_data_d' + str(3*dim) + '.tsv', sep='\t', index=False)
test_data.to_csv('../data/behance/test_data_d' + str(3*dim) + '.tsv', sep='\t', index=False)