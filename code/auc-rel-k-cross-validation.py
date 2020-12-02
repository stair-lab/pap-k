import warnings
warnings.filterwarnings('ignore')
import os,sys,inspect
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import json

import budhiraja_surrogate
import greedy_surrogate
import preck_surrogate
import ce_surrogate
import utils
from surrogates import *

# python3 auc-rel-k-cross-validation.py k dataset surrogate
k = int(sys.argv[1])
dataset = sys.argv[2]
if dataset not in ['behance', 'movielens', 'citation']:
    raise ValueError('invalid dataset name')
surrogate = sys.argv[3]
if surrogate not in ['rmax', 'ravg', 'tstruct', 'b_pauc', 'greedy', 'ce', 'preck']:
    raise ValueError('invalid surrogate name')

train_data, val_data, test_data = utils.load_data(dataset)
    
# rename column for compatibility
train_data = train_data.rename(columns={'user_id': 'user'})
val_data = val_data.rename(columns={'user_id': 'user'})
test_data = test_data.rename(columns={'user_id': 'user'})

user_list_train = list(set(train_data['user']))
user_list_val= list(set(val_data['user']))
user_list_test = list(set(test_data['user']))


for user in user_list_train:
    user_df = train_data[train_data['user'] == user]

for user in user_list_test:
    user_df = test_data[test_data['user'] == user]


user_n_train = {}
user_n_plus_train = {}
user_beta_train = {}

for user_id in user_list_train:
    user_df_train = train_data[train_data['user'] == user_id]
    user_n_train[user_id] = user_df_train.shape[0]
    user_n_plus_train[user_id] = sum(user_df_train.label)
    user_beta_train[user_id] = min(user_n_plus_train[user_id], k)

user_n_val = {}
user_n_plus_val = {}
user_beta_val = {}

for user_id in user_list_val:
    user_df_val = val_data[val_data['user'] == user_id]
    user_n_val[user_id] = user_df_val.shape[0]
    user_n_plus_val[user_id] = sum(user_df_val.label)
    user_beta_val[user_id] = min(user_n_plus_val[user_id], k)
    
user_n_test = {}
user_n_plus_test = {}
user_beta_test = {}

for user_id in user_list_test:
    user_df_test = test_data[test_data['user'] == user_id]
    user_n_test[user_id] = user_df_test.shape[0]
    user_n_plus_test[user_id] = sum(user_df_test.label)
    user_beta_test[user_id] = min(user_n_plus_test[user_id], k)

# filter out users not in both test and train, and users with fewer than k negatives
user_list_train_filtered = []
user_list_val_filtered = []
user_list_test_filtered = []

for user_id in user_list_train:
    beta = int(user_beta_train[user_id])
    n_minus = int(user_n_train[user_id]) - int(user_n_plus_train[user_id])
    if(beta > 0 and n_minus >= k):
        user_list_train_filtered.append(user_id)
for user_id in user_list_val:
    beta = int(user_beta_val[user_id])
    n_minus = int(user_n_val[user_id]) - int(user_n_plus_val[user_id])
    if(beta > 0 and n_minus >= k):
        user_list_val_filtered.append(user_id)
for user_id in user_list_test:
    beta = int(user_beta_test[user_id])
    n_minus = int(user_n_test[user_id]) - int(user_n_plus_test[user_id])
    if(beta > 0 and n_minus >= k):
        user_list_test_filtered.append(user_id)

print("train data is ", train_data.shape)
print("test data is ", test_data.shape)
print("users with at least one positive in train are", len(user_list_train_filtered))
print("users with at least one positive in val are", len(user_list_val_filtered))
print("users with at least one positive in test are", len(user_list_test_filtered))

def sort_order(user_feat, w):
    y_scores = user_feat.dot(w)
    indi_sort = np.argsort(-y_scores)
    return indi_sort, y_scores


def train(
    train_data,
    val_data, 
    user_list_train_filtered,
    user_list_val_filtered,
    surr,
    eta=0.1,
    momentum=0.0,
    lamb=0.1,
    num_iter_val=5,
    tolerance=1e-4,
    num_total_iter_training=61,
    draw=False,
    verbose=True,
    random_seed = 786,
    w=None):
    
    np.random.seed(random_seed)
    metrics = utils.Metrics()

    metrics.eta_lr = eta
    metrics.lamb_reg = lamb

    if w is None:
        w = np.random.normal(0, 1, (train_data.shape[1] - 2, ))
    
    prev_grad_w = np.zeros((train_data.shape[1] - 2, ))

    for num_iter in np.arange(num_total_iter_training):

        tic = time.time()

        metrics.w_list.append(w)

        loss_opt = 0
        k_minus_w_opt = 0
        grad_w = np.zeros((train_data.shape[1] - 2, ))
        
        sorting_time = 0
        surrogate_time = 0
        for user_id in user_list_train_filtered:

            user_df_train = train_data[train_data['user'] == user_id]

            if(len(user_df_train.label.unique()) == 1):
                if(user_df_train.label.iloc[0] == 1.0):
                    loss_opt += 0.0
                else:
                    loss_opt += 1.0
            else:

                sorting_start_time = time.time()
                beta = int(user_beta_train[user_id])

                user_df_pos = user_df_train[user_df_train['label'] == 1]
                user_df_neg = user_df_train[user_df_train['label'] == 0]

                user_feat_pos = user_df_pos.drop(['user', 'label'], axis = 1).values
                user_feat_neg = user_df_neg.drop(['user', 'label'], axis = 1).values

                indices_pos, scores_pos = sort_order(user_feat_pos, w)

                indices_neg, scores_neg = sort_order(user_feat_neg, w)

                sorted_user_feat_pos = user_feat_pos[indices_pos, :] 
                sorted_user_feat_neg = user_feat_neg[indices_neg, :] 
                sorted_scores_pos = scores_pos[indices_pos]
                sorted_scores_neg = scores_neg[indices_neg]
    
                sorting_time += time.time() - sorting_start_time

                surrogate_start_time = time.time()
                pi_opt, score_mat = surr.compute_pi(
                    sorted_scores_pos, sorted_scores_neg,
                    w, k, beta)
                
                loss_opt_user, _, _, _ = surr.loss(
                    pi_opt, sorted_scores_pos, sorted_scores_neg, k, beta)
                
                if draw and user_id == 0:
                    plt.subplot(1,2,1)
                    plt.imshow(score_mat)
                    plt.subplot(1,2,2)
                    plt.imshow(pi_opt)
                    plt.show()

                grad_w_user = surr.gradient(
                    sorted_user_feat_pos, sorted_user_feat_neg, pi_opt, k, beta)
                
                surrogate_time += time.time() - surrogate_start_time

                grad_w_user += lamb*w
                loss_opt += loss_opt_user

                grad_w += grad_w_user

        grad_w = grad_w/len(user_list_train_filtered)

        metrics_start_time = time.time()
                
        # sort data once for both micro 
        user_feat = train_data.drop(['user', 'label'], axis = 1).values
        y_scores = user_feat.dot(w)
        data_true = deepcopy(train_data)
        data_true['scores'] = y_scores
        data_true = data_true.sort_values(by='scores', ascending=False)
        data_true = data_true.reset_index(drop=True)

        metrics.grad_w_list.append(np.linalg.norm(grad_w))
        metrics.loss_opt_list_train.append(loss_opt/len(user_list_train_filtered))
        metrics.micro_auc_rel_k_list_train.append(utils.compute_micro(data_true, user_list_train_filtered, user_beta_train, w, k))
        
        if verbose:
            print('k=', k,'Epoch', num_iter+1, 'done out of',num_total_iter_training, 'for', surr.name, 'loss train:',metrics.loss_opt_list_train[-1])
            print('k=', k,'Epoch', num_iter+1, 'done out of',num_total_iter_training, 'for', surr.name, 'grad_w:',metrics.grad_w_list[-1])
            print('k=', k,'Epoch', num_iter+1, 'done out of',num_total_iter_training, 'for', surr.name, 'microaucrelk train:',metrics.micro_auc_rel_k_list_train[-1])
        else:
            print('epoch', num_iter+1, 'completed for', surr.name, 'micro:{}'.format(metrics.micro_auc_rel_k_list_train[-1]))
    
        if(num_iter%num_iter_val == 0):

            loss_opt_val = 0
            k_minus_w_opt_val = 0

            for user_id in user_list_val_filtered:

                user_df_val = val_data[val_data['user'] == user_id]

                if(len(user_df_val.label.unique()) == 1):
                    if(user_df_val.label.iloc[0] == 1.0):
                        loss_opt_val += 0.0
                    else:
                        loss_opt_val += 1.0
                else:
                    beta = int(user_beta_val[user_id])

                    user_df_pos = user_df_val[user_df_val['label'] == 1]
                    user_df_neg = user_df_val[user_df_val['label'] == 0]

                    user_feat_pos = user_df_pos.drop(['user', 'label'], axis = 1).values
                    user_feat_neg = user_df_neg.drop(['user', 'label'], axis = 1).values

                    indices_pos, scores_pos = sort_order(user_feat_pos, w)
                    indices_neg, scores_neg = sort_order(user_feat_neg, w)

                    sorted_user_feat_pos = user_feat_pos[indices_pos, :] 
                    sorted_user_feat_neg = user_feat_neg[indices_neg, :] 

                    sorted_scores_pos = scores_pos[indices_pos]
                    sorted_scores_neg = scores_neg[indices_neg]

                    pi_opt_val, score_mat_val = surr.compute_pi(sorted_scores_pos, sorted_scores_neg, w, k, beta)
                    loss_opt_user_val, _, _, _ = surr.loss(
                        pi_opt_val, sorted_scores_pos, sorted_scores_neg, k, beta)

                    if draw and user_id == 0:
                        plt.subplot(1,2,1)
                        plt.imshow(score_mat_val)
                        plt.subplot(1,2,2)
                        plt.imshow(pi_opt_val)
                        plt.show()

                    loss_opt_val += loss_opt_user_val

            # sort data once for both micro 
            user_feat = val_data.drop(['user', 'label'], axis = 1).values
            y_scores = user_feat.dot(w)
            data_true = deepcopy(val_data)
            data_true['scores'] = y_scores
            data_true = data_true.sort_values(by='scores', ascending=False)
            data_true = data_true.reset_index(drop=True)
        
            metrics.loss_opt_list_val.append(loss_opt_val/len(user_list_val_filtered))
            metrics.micro_auc_rel_k_list_val.append(utils.compute_micro(data_true, user_list_val_filtered, user_beta_val, w, k))

            if verbose:
                print('k=', k,'Epoch', num_iter+1, 'completed out of',num_total_iter_training, 'for', surr.name, 'loss val:',metrics.loss_opt_list_val[-1])
                print('k=', k,'Epoch', num_iter+1, 'completed out of',num_total_iter_training, 'for', surr.name, 'microaucrelk val:',metrics.micro_auc_rel_k_list_val[-1])
            else:
                print('    val micro:{}'.format(metrics.micro_auc_rel_k_list_val[-1]))

        prev_grad_w = momentum * prev_grad_w + (1-momentum) * grad_w
        
        w = w - (eta/np.sqrt(num_iter+1))*(prev_grad_w)

        if verbose:
            print('Epoch', num_iter+1, ' time taken is: ', time.time() - tic)
            print("\n")

        # also break if reached tolerance condition
        if num_iter >= 10 and max(metrics.loss_opt_list_train[-10:])-min(metrics.loss_opt_list_train[-10:]) <= tolerance:
            break
            
        if num_iter >= 50 and (np.diff(metrics.micro_auc_rel_k_list_val[-4:], n=1) < 0).all():
            break

    best_iter = (np.where(np.asarray(metrics.loss_opt_list_train)==np.min(metrics.loss_opt_list_train))[0][0]//num_iter_val)*num_iter_val
    best_microaucrelk = metrics.micro_auc_rel_k_list_val[best_iter//num_iter_val]
    print('Best micro aucrelk at iter: %d (metric: %f)' % (best_iter, best_microaucrelk))
    
    return metrics, w


all_metrics = []
for eta_try in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    for lamb_try in [0.001, 0.01, 0.1, 1.0]:
        # call appropriate training function
        if surrogate == 'greedy':
            orig_stdout = sys.stdout
            f = open('../results/' + dataset + '/' + surrogate + '/Check_greedy-{}-{}-{}.txt'.format(k, eta_try, lamb_try), 'w')
            sys.stdout = f
            metric = greedy_surrogate.train(train_data, val_data,
                user_list_train_filtered, user_list_val_filtered, user_beta_train,
                user_beta_val, k, num_total_iter_training = 61, verbose=True, eta=eta_try, lamb=lamb_try)[0]
            sys.stdout = orig_stdout
            f.close()
        elif surrogate == 'preck':
            orig_stdout = sys.stdout
            f = open('../results/' + dataset + '/' + surrogate + '/Check_preck-{}-{}-{}.txt'.format(k, eta_try, lamb_try), 'w')
            sys.stdout = f
            metric = preck_surrogate.train(train_data, val_data,
                user_list_train_filtered, user_list_val_filtered, user_beta_train,
                user_beta_val, k, dataset, num_total_iter_training = 61, verbose=True, eta=eta_try, lamb=lamb_try)[0]
            sys.stdout = orig_stdout
            f.close()
        else:
            surrogate_map = {
                'rmax': RMax,
                'ravg': RAvg,
                'tstruct': TStruct,
                'b_pauc': Baseline_Struct_Pauc,
            }
            metric = train(train_data, val_data, user_list_train_filtered,
                user_list_val_filtered, surrogate_map[surrogate](), num_total_iter_training = 61, verbose=True, eta=eta_try, lamb=lamb_try)[0]

        best_iter = np.where(np.asarray(metric.loss_opt_list_train)==np.min(metric.loss_opt_list_train))[0][0]
        all_metrics.append(
            ('{}-{}-{}'.format(surrogate, eta_try, lamb_try), metric)
        )
        # save output to file
        with open('../results/' + dataset + '/' + surrogate + '/result-{}-{}-{}-{}-{}.json'.format(dataset, k, surrogate, eta_try, lamb_try), 'w') as fp:
            json.dump(metric.to_dict(best_iter), fp)

print('../../results/' + dataset + '/' + surrogate + '/FinalTestResult-{}-{}-{}.txt'.format(surrogate, dataset, k))
orig_stdout = sys.stdout
f = open('../../results/' + dataset + '/' + surrogate + '/FinalTestResult-{}-{}-{}.txt'.format(surrogate, dataset, k), 'w')
sys.stdout = f

# compute val auc-rel-k at best epoch
for name, metric in all_metrics:
    best_iter = np.where(np.asarray(metric.loss_opt_list_train)==np.min(metric.loss_opt_list_train))[0][0]
    w = metric.w_list[best_iter]
                 
    loss_opt_val = 0
    k_minus_w_opt_val = 0

    # sort data once for both micro 
    user_feat = val_data.drop(['user', 'label'], axis = 1).values
    y_scores = user_feat.dot(w)
    data_true = deepcopy(val_data)
    data_true['scores'] = y_scores
    data_true = data_true.sort_values(by='scores', ascending=False)
    data_true = data_true.reset_index(drop=True)

    print('k:', k)
    print('Dataset:', dataset)
    print('Name:', name)
    print('    ', loss_opt_val/len(user_list_val_filtered))
    print('    ', utils.compute_micro(data_true, user_list_val_filtered, user_beta_val, w, k))

sys.stdout = orig_stdout
f.close()
