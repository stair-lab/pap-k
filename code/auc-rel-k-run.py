import warnings
warnings.filterwarnings('ignore')
import os,sys,inspect
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
import json

import greedy_surrogate
import preck_surrogate
import utils
from surrogates import *

tic_all = time.time()

n_runs = 1
np.random.seed(5)


k = int(sys.argv[1])
dataset = sys.argv[2]
if dataset not in ['behance', 'movielens', 'citation']:
    raise ValueError('invalid dataset name')

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
    num_total_iter_training=21,
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
        
        pi_opt_avg_user_b = 0
        pi_opt_avg_user_n = 0
        
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
                
                pi_opt_avg_user_b += np.sum(pi_opt)/(beta*k)
                pi_opt_avg_user_n += np.sum(pi_opt)/(len(indices_pos)*k)
                
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
        
        pi_opt_avg_user_b = pi_opt_avg_user_b/len(user_list_train_filtered)
        pi_opt_avg_user_n = pi_opt_avg_user_n/len(user_list_train_filtered)

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
            print('k=', k,'Epoch', num_iter+1, 'done out of',num_total_iter_training, 'for', surr.name, 'pi_opt_avg_user_b:',pi_opt_avg_user_b)
            print('k=', k,'Epoch', num_iter+1, 'done out of',num_total_iter_training, 'for', surr.name, 'pi_opt_avg_user_n:',pi_opt_avg_user_n)
        else:
            print('k=', k,'Epoch', num_iter+1, 'done. micro:{}'.format(metrics.micro_auc_rel_k_list_train[-1]))
    
        prev_grad_w = momentum * prev_grad_w + (1-momentum) * grad_w
        
        w = w - (eta/np.sqrt(num_iter+1))*(prev_grad_w)

        if verbose:
            print('Epoch', num_iter+1, ' time taken is: ', time.time() - tic)
            print("\n")

        # also break if reached tolerance condition
        if num_iter >= 10 and max(metrics.loss_opt_list_train[-10:])-min(metrics.loss_opt_list_train[-10:]) <= tolerance:
            break

    # save output to file
    best_iter = np.where(np.asarray(metrics.loss_opt_list_train)==np.min(metrics.loss_opt_list_train))[0][0]
    with open('../results/' + dataset + '/test_results/result-testcomp-{}-{}-{}-{}-{}.json'.format(eta, lamb, k, surr.name, random_seed), 'w') as fp:
        json.dump(metrics.to_dict(best_iter), fp)
    
    return metrics, w


seeds = np.random.randint(0, 1e7, size=n_runs)
all_metrics = []
for seed in seeds:
    run_metrics = [
        ('max', train(train_data, val_data, user_list_train_filtered,
            user_list_val_filtered, RMax(), verbose=True, eta = 0.2, lamb = 0.01, random_seed=seed, num_total_iter_training = 121)[0]),
        ('avg', train(train_data, val_data, user_list_train_filtered,
            user_list_val_filtered, RAvg(), verbose=True, eta = 0.2, lamb = 0.1, random_seed=seed, num_total_iter_training = 201)[0]),
        ('tstruct', train(train_data, val_data, user_list_train_filtered,
            user_list_val_filtered, TStruct(), verbose=True,  eta=0.1, lamb= 0.01, random_seed=seed, num_total_iter_training = 121)[0]),
        ('struct-pauc', train(train_data, val_data, user_list_train_filtered,
            user_list_val_filtered, Baseline_Struct_Pauc(), verbose=True, eta=0.2, lamb=0.1, random_seed=seed, num_total_iter_training = 201)[0]),
        ('greedy', greedy_surrogate.train(train_data, val_data,
            user_list_train_filtered, user_list_val_filtered, user_beta_train,
            user_beta_val, k, verbose=True, eta=0.02, lamb=0.01, random_seed=seed, num_total_iter_training = 121)[0]),
       ('prec@k', preck_surrogate.train(train_data, val_data,
           user_list_train_filtered, user_list_val_filtered, user_beta_train,
           user_beta_val, k, dataset, verbose=True, eta=0.1, lamb=0.1, random_seed=seed, cv_flag = False, num_total_iter_training = 121)[0])
    ]
    all_metrics += run_metrics

print('../results/' + dataset + '/test_results/FinalTestResult-testcomp-{}.txt'.format(k))
orig_stdout = sys.stdout
f = open('../results/' + dataset + '/test_results/FinalTestResult-testcomp-{}.txt'.format(k), 'w')
sys.stdout = f

# compute test auc-rel-k at best epoch (from validation)
for name, metric in all_metrics:
    best_iter = np.where(np.asarray(metric.loss_opt_list_train)==np.min(metric.loss_opt_list_train))[0][0]
    w = metric.w_list[best_iter]
                 
    loss_opt_test = 0
    k_minus_w_opt_test = 0

    # sort data once for both micro 
    user_feat = test_data.drop(['user', 'label'], axis = 1).values
    y_scores = user_feat.dot(w)
    data_true = deepcopy(test_data)
    data_true['scores'] = y_scores
    data_true = data_true.sort_values(by='scores', ascending=False)
    data_true = data_true.reset_index(drop=True)

    print('k:', k)
    print('Dataset:', dataset)
    print('Name:', name)
    print('    ', loss_opt_test/len(user_list_test_filtered))
    print('    ', utils.compute_micro(data_true, user_list_test_filtered, user_beta_test, w, k))

sys.stdout = orig_stdout
f.close()

print("Total time taken is ", time.time() - tic_all)