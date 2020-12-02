import warnings
warnings.filterwarnings('ignore')
import os,sys,inspect
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pickle
from copy import deepcopy
import json
pd.set_option('display.max_rows', 500)
import greedy_surrogate
import preck_surrogate
import utils
from surrogates import *


saveit = True
margin_type = sys.argv[1]       # Provide margin type as STRONG_BETA_MARGIN, BETA_MARGIN, and MODERATE_BETA_MARGIN
                                # while running this file

k = 30

np.random.seed(325)
num_users = 1
num_pos = 250
num_neg = num_pos*8
features = 5
margin = 1.0
beta = k
dataset = 'synth'
dump_to_file = False
n_iters = 151

col_names = ['user']

for i in range(features):
    col_names.append('f' + str(i))
col_names.append('label')

df_list = []

for i in range(num_users):
    num_pos_user = np.random.randint(num_pos - 20, num_pos + 20, size=1)[0]

    mean_user_pos = np.random.multivariate_normal(0*np.ones(features), 0.2*np.eye(features))
    dat_pos_user = np.random.multivariate_normal(mean = mean_user_pos, cov = np.identity(features), size = num_pos_user)

    mean_user_neg = np.random.multivariate_normal(0.4*np.ones(features), 0.2*np.eye(features))
    dat_neg_user = np.random.multivariate_normal(mean = mean_user_neg, cov = np.identity(features), size = num_neg)

    optimal_w = mean_user_pos - mean_user_neg
    optimal_w = optimal_w / np.linalg.norm(optimal_w)
    if(margin_type == 'STRONG_BETA_MARGIN'):
        threshold = np.dot((mean_user_pos + mean_user_neg)/2, optimal_w)
        dat_pos_user = dat_pos_user[np.dot(dat_pos_user, optimal_w) - threshold >= 0.5*margin]
        dat_neg_user = dat_neg_user[np.dot(dat_neg_user, optimal_w) - threshold <= -0.5*margin]
    elif(margin_type == 'MODERATE_BETA_MARGIN'):
        pos_scores = np.dot(dat_pos_user, optimal_w)
        top_beta_threshold = np.sort(pos_scores)[-beta]
        dat_neg_user = dat_neg_user[np.dot(dat_neg_user, optimal_w) - top_beta_threshold <= -1*margin]
        neg_scores = np.dot(dat_neg_user, optimal_w)
        top_negative_threshold = np.sort(neg_scores)[-1]
        dat_pos_user = dat_pos_user[np.dot(dat_pos_user, optimal_w) - top_negative_threshold >= 0]
    elif(margin_type == 'BETA_MARGIN'):
        avg_pos_user = np.mean(dat_pos_user, axis=0)
        threshold = np.dot(avg_pos_user, optimal_w)
        dat_neg_user = dat_neg_user[np.dot(dat_neg_user, optimal_w) - threshold <= -1*margin]
    elif(margin_type == 'WEAK_BETA_MARGIN'):
        pos_scores = np.dot(dat_pos_user, optimal_w)
        threshold = np.sort(pos_scores)[-beta]
        dat_neg_user = dat_neg_user[np.dot(dat_neg_user, optimal_w) - threshold <= -1*margin]
        

    print("Positives: {}, Negatives: {}".format(dat_pos_user.shape[0], dat_neg_user.shape[0]))

    # append column of zeroes
    dat_pos_user = np.hstack((dat_pos_user, np.ones((len(dat_pos_user),1))))
    dat_neg_user = np.hstack((dat_neg_user, np.zeros((len(dat_neg_user),1))))

    dat_user = np.concatenate((dat_pos_user, dat_neg_user), axis = 0)
    dat_user = np.concatenate((i*np.ones(dat_user.shape[0])[:, None], dat_user), axis = 1)

    pd_data_user = pd.DataFrame(data=dat_user, columns=col_names)
    pd_data_user.user = pd_data_user.user.astype('int')

    df_list.append(pd_data_user)

# split into train and test data
dat = pd.concat(df_list)

dat_features = dat.drop(['user', 'label'], axis = 1).values
dat.iloc[:, 1:(features+1)] = dat_features

dat = dat.sample(frac=1).reset_index(drop=True)

msk = np.random.rand(len(dat))

train_data = dat[msk < 0.7]
val_data = dat[np.logical_and(msk > 0.7, msk < 0.85)]
test_data = dat[msk > 0.85]

train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

for i in range(features):
    f = 'f{}'.format(i)
    feature_range = train_data[f].max() - train_data[f].min()
    feature_mean = (train_data[f].max() + train_data[f].min()) / 2
    train_data[f] = (train_data[f] - feature_mean) / feature_range
    val_data[f] = (val_data[f] - feature_mean) / feature_range
    test_data[f] = (test_data[f] - feature_mean) / feature_range

user_list_train = list(set(train_data['user']))
user_list_val= list(set(val_data['user']))
user_list_test = list(set(test_data['user']))

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
print("val data is ", val_data.shape)
print("test data is ", test_data.shape)
print("users with at least one positive in train are", len(user_list_train_filtered))
print("users with at least one positive in val are", len(user_list_val_filtered))
print("users with at least one positive in test are", len(user_list_test_filtered))

def sort_order(user_feat, w, limit=None):
    y_scores = user_feat.dot(w)
    indi_sort = np.argsort(-y_scores)
    return  indi_sort, y_scores


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
    tolerance=1e-6,
    num_total_iter_training=151,
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
            print('    sorting elapsed time:   ', sorting_time)
            print('    surrogate elapsed time: ', surrogate_time)
            print('    metrics elapsed time:   ', time.time() - metrics_start_time)
            print('Epoch', num_iter+1, 'completed out of',num_total_iter_training, 'for', surr.name, 'loss train:',metrics.loss_opt_list_train[-1])
            print('Epoch', num_iter+1, 'completed out of',num_total_iter_training, 'for', surr.name, 'grad_w:',metrics.grad_w_list[-1])
            print('Epoch', num_iter+1, 'completed out of',num_total_iter_training, 'for', surr.name, 'microaucrelk train:',metrics.micro_auc_rel_k_list_train[-1])
        else:
            print('epoch', num_iter+1, 'completed. micro:{}'.format(metrics.micro_auc_rel_k_list_train[-1]))
    
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
                print('Epoch', num_iter+1, 'completed out of',num_total_iter_training, 'for', surr.name, 'loss val:',metrics.loss_opt_list_val[-1])
                print('Epoch', num_iter+1, 'completed out of',num_total_iter_training, 'for', surr.name, 'microaucrelk val:',metrics.micro_auc_rel_k_list_val[-1])
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

    best_iter = (np.where(np.asarray(metrics.loss_opt_list_train)==np.min(metrics.loss_opt_list_train))[0][0]//num_iter_val)*num_iter_val

    best_iter = (np.where(np.asarray(metrics.loss_opt_list_train)==np.min(metrics.loss_opt_list_train))[0][0]//num_iter_val)*num_iter_val
    best_microaucrelk = metrics.micro_auc_rel_k_list_val[best_iter//num_iter_val]
    print('Best micro aucrelk at iter: %d (metric: %f)' % (best_iter, best_microaucrelk))
    
    return metrics, w


if(margin_type == 'STRONG_BETA_MARGIN'):
    all_metrics = [('max', train(train_data, val_data, user_list_train_filtered,
       user_list_val_filtered, RMax(), verbose=False, num_total_iter_training=n_iters, eta = 2, lamb = 0.001)[0])]
elif(margin_type == 'BETA_MARGIN'):
    all_metrics = [('avg', train(train_data, val_data, user_list_train_filtered,
       user_list_val_filtered, RAvg(), verbose=True, num_total_iter_training=n_iters, eta = 3, lamb = 0.001)[0])]
elif(margin_type == 'MODERATE_BETA_MARGIN'):
    all_metrics = [('tstruct', train(train_data, val_data, user_list_train_filtered,
    user_list_val_filtered, TStruct(), verbose=False, num_total_iter_training=n_iters, eta = 3, lamb = 0.001)[0])]
else:
    pass


surrogates = [RMax(), RAvg(), TStruct(), RRamp()]
losses = []

name, metric = all_metrics[0]
for surr in surrogates:
    loss_opt_list = []
    for w in metric.w_list:
        loss_opt_train = 0
        for user_id in user_list_train_filtered:
            user_df_train = train_data[train_data['user'] == user_id]

            if(len(user_df_train.label.unique()) == 1):
                if(user_df_train.label.iloc[0] == 1.0):
                    loss_opt_train += 0.0
                else:
                    loss_opt_train += 1.0
            else:
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

                pi_opt_train, score_mat_train = surr.compute_pi(sorted_scores_pos, sorted_scores_neg, w, k, beta)
                loss_opt_user_train, _, _, _ = surr.loss(
                    pi_opt_train, sorted_scores_pos, sorted_scores_neg, k, beta)

            loss_opt_train += loss_opt_user_train
        loss_opt_list.append(loss_opt_train)
    losses.append(loss_opt_list)

fig, ax1 = plt.subplots(figsize=(20,10))

linestyles = ['-', '--', '-.', ':']
for surr, loss_opt_list, linestyle in zip(surrogates, losses, linestyles):
    name = surr.name
    test_points = len(loss_opt_list)
    ax1.plot(np.arange(test_points), loss_opt_list, linestyle=linestyle, label=name, linewidth=6)

fontP = FontProperties()
fontP.set_size('36')
ax1.set_xlabel('Iteration Number', fontsize='40')
ax1.set_ylabel('Surrogate Value', fontsize='40')
ax1.set_ylim([0,2])
plt.legend(prop=fontP)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
ax1.set_aspect(50)

if(margin_type == 'STRONG_BETA_MARGIN'):
    plt.title('Surrogate Behavior in Strong β-margin', fontsize='40')
elif(margin_type == 'BETA_MARGIN'):
    plt.title('Surrogate Behavior in β-margin', fontsize='40')
elif(margin_type == 'MODERATE_BETA_MARGIN'):
    plt.title('Surrogate Behavior in Moderate β-margin', fontsize='40')
else:
    pass

if(saveit):
    if(margin_type == 'STRONG_BETA_MARGIN'):
        plt.savefig('../plots/strong_beta_margin.png', dpi=500, bbox_inches='tight', fmt = 'png')
    elif(margin_type == 'BETA_MARGIN'):
        plt.savefig('../plots/beta_margin.png', dpi=500, bbox_inches='tight', fmt = 'png')
    elif(margin_type == 'MODERATE_BETA_MARGIN'):
        plt.savefig('../plots/moderate_beta_margin.png', dpi=500, bbox_inches='tight', fmt = 'png')
    else:
        pass
else:
    plt.show()