from utils import Metrics, compute_micro
import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy
from collections import defaultdict
import random

def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def merge_micro(val_data, w1, w2, user_list_val_filtered, user_beta_val, k, H=20):
    
    def construct_matrices(val_feat_pos, val_feat_neg, w):
        val_pos_scores = np.dot(val_feat_pos, w)
        val_neg_scores = np.dot(val_feat_neg, w)
        l = (val_pos_scores.reshape((-1,1)) > val_neg_scores)
        Delta = val_pos_scores.reshape((-1,1)) - val_neg_scores
        return val_pos_scores, val_neg_scores, l, Delta
    
    phi_dict = defaultdict(int)
    for user_id in user_list_val_filtered:
        user_df_val = val_data[val_data['user'] == user_id]
        user_df_pos = user_df_val[user_df_val['label'] == 1]
        user_df_neg = user_df_val[user_df_val['label'] == 0]

        val_feat_pos = user_df_pos.drop(['user', 'label'], axis = 1).values
        val_feat_neg = user_df_neg.drop(['user', 'label'], axis = 1).values
        n_plus = user_df_pos.shape[0]
        n_minus = user_df_neg.shape[0]
        beta = int(user_beta_val[user_id])

        pos_scores_1, neg_scores_1, l1, Delta1 = construct_matrices(val_feat_pos, val_feat_neg, w1)
        pos_scores_2, neg_scores_2, l2, Delta2 = construct_matrices(val_feat_pos, val_feat_neg, w2)
        T12 = np.logical_and(l1, l2)
        T_12 = np.logical_and(np.logical_not(l1), l2)
        T1_2 = np.logical_and(l1, np.logical_not(l2))
        Delta_ratio = Delta1 / Delta2

        for alpha_on_first in [False, True]:
            for alpha in np.linspace(0, 1, H):
                # select top k negative columns and top beta positive rows
                if alpha_on_first:
                    combined_neg_scores = alpha * neg_scores_1 + neg_scores_2
                    combined_pos_scores = alpha * pos_scores_1 + pos_scores_2
                else:
                    combined_neg_scores = neg_scores_1 + alpha * neg_scores_2
                    combined_pos_scores = pos_scores_1 + alpha * pos_scores_2
                
                Gamma = np.zeros((n_plus, n_minus))
                Gamma_minus = np.zeros((n_plus, n_minus))
                Gamma_plus = np.zeros((n_plus, n_minus))
                
                if n_minus == k:
                    Gamma_minus[:, :] = 1
                else:
                    neg_top_indices = np.argpartition(-1 * combined_neg_scores, k)[:k]
                    Gamma_minus[:, neg_top_indices] = 1
                if n_plus == beta:
                    Gamma_plus[:, :] = 1
                else:
                    pos_top_indices = np.argpartition(-1 * combined_pos_scores, beta)[:beta]
                    Gamma_plus[pos_top_indices, :] = 1
                
                Gamma = np.logical_and(Gamma_minus, Gamma_plus)
            
                # only count top k negatives and top beta positives
                Tp12 = np.logical_and(T12, Gamma)
                Tp_12 = np.logical_and(T_12, Gamma)
                Tp1_2 = np.logical_and(T1_2, Gamma)
                
                if alpha_on_first:
                    F12 = np.sum(Tp12 * (-1 * (1/(Delta_ratio)) < alpha))
                    F_12 = np.sum(Tp_12 * (-1 * (1/(Delta_ratio)) > alpha))
                    F1_2 = np.sum(Tp1_2 * (-1 * (1/(Delta_ratio)) < alpha))
                else:
                    F12 = np.sum(Tp12 * (-1 * Delta_ratio < alpha))
                    F_12 = np.sum(Tp_12 * (-1 * Delta_ratio < alpha))
                    F1_2 = np.sum(Tp1_2 * (-1 * Delta_ratio > alpha))
                    
                phi = F12 + F_12 + F1_2
                
                phi_dict[(alpha_on_first, alpha)] += phi
        # compute combined classifier
        alpha_on_first, alpha = max(phi_dict, key=phi_dict.get)
    if alpha_on_first:
        return alpha * w1 + w2
    else:
        return w1 + alpha * w2

def train(
    train_data,
    val_data,
    user_list_train_filtered,
    user_list_val_filtered,
    user_beta_train,
    user_beta_val,
    k,
    eta=0.1,
    lamb=0.1,
    num_iter_val=5,
    num_total_iter_training=6,
    n_classifiers=5,
    random_seed = 786,
    verbose=False):
    
    np.random.seed(random_seed)
    
    user_list_val_filtered = user_list_train_filtered[0: int(0.2*len(user_list_train_filtered))]
    user_list_train_filtered = list(set(user_list_train_filtered) - set(user_list_val_filtered))
    val_data = train_data[train_data['user'].isin(user_list_val_filtered)]
    train_data = train_data[train_data['user'].isin(user_list_train_filtered)]
    
    metrics = Metrics()
    metrics.eta_lr = eta
    metrics.lamb_reg = lamb

    classifier_list = []
    
    kf = KFold(n_splits=n_classifiers, shuffle=True)
    features = train_data.drop(['user', 'label'], axis = 1)
    labels = train_data['label']
    for _, split_indices in kf.split(features):
        split_features = features.iloc[split_indices].values
        split_labels = labels.iloc[split_indices].values
        num_examples = split_features.shape[0]
        
        w = np.random.normal(0, 1, (split_features.shape[1], ))
        w = w/np.linalg.norm(w)
        for num_iter in np.arange(num_total_iter_training):
            scores = sigmoid(np.dot(split_features, w))
            loss = -1/num_examples * np.sum(split_labels * np.log(scores) + (1-split_labels) * np.log(1-scores))
            print("loss is ", loss)
            dLdwx = (scores - split_labels)*scores*(1-scores)
            grad = 1/num_examples * np.sum(dLdwx.reshape(-1, 1)*split_features)
            grad += lamb * w
            print("grad is ", np.linalg.norm(grad))
            print("\n")
            w = w - (eta/np.sqrt(num_iter+1))*grad
        accuracy = np.sum(split_labels * (scores > 0.5) + (1-split_labels) * (scores < 0.5))
        print('accuracy: {}'.format(accuracy / num_examples))
        classifier_list.append(w)
    print('eta is ', eta, 'and lambda is ', lamb)
    print('\n')

    classifiers_with_metrics = []
    for w in classifier_list:
        user_feat = val_data.drop(['user', 'label'], axis = 1).values
        y_scores = user_feat.dot(w)
        data_true = deepcopy(val_data)
        data_true['scores'] = y_scores
        data_true = data_true.sort_values(by='scores', ascending=False)
        data_true = data_true.reset_index(drop=True)
        metric = compute_micro(data_true, user_list_val_filtered, user_beta_train, w, k)
        classifiers_with_metrics.append((metric, w))
    classifiers_with_metrics.sort(reverse=True, key=lambda x: x[0])
    combined_w = classifiers_with_metrics[0][1]
    for _, w in classifiers_with_metrics[1:]:
        combined_w = merge_micro(val_data, combined_w, w, user_list_val_filtered, user_beta_train, k)
    
    # create dummy metrics
    # need weights and one validation loss for the "best iter" logic
    metrics = Metrics()
    metrics.w_list.append(combined_w)
    metrics.micro_auc_rel_k_list_val.append(0)
    metrics.micro_auc_rel_k_list_train.append(0)
    metrics.loss_opt_list_train.append(0)
    return metrics, None
