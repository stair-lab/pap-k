from utils import Metrics, compute_micro
import numpy as np
from copy import deepcopy
import json

def subgradient(w, train_data, user_list_train_filtered, user_beta_train, k):
    Deltakp_arr = np.zeros(k+1)
    g_arr = np.zeros((k+1, w.shape[0]))
    for user_id in user_list_train_filtered:
        user_df_train = train_data[train_data['user'] == user_id]
        user_df_pos = user_df_train[user_df_train['label'] == 1]
        user_df_neg = user_df_train[user_df_train['label'] == 0]

        train_feat_pos = user_df_pos.drop(['user', 'label'], axis = 1).values
        train_feat_neg = user_df_neg.drop(['user', 'label'], axis = 1).values
        n_plus = user_df_pos.shape[0]
        n_minus = user_df_neg.shape[0]
        beta = int(user_beta_train[user_id])
        
        train_pos_scores = np.dot(train_feat_pos, w)
        sorted_pos_indices = np.argsort(-1 * train_pos_scores)
        
        train_neg_scores = np.dot(train_feat_neg, w)
        sorted_neg_indices = np.argsort(-1 * train_neg_scores)
        
        for kp in np.arange(0, min(n_plus, k+1)):
            Dkp = (min(k, n_plus) - kp) / (n_plus - kp)
            sum_lower_pos = np.sum(train_pos_scores[sorted_pos_indices[kp:]], axis=0)
            sum_higher_neg = np.sum(train_neg_scores[sorted_neg_indices[:k-kp]], axis=0)
            Deltakp = k - kp - Dkp * sum_lower_pos + sum_higher_neg
            sum_lower_pos = np.sum(train_feat_pos[sorted_pos_indices[kp:]], axis=0)
            sum_higher_neg = np.sum(train_feat_neg[sorted_neg_indices[:k-kp]], axis=0)
            g = sum_higher_neg - Dkp * sum_lower_pos
            Deltakp_arr[kp] += Deltakp / min(n_plus, k+1)
            g_arr[kp] += g
    g_arr = g_arr/len(user_list_train_filtered)
    return g_arr[np.argmax(Deltakp_arr)], np.max(Deltakp_arr)


def train(
    train_data,
    val_data,
    user_list_train_filtered,
    user_list_val_filtered,
    user_beta_train,
    user_beta_val,
    k,
    dataset,
    eta=0.1,
    lamb=0.1,
    tolerance=1e-4,
    num_iter_val=5,
    num_total_iter_training=6,
    random_seed = 786,
    kU=None,
    cv_flag = True,
    verbose=False):

    np.random.seed(random_seed)

    user_feat = val_data.drop(['user', 'label'], axis = 1).values
    user_feat_train = train_data.drop(['user', 'label'], axis = 1).values
    w = np.random.normal(0, 1, user_feat.shape[1])

    metrics = Metrics()
    metrics.eta_lr = eta
    metrics.lamb_reg = lamb
    print("running for eta", eta, "and lambda", lamb)

    for i in range(num_total_iter_training):
        grad, loss = subgradient(w, train_data, user_list_train_filtered, user_beta_train, k)
        grad += lamb*w
        w = w - (eta/np.sqrt(i+1)) * grad
        metrics.w_list.append(w)
        metrics.loss_opt_list_train.append(loss)

        y_scores = user_feat_train.dot(w)
        data_true = deepcopy(train_data)
        data_true['scores'] = y_scores
        data_true = data_true.sort_values(by='scores', ascending=False)
        data_true = data_true.reset_index(drop=True)
        metrics.micro_auc_rel_k_list_train.append(compute_micro(data_true, user_list_train_filtered, user_beta_train, w, k))
        
        if verbose:
            print('Epoch', i+1, 'completed out of',num_total_iter_training, 'for prec@k loss train:', metrics.loss_opt_list_train[-1])
            print('Epoch', i+1, 'completed out of',num_total_iter_training, 'for prec@k grad train:', np.linalg.norm(grad))
        
        # evaluate combined weights
        if(cv_flag):
            if i % num_iter_val == 0:
                y_scores = user_feat.dot(w)
                data_true = deepcopy(val_data)
                data_true['scores'] = y_scores
                data_true = data_true.sort_values(by='scores', ascending=False)
                data_true = data_true.reset_index(drop=True)
                metrics.micro_auc_rel_k_list_val.append(compute_micro(data_true, user_list_val_filtered, user_beta_val, w, k))

                if verbose:
                    print("\n")
                    print('Epoch', i+1, 'completed out of',num_total_iter_training, 'for prec@k loss val:', metrics.micro_auc_rel_k_list_val[-1])
                    print("\n")

    return metrics, None
