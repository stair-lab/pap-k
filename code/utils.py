import numpy as np
import pandas as pd
import sklearn.metrics as skm
import os

class Metrics:
    def __init__(self):
        self.loss_opt_list_train = []
        self.loss_opt_list_val = []
        self.k_minus_w_opt_list_train = []
        self.k_minus_w_opt_list_val = []
        self.grad_w_list = []
        self.w_list = []
        
        self.eta_lr = None
        self.lamb_reg = None

        self.micro_auc_rel_k_list_val = []
        self.micro_auc_rel_k_list_train = []
    def to_dict(self, best_iter):
        return {
            "eta_lr": self.eta_lr,
            "lamb_reg": self.lamb_reg,
            "loss_opt_list_train": self.loss_opt_list_train,
            "loss_opt_list_val": self.loss_opt_list_val,
            "w_model": self.w_list[best_iter].tolist(),
            "micro_auc_rel_k_list_val": self.micro_auc_rel_k_list_val,
            "micro_auc_rel_k_list_train": self.micro_auc_rel_k_list_train,
        }


def compute_micro(data_true, user_list_filtered, user_beta_data, w, k):
    aucrelkmicrolist = []
    for user_id in user_list_filtered:
        user_df = data_true[data_true['user'] == user_id]
        beta = int(user_beta_data[user_id])
        users_scores = user_df[user_df['label'] == 1].scores.values[:beta].tolist()
        users_labels = [1.0] * beta
        
        users_scores.extend(user_df[user_df['label'] == 0].scores.values[:k].tolist())
        users_labels.extend([0.0] * k)

        try:
            aucrelkmicrolist.append(skm.roc_auc_score(users_labels, users_scores))
        except:
            if(users_labels[0] == 1):
                aucrelkmicrolist.append(1.0)
            else:
                aucrelkmicrolist.append(0.0)
    
    return sum(aucrelkmicrolist)/len(aucrelkmicrolist)


def compute_pauc(data_true, user_list_filtered, user_beta_data, w, k):
    scores_positive = data_true[data_true['label'] == 1].scores.values.tolist()
    scores_negative = data_true[data_true['label'] == 0].scores.values.tolist()
        
    kU = k * len(user_list_filtered)
    n_plus = len(scores_positive)
    scores_positive_filtered = scores_positive
    scores_negative_filtered = scores_negative[:kU]
    scores = scores_positive_filtered + scores_negative_filtered
    labels = [1.0] * n_plus + [0.0] * kU
    try:
        pauc = skm.roc_auc_score(labels, scores)
    except:
        if(labels[0] == 1):
            pauc = 1.0
        else:
            pauc = 0.0
    return pauc

def compute_preck(data_true, user_list_filtered, user_beta_data, w, k):
    ordered_indices = np.argsort(-data_true.scores.values)
    print(np.sum(data_true.label.values[ordered_indices[:k]]), '/', np.sum(data_true.label.values))
    return np.sum(data_true.label.values[ordered_indices[:k]]) / k

def compute_auck(data_true, user_list_filtered, user_beta_data, w, k):
    ordered_indices = np.argsort(-data_true.scores.values)
    labels = data_true.label.values[:k]
    scores = data_true.scores.values[:k]
    try:
        auck = skm.roc_auc_score(labels, scores)
    except:
        if(labels[0] == 1):
            auck = 1.0
        else:
            auck = 0.0
    return auck


def load_data(dataset):
    if dataset == 'behance':
        features = 150
        with open('../data/behance/train_data_d150.tsv', 'rb') as f:
            train_data = pd.read_csv(f, sep='\t')
        with open('../data/behance/val_data_d150.tsv', 'rb') as f:
            val_data = pd.read_csv(f, sep='\t')
        with open('../data/behance/test_data_d150.tsv', 'rb') as f:
            test_data = pd.read_csv(f, sep='\t')
    elif dataset == 'citation':
        features = 155
        with open('../data/citation/train_data_d155.tsv', 'rb') as f:
            train_data = pd.read_csv(f, sep='\t')
        with open('../data/citation/val_data_d155.tsv', 'rb') as f:
            val_data = pd.read_csv(f, sep='\t')
        with open('../data/citation/test_data_d155.tsv', 'rb') as f:
            test_data = pd.read_csv(f, sep='\t')
    elif dataset == 'movielens':
        features = 90
        with open('../data/movielens/train_data_d90.tsv', 'rb') as f:
            train_data = pd.read_csv(f, sep='\t')
        with open('../data/movielens/val_data_d90.tsv', 'rb') as f:
            val_data = pd.read_csv(f, sep='\t')
        with open('../data/movielens/test_data_d90.tsv', 'rb') as f:
            test_data = pd.read_csv(f, sep='\t')
    return train_data, val_data, test_data
