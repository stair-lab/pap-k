import numpy as np

class RMax:
    def __init__(self):
        self.name = "max"
        pass


    def loss(self, pi_matrix, sorted_pos_score, sorted_neg_score, k, beta):
        n_plus = sorted_pos_score.shape[0]

        pi_loss = np.sum(pi_matrix) / (beta * k)

        neg_loss = 0
        for j in range(k):
            q_j = np.sum(pi_matrix[:, j])
            neg_loss += q_j*sorted_neg_score[j] 

        neg_loss = neg_loss/(beta*k)

        pos_loss = 0
        for i in range(beta):
            p_i = np.sum(pi_matrix[i, :])
            pos_loss += p_i*sorted_pos_score[n_plus - beta + i]

        pos_loss = pos_loss/(beta*k)

        return pi_loss + neg_loss - pos_loss, pi_loss, neg_loss, pos_loss


    def compute_pi(self, sorted_scores_pos, sorted_scores_neg, w, k, beta):
        n_plus = sorted_scores_pos.shape[0]

        margin = sorted_scores_pos[n_plus-beta:].reshape((-1,1)) - sorted_scores_neg[:k]
        pi_mat = margin <= 1
        
        return pi_mat, None


    def gradient(self, user_feat_pos, user_feat_neg, pi_opt, k, beta):
        n_plus = user_feat_pos.shape[0]

        neg_loss_grad = np.zeros(user_feat_neg.shape[1])

        for j in range(k):
            q_j = np.sum(pi_opt[:, j])
            neg_loss_grad += q_j*user_feat_neg[j, :]

        neg_loss_grad = neg_loss_grad/(beta*k)

        pos_loss_grad = np.zeros(user_feat_pos.shape[1])

        for i in range(beta):
            p_i = np.sum(pi_opt[i, :])
            pos_loss_grad += p_i*user_feat_pos[n_plus - beta + i, :]

        pos_loss_grad = pos_loss_grad/(beta*k)

        rmax_grad = neg_loss_grad - pos_loss_grad

        return rmax_grad


class RAvg:
    def __init__(self):
        self.name = "avg"
        pass


    def loss(self, pi_matrix, sorted_pos_score, sorted_neg_score, k, beta):
        n_plus = sorted_pos_score.shape[0]

        pi_loss = np.sum(pi_matrix) / (beta * k)

        neg_loss = 0
        for j in range(k):
            q_j = np.sum(pi_matrix[:, j])
            neg_loss += q_j*sorted_neg_score[j] 

        neg_loss = neg_loss/(beta*k)

        avg_pos_score = np.sum(sorted_pos_score) / n_plus
        pos_loss = avg_pos_score * np.sum(pi_matrix) / (beta*k)

        return pi_loss + neg_loss - pos_loss, pi_loss, neg_loss, pos_loss


    def compute_pi(self, sorted_scores_pos, sorted_scores_neg, w, k, beta):
        n_plus = sorted_scores_pos.shape[0]
        
        avg_pos_score = np.sum(sorted_scores_pos) / n_plus
        margin = avg_pos_score - sorted_scores_neg[:k]
        pi_mat = np.broadcast_to(margin <= 1, (beta, k))
        
        return pi_mat, None


    def gradient(self, user_feat_pos, user_feat_neg, pi_opt, k, beta):

        n_plus = user_feat_pos.shape[0]
        
        neg_loss_grad = np.zeros(user_feat_neg.shape[1])

        for j in range(k):
            q_j = np.sum(pi_opt[:, j])
            neg_loss_grad += q_j*user_feat_neg[j, :]

        neg_loss_grad = neg_loss_grad/(beta*k)

        pos_loss_grad = np.zeros(user_feat_pos.shape[1])

        avg_feat_pos = np.sum(user_feat_pos, axis=0) / n_plus
        pos_loss_grad = avg_feat_pos * np.sum(pi_opt) / (beta*k)

        ravg_grad = neg_loss_grad - pos_loss_grad

        return ravg_grad


class TStruct:
    def __init__(self):
        self.name = "TS"
        pass


    def loss(self, pi_matrix, sorted_pos_score, sorted_neg_score, k, beta):
        
        n_plus = sorted_pos_score.shape[0]

        pi_loss = np.sum(pi_matrix[:beta, :k]) / (beta * k)

        neg_loss = 0
        for j in range(k):
            q_j = np.sum(pi_matrix[:, j])
            neg_loss += q_j*sorted_neg_score[j] 

        neg_loss = neg_loss/(beta*k)
        
        pos_loss = 0
        for i in range(n_plus):
            p_i = np.sum(pi_matrix[i, :])
            pos_loss += p_i*sorted_pos_score[i]

        pos_loss = pos_loss/(beta*k)

        return pi_loss + neg_loss - pos_loss, pi_loss, neg_loss, pos_loss


    def compute_pi(self, sorted_scores_pos, sorted_scores_neg, w, k, beta):
        n_plus = sorted_scores_pos.shape[0]

        margin = sorted_scores_pos.reshape((-1,1)) - sorted_scores_neg[:k]
        pi_mat = np.zeros((n_plus, k))
        pi_mat[:beta] = margin[:beta] <= 1
        pi_mat[beta:] = margin[beta:] <= 0

        return pi_mat, None


    def gradient(self, user_feat_pos, user_feat_neg, pi_opt, k, beta):
        n_plus = user_feat_pos.shape[0]

        neg_loss_grad = np.zeros(user_feat_neg.shape[1])

        for j in range(k):
            q_j = np.sum(pi_opt[:, j])
            neg_loss_grad += q_j*user_feat_neg[j, :]

        neg_loss_grad = neg_loss_grad/(beta*k)

        pos_loss_grad = np.zeros(user_feat_pos.shape[1])

        for i in range(n_plus):
            p_i = np.sum(pi_opt[i, :])
            pos_loss_grad += p_i*user_feat_pos[i, :]

        pos_loss_grad = pos_loss_grad/(beta*k)

        tstruct_grad = neg_loss_grad - pos_loss_grad

        return tstruct_grad


class RRamp:
    def __init__(self):
        self.name = "ramp"
        pass


    def loss(self, pi_matrix, sorted_pos_score, sorted_neg_score, k, beta):
        n_plus = sorted_pos_score.shape[0]

        pi_loss = np.sum(pi_matrix) / (beta * k)

        neg_loss = 0
        for j in range(k):
            q_j = np.sum(pi_matrix[:, j])
            neg_loss += q_j*sorted_neg_score[j] 

        neg_loss = neg_loss/(beta*k)

        pos_loss = 0
        for i in range(beta):
            p_i = np.sum(pi_matrix[i, :])
            pos_loss += p_i*sorted_pos_score[i]

        pos_loss = pos_loss/(beta*k)

        return pi_loss + neg_loss - pos_loss, pi_loss, neg_loss, pos_loss


    def compute_pi(self, sorted_scores_pos, sorted_scores_neg, w, k, beta):
        margin = sorted_scores_pos[:beta].reshape((-1,1)) - sorted_scores_neg[:k]
        pi_mat = margin <= 1
        
        return pi_mat, None


    def gradient(self, user_feat_pos, user_feat_neg, pi_opt, k, beta):
        # we don't actually optimize this one
        pass
    

class Baseline_Struct_Pauc:
    def __init__(self):
        self.name = "b_pauc"
        pass


    def loss(self, pi_matrix, sorted_pos_score, sorted_neg_score, k, beta):
        # beta is not used.
        n_plus = sorted_pos_score.shape[0]

        pi_loss = np.sum(pi_matrix) / (n_plus * k)

        neg_loss = 0
        for j in range(k):
            q_j = np.sum(pi_matrix[:, j])
            neg_loss += q_j*sorted_neg_score[j] 

        neg_loss = neg_loss/(n_plus*k)

        pos_loss = 0
        for i in range(n_plus):
            p_i = np.sum(pi_matrix[i, :])
            pos_loss += p_i*sorted_pos_score[i]

        pos_loss = pos_loss/(n_plus*k)

        return pi_loss + neg_loss - pos_loss, pi_loss, neg_loss, pos_loss


    def compute_pi(self, sorted_scores_pos, sorted_scores_neg, w, k, beta):
        # beta is not used.
        n_plus = sorted_scores_pos.shape[0]

        margin = sorted_scores_pos.reshape((-1,1)) - sorted_scores_neg[:k]
        pi_mat = margin <= 1

        return pi_mat, None


    def gradient(self, user_feat_pos, user_feat_neg, pi_opt, k, beta):
        # does not use beta.
        n_plus = user_feat_pos.shape[0]

        neg_loss_grad = np.zeros(user_feat_neg.shape[1])

        for j in range(k):
            q_j = np.sum(pi_opt[:, j])
            neg_loss_grad += q_j*user_feat_neg[j, :]

        neg_loss_grad = neg_loss_grad/(n_plus*k)

        pos_loss_grad = np.zeros(user_feat_pos.shape[1])

        for i in range(n_plus):
            p_i = np.sum(pi_opt[i, :])
            pos_loss_grad += p_i*user_feat_pos[i, :]

        pos_loss_grad = pos_loss_grad/(n_plus*k)

        struct_pauc_grad = neg_loss_grad - pos_loss_grad

        return struct_pauc_grad
