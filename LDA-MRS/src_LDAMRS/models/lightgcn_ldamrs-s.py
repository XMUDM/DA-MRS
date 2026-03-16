# coding: utf-8

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender

class LightGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']

        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model'] 
        
        self.knn_k = config['knn_k']
        self.n_layers = config['n_mm_layers']
        self.mm_image_weight = config['mm_image_weight']

        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.kl_weight = config['kl_weight']
        self.neighbor_weight = config['neighbor_weight']
        self.neighbor_loss_weight = config['neighbor_loss_weight']

        self.alpha = config['alpha']
        self.beta = config['beta']

        self.build_item_graph = True

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj, self.R = self.get_norm_adj_mat()
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                       allow_pickle=True).item()
        
        # self.mm_adj, self.v_sim, self.t_sim = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), config)
        self.text_adj, self.image_adj, self.session_adj, self.v_sim, self.t_sim = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), config)


    def get_knn_adj_mat(self, v_embeddings, t_embeddings, config):
        v_context_norm = v_embeddings.div(torch.norm(v_embeddings, p=2, dim=-1, keepdim=True))
        v_sim = torch.mm(v_context_norm, v_context_norm.transpose(1, 0))

        t_context_norm = t_embeddings.div(torch.norm(t_embeddings, p=2, dim=-1, keepdim=True))
        t_sim = torch.mm(t_context_norm, t_context_norm.transpose(1, 0))

        mask_v = v_sim < v_sim.mean()
        mask_t = t_sim < t_sim.mean()

        t_sim[mask_v] = 0
        v_sim[mask_t] = 0  
        t_sim[mask_t] = 0
        v_sim[mask_v] = 0  

        index_x = []
        index_v = []
        index_t = []
        index_y = []

        for i in range(self.n_items):
            item_num = len(torch.nonzero(t_sim[i]))
            if item_num <= self.knn_k:
                _,v_knn_ind = torch.topk(v_sim[i], item_num)
                _,t_knn_ind = torch.topk(t_sim[i], item_num)
            else:
                _,v_knn_ind = torch.topk(v_sim[i], self.knn_k)
                _,t_knn_ind = torch.topk(t_sim[i], self.knn_k)

            index_x.append(torch.ones_like(t_knn_ind) * i)
            index_v.append(v_knn_ind)
            index_t.append(t_knn_ind)

        index_x = torch.cat(index_x, dim=0).cuda() # source node
        index_v = torch.cat(index_v, dim=0).cuda() # target node
        index_t = torch.cat(index_t, dim=0).cuda()

        adj_size = (self.n_items, self.n_items)

        v_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_v)), 0)        
        image_adj = self.compute_normalized_laplacian(v_indices,  torch.ones_like(v_indices[0]), self.n_items, normalization='sym')
        t_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_t)), 0)        
        text_adj = self.compute_normalized_laplacian(t_indices,  torch.ones_like(t_indices[0]), self.n_items, normalization='sym')        
        
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                       allow_pickle=True).item()
        
        index_x = []
        index_y = []
        values = []
        for i in range(self.n_items):
            index_x.append(i)
            index_y.append(i)
            values.append(1)
            if i in item_graph_dict.keys():
                item_graph_sample = item_graph_dict[i][0]
                item_graph_weight = item_graph_dict[i][1]

                for j in range(len(item_graph_sample)):
                    index_x.append(i)
                    index_y.append(item_graph_sample[j])
                    values.append(item_graph_weight[j])

        index_x = torch.tensor(index_x, dtype=torch.long)
        index_y = torch.tensor(index_y, dtype=torch.long)
        indices = torch.stack((index_x, index_y), 0).to(self.device)
        session_adj = self.compute_normalized_laplacian(indices,  torch.ones_like(indices[0]), self.n_items, normalization='sym')

        return text_adj, image_adj, session_adj, v_sim, t_sim
        
    def compute_normalized_laplacian(self, edge_index, edge_weight, num_nodes, normalization='sym'):
        from torch_scatter import scatter_add
        row, col = edge_index[0].long(), edge_index[1].long()
        deg = scatter_add(edge_weight.float(), row, dim=0, dim_size=num_nodes)

        if normalization == 'sym':
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'rw':
            deg_inv = 1.0 / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[row] * edge_weight
        size = torch.Size((num_nodes, num_nodes))
        return torch.sparse.FloatTensor(edge_index, edge_weight.float(), size)

    def generate_pesudo_labels(self, emb, i_id):
        n_emb = F.normalize(emb[i_id], dim=1)
        n_aug_emb = F.normalize(emb, dim=1) 
        prob = torch.mm(n_emb, n_aug_emb.transpose(0, 1))
        prob = F.softmax(prob, dim=1) # 按行
        _ , mm_pos_ind = torch.topk(prob, self.knn_k, dim=-1)
        
        v_sim_prob = self.v_sim.clone()
        v_sim_prob.scatter_(1, mm_pos_ind, 0)
        t_sim_prob = self.t_sim.clone()
        t_sim_prob.scatter_(1, mm_pos_ind, 0)

        _, vs_ind = torch.topk(v_sim_prob[i_id], self.knn_k, dim=-1)
        _, ts_ind = torch.topk(t_sim_prob[i_id], self.knn_k, dim=-1)

        return mm_pos_ind, ts_ind, vs_ind
    
    def neighbor_discrimination(self, mm_positive, s_positive, id_all_index, emb, aug_emb, neighbor_weight, temperature=0.2):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=2)

        n_aug_emb = F.normalize(aug_emb, dim=1)
        n_emb = F.normalize(emb, dim=1)

        mm_pos_emb = n_aug_emb[mm_positive]
        s_pos_emb = n_aug_emb[s_positive]
        batch_emb = n_aug_emb[id_all_index] 

        emb2 = torch.reshape(n_emb, [-1, 1, self.embedding_dim])
        emb2 = torch.tile(emb2, [1, self.knn_k, 1]) # 锚点

        mm_pos_score = score(emb2, mm_pos_emb) # 正 
        mm_pos_score = torch.sum(torch.exp(mm_pos_score / temperature), dim=1)
        s_pos_score = score(emb2, s_pos_emb) # 小负
        s_pos_score = torch.sum(torch.exp(s_pos_score / temperature), dim=1)

        # all_score = score(emb2, batch_emb) # 大负
        ttl_score = torch.matmul(n_emb, batch_emb.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1) # 1
        
        cl_loss = - torch.log(mm_pos_score / (ttl_score - s_pos_score + s_pos_score * neighbor_weight + 10e-10))

        return torch.mean(cl_loss)

    def InfoNCE(self, view1, view2, temperature=0.2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)        
        return torch.mean(cl_loss)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        R = sp.coo_matrix(L[:self.n_users, self.n_users:])
        R_row = R.row
        R_col = R.col
        R_i = torch.LongTensor(np.array([R_row, R_col]))
        R_data = torch.FloatTensor(R.data)

        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes))), torch.sparse.FloatTensor(R_i, R_data, torch.Size((self.n_users, self.n_items)))

    def forward(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj.cuda(), ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        del ego_embeddings, side_embeddings
        mm_adj = self.text_adj * 0.34 + self.image_adj * 0.33 + self.session_adj * 0.33
        h_m = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_m = torch.sparse.mm(mm_adj.cuda(), h_m)
    
        u_m = torch.sparse.mm(self.R.cuda(), h_m)

        return u_g_embeddings, i_g_embeddings, h_m, u_m

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_embeddings, item_embeddings, h_m, u_m = self.forward() # 
        self.build_item_graph = False        

        u_idx = torch.unique(users, return_inverse=True, sorted=False)
        i_idx = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        u_id = u_idx[0]
        i_id = i_idx[0]

        mm_knn_ind, vs_knn_ind, ts_knn_ind = self.generate_pesudo_labels(h_m, i_id)
        vs_knn_ind_flatten = vs_knn_ind.flatten()
        ts_knn_ind_flatten = ts_knn_ind.flatten()
        batch_i_id = torch.unique(torch.cat((vs_knn_ind_flatten, ts_knn_ind_flatten, i_id)), return_inverse=True, sorted=False)[0] 

        neighbor_dis_loss_m = (self.neighbor_discrimination(mm_knn_ind, vs_knn_ind, batch_i_id, h_m[i_id], h_m, self.neighbor_weight) + self.neighbor_discrimination(mm_knn_ind, ts_knn_ind, batch_i_id, h_m[i_id], h_m, self.neighbor_weight))
        
        neighbor_dis_loss = neighbor_dis_loss_m 
        
        KLU_loss = torch.mean(self.InfoNCE(user_embeddings[u_id], u_m[u_id]))
        
        # D-BPR 
        p_weight, n_weight = self.get_weight_modal(users, pos_items, neg_items, user_embeddings, h_m) 

        ua_embeddings = user_embeddings 
        u_g_embeddings = ua_embeddings[users]
        ia_embeddings = item_embeddings + h_m
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, p_weight, n_weight)  
        KLU_loss = KLU_loss * self.kl_weight
        neighbor_dis_loss = self.neighbor_loss_weight * neighbor_dis_loss
        return batch_mf_loss + neighbor_dis_loss + KLU_loss
    
    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_embeddings, item_embeddings, h_m, u_m = self.forward() # 

        ua_embeddings = user_embeddings
        u_g_embeddings = ua_embeddings[users]
        ia_embeddings = item_embeddings + h_m

        score = torch.matmul(u_g_embeddings, ia_embeddings.transpose(0, 1))
        return score, ia_embeddings

    def get_weight_modal(self, users, pos_items, neg_items, user_embeddings, h_m):
        u_g_embeddings = user_embeddings[users]
        
        # pos 
        p_m = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_m[pos_items], dim=-1)), dim=1)
        # neg
        n_m = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_m[neg_items], dim=-1)), dim=1)
        
        p_difference = F.sigmoid(p_m - n_m).data        
        p_value = F.sigmoid(p_m).data
        n_value = F.sigmoid(n_m).data
        pos_weight = torch.pow(p_value, self.alpha) * torch.pow(p_difference, self.alpha)
        pos_weight = torch.clamp(pos_weight, 0, 1).data

        # mmi < y < mmj 
        mask = torch.zeros_like(n_value)
        n_mean_value = torch.mean(n_value)
        mask[(p_value < n_mean_value) & (p_value < n_value)] = 1

        neg_difference = torch.pow(torch.exp(n_m - p_m), self.beta) * mask
        neg_value = torch.exp(-n_mean_value * self.beta)
        neg_weight = torch.clamp(neg_value * neg_difference, 0, 1).data
        return pos_weight, neg_weight

    def bpr_loss(self, users, pos_items, neg_items, p_weight, n_weight):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        p_maxi = torch.log(F.sigmoid(pos_scores - neg_scores)) * p_weight
        n_maxi = torch.log(F.sigmoid(neg_scores - pos_scores)) * n_weight
        mf_loss = -torch.mean(p_maxi + n_maxi)
        return mf_loss
