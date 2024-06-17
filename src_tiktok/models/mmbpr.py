# coding: utf-8

r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, DiceLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class MMBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(MMBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))

        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat, self.a_feat), -1) 
        elif self.t_feat is not None:
            self.item_raw_features = self.t_feat
        else:
            self.item_raw_features = self.v_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss() # DiceLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        item_embeddings = self.item_linear(self.item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        #neg_e = self.get_item_embedding(neg_item)
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        
        input = torch.cat((torch.sigmoid(pos_item_score), torch.sigmoid(neg_item_score))).to(self.device)
        target = torch.cat((torch.ones_like(pos_item_score), torch.zeros_like(neg_item_score))).to(self.device)
        
        mf_loss = self.loss(pos_item_score, neg_item_score)
        # dice_loss = self.loss(input, target)
        # ce_loss = torch.mean(F.binary_cross_entropy_with_logits(input, target, reduce = False))

        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

# class MMBPR(GeneralRecommender):
#     r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
#     """
#     def __init__(self, config, dataloader):
#         super(MMBPR, self).__init__(config, dataloader)

#         # load parameters info
#         self.u_embedding_size = self.i_embedding_size = config['embedding_size']
#         self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
#         self.idx = 0

#         # define layers and loss
#         self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
#         self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))

#         if self.v_feat is not None and self.t_feat is not None and self.a_feat is not None:
#             self.item_raw_features = torch.cat((self.t_feat, self.v_feat, self.a_feat), -1) 
#         elif self.t_feat is not None:
#             self.item_raw_features = self.t_feat
#         else:
#             self.item_raw_features = self.v_feat

#         self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
#         self.loss = BPRLoss() # DiceLoss()
#         self.reg_loss = EmbLoss()

#         # parameters initialization
#         self.apply(xavier_normal_initialization)

#     def get_user_embedding(self, user):
#         r""" Get a batch of user embedding tensor according to input user's id.

#         Args:
#             user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
#         """
#         return self.u_embedding[user, :]

#     def get_item_embedding(self, item):
#         r""" Get a batch of item embedding tensor according to input item's id.

#         Args:
#             item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
#         """
#         return self.item_embedding[item, :]

#     def forward(self, dropout=0.0):
#         item_embeddings = self.item_linear(self.item_raw_features)
#         item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

#         user_e = F.dropout(self.u_embedding, dropout)
#         item_e = F.dropout(item_embeddings, dropout)
#         return user_e, item_e

#     def calculate_loss(self, interaction, idx):
#         """
#         loss on one batch
#         :param interaction:
#             batch data format: tensor(3, batch_size)
#             [0]: user list; [1]: positive items; [2]: negative items
#         :return:
#         """
#         user = interaction[0]
#         pos_item = interaction[1]
#         neg_item = interaction[2]

#         user_embeddings, item_embeddings = self.forward()
#         user_e = user_embeddings[user, :]
#         pos_e = item_embeddings[pos_item, :]
#         #neg_e = self.get_item_embedding(neg_item)
#         neg_e = item_embeddings[neg_item, :]
#         pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        
#         input = torch.cat((torch.sigmoid(pos_item_score), torch.sigmoid(neg_item_score))).to(self.device)
#         target = torch.cat((torch.ones_like(pos_item_score), torch.zeros_like(neg_item_score))).to(self.device)
        
#         # mf_loss = self.loss(pos_item_score, neg_item_score)
#         # dice_loss = self.loss(input, target)

#         def drop_rate_schedule(idx):
#             drop_rate = np.linspace(0, 0.2, 300000)
#             if idx < 300000:
#                 return drop_rate[idx]
#             else:
#                 return 0.2

#         ce_loss =F.binary_cross_entropy_with_logits(input, target, reduce = False)
#         loss_mul = ce_loss * target # torch.cat((torch.ones_like(pos_item_score), -torch.ones_like(neg_item_score))).to(self.device)
#         ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
#         loss_sorted = ce_loss[ind_sorted]
        
#         # self.idx = self.idx + 1
#         # print(self.idx)
#         # print(drop_rate_schedule(idx))
#         remember_rate = 1 - drop_rate_schedule(self.idx) # 0.00001 # drop_rate_schedule(idx)
#         num_remember = int(remember_rate * len(loss_sorted))

#         ind_update = ind_sorted[:num_remember]

#         loss_update = torch.mean(F.binary_cross_entropy_with_logits(input[ind_update], target[ind_update]))

#         reg_loss = self.reg_loss(user_e, pos_e, neg_e)
#         loss = loss_update + self.reg_weight * reg_loss
#         return loss

#     def full_sort_predict(self, interaction):
#         user = interaction[0]
#         user_embeddings, item_embeddings = self.forward()
#         user_e = user_embeddings[user, :]
#         all_item_e = item_embeddings
#         score = torch.matmul(user_e, all_item_e.transpose(0, 1))
#         return score

# class MMBPR(GeneralRecommender):
#     r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
#     """
#     def __init__(self, config, dataloader):
#         super(MMBPR, self).__init__(config, dataloader)

#         # load parameters info
#         self.u_embedding_size = self.i_embedding_size = config['embedding_size']
#         self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

#         # define layers and loss
#         self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
#         self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))

#         if self.v_feat is not None and self.t_feat is not None:
#             self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1) 
#         elif self.t_feat is not None:
#             self.item_raw_features = self.t_feat
#         else:
#             self.item_raw_features = self.v_feat

#         self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
#         self.loss = BPRLoss() # DiceLoss()
#         self.reg_loss = EmbLoss()

#         # parameters initialization
#         self.apply(xavier_normal_initialization)

#     def get_user_embedding(self, user):
#         r""" Get a batch of user embedding tensor according to input user's id.

#         Args:
#             user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
#         """
#         return self.u_embedding[user, :]

#     def get_item_embedding(self, item):
#         r""" Get a batch of item embedding tensor according to input item's id.

#         Args:
#             item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
#         """
#         return self.item_embedding[item, :]

#     def forward(self, dropout=0.0):
#         item_embeddings = self.item_linear(self.item_raw_features)
#         item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

#         user_e = F.dropout(self.u_embedding, dropout)
#         item_e = F.dropout(item_embeddings, dropout)
#         return user_e, item_e

#     def calculate_loss(self, interaction, idx):
#         """
#         loss on one batch
#         :param interaction:
#             batch data format: tensor(3, batch_size)
#             [0]: user list; [1]: positive items; [2]: negative items
#         :return:
#         """
#         user = interaction[0]
#         pos_item = interaction[1]
#         neg_item = interaction[2]

#         user_embeddings, item_embeddings = self.forward()
#         user_e = user_embeddings[user, :]
#         pos_e = item_embeddings[pos_item, :]
#         #neg_e = self.get_item_embedding(neg_item)
#         neg_e = item_embeddings[neg_item, :]
#         pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        
#         input = torch.cat((torch.sigmoid(pos_item_score), torch.sigmoid(neg_item_score))).to(self.device)
#         target = torch.cat((torch.ones_like(pos_item_score), torch.zeros_like(neg_item_score))).to(self.device)
        
#         # mf_loss = self.loss(pos_item_score, neg_item_score)
#         # dice_loss = self.loss(input, target)

#         alpha = 0.2 
#         ce_loss =F.binary_cross_entropy_with_logits(input, target, reduce = False)
#         y_ = torch.sigmoid(input).detach()
#         weight = torch.pow(y_, alpha) * target + torch.pow((1-y_), alpha) * (1-target)
#         loss_ = ce_loss * weight
#         loss_ = torch.mean(loss_)

#         reg_loss = self.reg_loss(user_e, pos_e, neg_e)
#         loss = loss_ + self.reg_weight * reg_loss
#         return loss

#     def full_sort_predict(self, interaction):
#         user = interaction[0]
#         user_embeddings, item_embeddings = self.forward()
#         user_e = user_embeddings[user, :]
#         all_item_e = item_embeddings
#         score = torch.matmul(user_e, all_item_e.transpose(0, 1))
#         return score