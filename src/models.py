# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import math
import os
import pickle
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
import faiss
from kmeans_pytorch import kmeans
from modules import Encoder, LayerNorm


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class KMeans_Pytorch(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 10
        self.first_batch = True
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id
        self.device = device
        print(self.device, "-----")

    def run_kmeans(self, x, Niter=20, tqdm_flag=False):
        if x.shape[0] >= self.num_cluster:
            seq2cluster, centroids = kmeans(
                X=x, num_clusters=self.num_cluster, distance="euclidean", device=self.device, tqdm_flag=False
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        # last batch where
        else:
            seq2cluster, centroids = kmeans(
                X=x, num_clusters=x.shape[0] - 1, distance="euclidean", device=self.device, tqdm_flag=False
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        return seq2cluster, centroids


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  #item size , item_num+2 ???? 这个要重新看下
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.intention_cluster = nn.Embedding(int(args.num_intent_clusters), args.hidden_size)


        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.device = args.device

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)
        self.load_item_embeddings(args.item_emb_path)

    def load_item_embeddings(self, item_emb_path):
        if item_emb_path is not None:
            temp_tensor = torch.nn.Embedding.from_pretrained(torch.load(item_emb_path)).weight.to(self.device)
            temp_tensor_norm = torch.linalg.norm(temp_tensor, dim=1)
            temp_tensor /= temp_tensor_norm.unsqueeze(1)

            with torch.no_grad():
                self.item_embeddings.weight[1:,:] = temp_tensor
        else:
            print("item embedding path is incorrect!")
            exit(-9)

    # Positional Embedding
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output




    def next_item_prediction(self, log_seqs, interacted, target_item, rec2inf=False, k=50, alpha=0):
        final_feat = self.forward(log_seqs)[:, -1, :]
        item_embs = self.item_embeddings.weight
        if not rec2inf:
            item_similarity = torch.matmul(item_embs[target_item], item_embs.T)
            logits = torch.matmul(final_feat, item_embs.T) + alpha * item_similarity
            logits[interacted[:, 1:] == 1] = float('-inf')
            _, next_items = logits.max(dim=1)
        else:
            logits = torch.matmul(final_feat, item_embs.T)
            logits[interacted[:, 1:] == 1] = float('-inf')
            _, candidate_items = torch.topk(logits, k, dim=1)
            candidate_embeddings = item_embs[candidate_items]
            candidate_similarity = torch.matmul(candidate_embeddings, item_embs[target_item])
            _, next_items_index = candidate_similarity.max(dim=1)
            next_items = candidate_items[torch.arange(len(log_seqs)), next_items_index]
        return next_items  # item_id start with 0

    def next_item_prediction_with_IPG(self, log_seqs, interacted, target_item):
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]

        item_embs = self.item_embeddings(torch.LongTensor(range(2, self.args.item_size)).to(self.device))
        user_item_score = torch.sigmoid(torch.matmul(final_feat, item_embs.T))

        item_target_score = torch.cosine_similarity(item_embs, item_embs[target_item])
        user_target_score = torch.cosine_similarity(final_feat, item_embs[target_item]).unsqueeze(1)
        diff = item_target_score - user_target_score
        scores = user_item_score * diff

        scores[interacted[:, 1:] == 1] = float('-inf')
        _, next_items = scores.max(dim=1)
        return next_items  # item_id start with 0


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module,  (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def init_weights_original(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




# if __name__ == "__main__":
#     onlineitemsim = OnlineItemSimilarity(item_size=10)
#     item_embeddings = nn.Embedding(10, 6, padding_idx=0)
#     onlineitemsim.update_embedding_matrix(item_embeddings)
#     item_idx = torch.tensor(2, dtype=torch.long)
#     similiar_items = onlineitemsim.most_similar(item_idx=item_idx, top_k=1)
#     print(similiar_items)
