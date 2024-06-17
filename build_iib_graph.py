import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import pandas as pd
import os
import yaml
import argparse
import networkx as nx


def gen_item_matrix(all_edge, no_items):
    edge_dict = defaultdict(set)

    for edge in all_edge:
        user, item = edge
        edge_dict[item].add(user)

    min_item = 0             # 0
    num_item = no_items      # in our case, items/items ids start from 1
    item_graph_matrix = torch.zeros(num_item, num_item)
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head+1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            # print(len(item_head.intersection(item_rear)))
            inter_len = len(item_head.intersection(item_rear))
            if inter_len >= 2:
                item_graph_matrix[head_key-min_item][rear_key-min_item] = inter_len
                item_graph_matrix[rear_key-min_item][head_key-min_item] = inter_len
    bar.close()

    return item_graph_matrix


if __name__ == 	'__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='games', help='name of dataset')
    parser.add_argument('--topk', type=int, default='10', help='top k')
    args = parser.parse_args()
    dataset_name = args.dataset
    top_k = args.topk
    
    print(f'Generating i-i matrix for {dataset_name} ...\n')

    config = {}
    os.chdir('src_tiktok')
    cur_dir = os.getcwd()
    con_dir = os.path.join(cur_dir, 'configs') # get config dir
    overall_config_file = os.path.join(con_dir, "overall.yaml")
    dataset_config_file = os.path.join(con_dir, "dataset", "{}.yaml".format(dataset_name))
    conf_files = [overall_config_file, dataset_config_file]
    # load configs
    for file in conf_files:
        if os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                tmp_d = yaml.safe_load(f)
                config.update(tmp_d)

    dataset_path = os.path.abspath(config['data_path'] + dataset_name)
    print('data path:\t', dataset_path)
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    print(config['inter_file_name'])
    train_df = pd.read_csv(os.path.join(dataset_path, config['inter_file_name']), sep='\t')
    num_item = len(pd.unique(train_df[iid_field]))
    train_df = train_df[train_df['x_label'] == 0].copy()
    train_data = train_df[[uid_field, iid_field]].to_numpy()
    # item_item_pairs =[]
    item_graph_matrix = gen_item_matrix(train_data, num_item)
    #############################generate item-item matrix
    # pdb.set_trace()
    item_graph = item_graph_matrix.numpy()
    # np.save(os.path.join(dataset_path, 'item_graph.npy'), item_graph, allow_pickle=True) 
    G = nx.from_numpy_array(item_graph)    
    pagerank_list = nx.pagerank(G, alpha=1)
    # np.save(os.path.join(dataset_path, 'pagerank.npy'), pagerank_list, allow_pickle=True) 

    item_num = torch.zeros(num_item)
    for i in range(num_item):
        item_num[i] = len(torch.nonzero(item_graph_matrix[i]))
        # print("this is ", i, "num", item_num[i])
    edge_list_i = []
    edge_list_j = []
    item_graph_dict = {}

    for i in range(num_item):
        if item_num[i] <= top_k:
            item_i = torch.topk(item_graph_matrix[i],int(item_num[i]))
            edge_list_i =item_i.indices.numpy().tolist()
            edge_list_j =item_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            item_graph_dict[i] = edge_list
        else:
            item_i = torch.topk(item_graph_matrix[i], top_k)
            edge_list_i = item_i.indices.numpy().tolist()
            edge_list_j = item_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            item_graph_dict[i] = edge_list
    name = 'item_graph_dict_' + str(top_k) + '.npy'
    np.save(os.path.join(dataset_path, name), item_graph_dict, allow_pickle=True) 


