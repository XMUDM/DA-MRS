import numpy as np
import pandas as pd

seed = 999
np.random.seed(seed)

uid_field = 'userID'
iid_field = 'itemID'
split = 'x_label'

names = ['baby', 'sports', 'clothing']
cols = [uid_field, iid_field, split]
ratio = 0.2


for name in names:
    df = pd.read_csv('./data/' + name + '/'+ name + '.inter', usecols=cols, sep="\t")

    item_num = int(max(df[iid_field].values)) + 1
    user_num = int(max(df[uid_field].values)) + 1
    item_num, user_num


    dfs = []
    # splitting into training/validation/test
    for i in range(3):
        temp_df = df[df[split] == i].copy()
        dfs.append(temp_df)
    # filtering out new users in val/test sets
    train_u = set(dfs[0][uid_field].values)
    for i in [1, 2]:
        dropped_inter = pd.Series(True, index=dfs[i].index)
        dropped_inter ^= dfs[i][uid_field].isin(train_u)
        dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

    train_df, valid_df, test_df = dfs

    def get_u_i_dict(dataset):
        data_dict = dict()
        uid_freq = dataset.groupby('userID')['itemID']
        for u, u_ls in uid_freq:
            data_dict[u] = set(u_ls.values)
        return data_dict

    train_dict = get_u_i_dict(train_df)
    valid_dict = get_u_i_dict(valid_df)
    test_dict = get_u_i_dict(test_df)

    num_train = train_df.shape[0]
    count = 0
    while count < num_train * ratio:
        u_id = np.random.randint(user_num)
        i_id = np.random.randint(item_num)
        if i_id not in train_dict[u_id]:
            if ((u_id not in test_dict) and (u_id not in valid_dict)):
                train_dict[u_id].add(i_id)
                df = df.append({uid_field: u_id, iid_field: i_id, split:0}, ignore_index=True)
                count += 1
            else:
                if ((i_id not in test_dict[u_id]) and (i_id not in valid_dict[u_id])):
                    train_dict[u_id].add(i_id)
                    df = df.append({uid_field: u_id, iid_field: i_id, split:0}, ignore_index=True)
                    count += 1

    df.to_csv('./data/' + name + '/' + name + '_add_' + str(ratio) + '.inter', sep='\t', index=False)
