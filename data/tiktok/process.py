import json
import pandas as pd

with open('train.json', 'r') as file:
    data1 = json.load(file)
with open('test.json', 'r') as file:
    data2 = json.load(file)
with open('val.json', 'r') as file:
    data3 = json.load(file)

df1 = pd.DataFrame(data1.items(), columns=['userID', 'itemID'])
df1 = df1.explode('itemID', ignore_index=True)
df1['x_label'] = 0

df2 = pd.DataFrame(data2.items(), columns=['userID', 'itemID'])
df2 = df2.explode('itemID', ignore_index=True)
df2['x_label'] = 1

df3 = pd.DataFrame(data3.items(), columns=['userID', 'itemID'])
df3 = df3.explode('itemID', ignore_index=True)
df3['x_label'] = 2


df = pd.concat([df1, df2, df3], ignore_index=True).dropna().sort_values('userID')
df.to_csv('tiktok.inter', sep='\t', index=False)

