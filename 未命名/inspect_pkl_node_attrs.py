import os
import pickle
import dgl

train_dir = './data/trace/'
train_files = [f for f in os.listdir(train_dir) if f.startswith('train') and f.endswith('.pkl')]

if not train_files:
    print('未找到 train*.pkl 文件')
    exit(1)

filepath = os.path.join(train_dir, train_files[0])
with open(filepath, 'rb') as f:
    g = pickle.load(f)
    print(f'文件: {train_files[0]}')
    print('节点属性名:', list(g.ndata.keys()))
    print('每个属性的数据类型:')
    for k in g.ndata.keys():
        print(f'  {k}: {type(g.ndata[k])}') 