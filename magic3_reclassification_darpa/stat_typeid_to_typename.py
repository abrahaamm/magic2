import os
import json
import pickle
import torch
import dgl
from collections import defaultdict

# 读取 types.json
with open('./data/trace/types.json', 'r', encoding='utf-8') as f:
    id_nodetype_map = json.load(f)  # uuid -> type_name

train_dir = './data/trace/'
train_files = [f for f in os.listdir(train_dir) if f.startswith('train') and f.endswith('.pkl')]

# type_id -> set of type_name
typeid_to_typename = defaultdict(set)

for filename in train_files:
    filepath = os.path.join(train_dir, filename)
    with open(filepath, 'rb') as f:
        g = pickle.load(f)
        if isinstance(g, dgl.DGLGraph) and 'type' in g.ndata and 'id' in g.ndata:
            node_types = g.ndata['type'].tolist()
            node_uuids = g.ndata['id']
            if torch.is_tensor(node_uuids):
                node_uuids = node_uuids.tolist()
            for tidx, uuid in enumerate(node_uuids):
                type_id = node_types[tidx]
                uuid_str = str(uuid)
                type_name = id_nodetype_map.get(uuid_str, None)
                if type_name is not None:
                    typeid_to_typename[type_id].add(type_name)

print('type_id <-> type_name 映射表:')
for type_id in sorted(typeid_to_typename.keys()):
    names = sorted(typeid_to_typename[type_id])
    print(f'{type_id}: {names}') 