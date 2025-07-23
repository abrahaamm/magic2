import os
import json
import pickle
import torch
import dgl
from collections import Counter

# 读取 type_id <-> 类型名称 映射
with open('./data/trace/node_type_mapping.json', 'r', encoding='utf-8') as f:
    type_name_to_id = json.load(f)
# 反转为 id -> name
id_to_type_name = {v: k for k, v in type_name_to_id.items()}

# 统计每个 type_id 在测试集中的出现次数
test_dir = './data/trace/'
test_files = [f for f in os.listdir(test_dir) if f.startswith('test') and f.endswith('.pkl')]

counter = Counter()
for filename in test_files:
    filepath = os.path.join(test_dir, filename)
    with open(filepath, 'rb') as f:
        g = pickle.load(f)
        if isinstance(g, dgl.DGLGraph) and 'type' in g.ndata:
            node_types = g.ndata['type']
            counter.update(torch.unique(node_types, return_counts=True)[0].tolist() if node_types.numel() else [])
            # 更精确地统计每个节点类型
            for t in node_types.tolist():
                counter[t] += 1

print('测试集中每种类型的节点数量：')
for type_id in sorted(id_to_type_name.keys()):
    type_name = id_to_type_name[type_id]
    count = counter[type_id]
    print(f'{type_name} (type_id={type_id}): {count}') 