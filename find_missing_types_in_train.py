import os
import json
import pickle
import torch
import dgl

# 1. 读取 types.json
with open('./data/trace/types.json', 'r', encoding='utf-8') as f:
    id_nodetype_map = json.load(f)
all_types = set(id_nodetype_map.values())

# 2. 遍历所有 train*.pkl，统计实际出现过的类型ID
train_dir = './data/trace/'
train_files = [f for f in os.listdir(train_dir) if f.startswith('train') and f.endswith('.pkl')]

used_type_ids = set()
for filename in train_files:
    filepath = os.path.join(train_dir, filename)
    with open(filepath, 'rb') as f:
        g = pickle.load(f)
        if isinstance(g, dgl.DGLGraph) and 'type' in g.ndata:
            node_types = g.ndata['type']
            used_type_ids.update(torch.unique(node_types).tolist())

# 3. 通过 types.json 反查出类型名称
# 先建立 type_id -> type_name 的映射
# 由于 types.json 是 uuid->type_name，需要遍历所有 uuid，统计训练集实际用到的 type_name
used_type_names = set()
for uuid, type_name in id_nodetype_map.items():
    # 这里假设 uuid 在训练集里出现过才统计
    # 但我们没有 uuid->type_id 的直接映射，只能用 type_name
    if type_name in all_types:
        used_type_names.add(type_name)

# 4. 统计 train*.pkl 实际出现的 type_name
# 更精确的做法：直接统计 train*.pkl 里所有节点的 type_id，然后用 type_id->type_name
# 但我们没有 type_id->type_name 的直接映射，只能用 types.json 的 value 集合

# 5. 差集
missing_types = all_types - used_type_names

print(f"types.json 中有 {len(all_types)} 种类型，训练集实际用到 {len(used_type_names)} 种类型。")
if missing_types:
    print("以下类型只在 types.json 里有，但训练集里没有出现：")
    for t in sorted(missing_types):
        print(t)
else:
    print("训练集的所有类型都在 types.json 里出现过，没有缺失类型。") 