import json
import os

types_path = './data/trace/types.json'

if not os.path.exists(types_path):
    print(f'未找到 {types_path}，请确认文件已放置在正确位置。')
    exit(1)

with open(types_path, 'r', encoding='utf-8') as f:
    id_nodetype_map = json.load(f)

# 统计所有唯一的节点类型名称
all_types = set(id_nodetype_map.values())

print(f'在 types.json 中共发现 {len(all_types)} 种不同的节点类型：')
for t in sorted(all_types):
    print(t) 