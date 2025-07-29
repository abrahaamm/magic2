import os
import pickle
import torch
import dgl
import sys

# 将父目录添加到Python路径，以便可以导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def count_node_types_in_pkl():
    """
    遍历 data/trace/ 目录下的所有 train*.pkl 文件，
    统计并打印出所有唯一的节点类型ID。
    """
    data_dir = './data/trace/'
    if not os.path.exists(data_dir):
        print(f"错误: 目录 '{data_dir}' 不存在。")
        return

    train_files = [f for f in os.listdir(data_dir) if f.startswith('train') and f.endswith('.pkl')]
    
    if not train_files:
        print(f"在 '{data_dir}' 中没有找到 'train*.pkl' 文件。")
        print("请确保数据已正确生成。")
        return

    all_node_types = set()
    print(f"正在扫描 {len(train_files)} 个训练文件...")

    for filename in train_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                # pkl 文件中直接存储了DGL图对象
                graph = pickle.load(f)
                
                if isinstance(graph, dgl.DGLGraph) and 'type' in graph.ndata:
                    # 从节点的 'type' 属性中提取类型ID
                    node_types_tensor = graph.ndata['type']
                    unique_types_in_graph = torch.unique(node_types_tensor).tolist()
                    all_node_types.update(unique_types_in_graph)
                else:
                    print(f"警告: 文件 '{filename}' 的格式不符合预期，已跳过。")

        except Exception as e:
            print(f"读取或处理文件 '{filename}' 时出错: {e}")

    if not all_node_types:
        print("未能从任何文件中提取出节点类型。")
        return

    print("\n" + "="*40)
    print("完成统计。")
    print(f"在所有训练PKL文件中，总共发现了 {len(all_node_types)} 种不同的节点类型。")
    
    # 排序后打印，方便查看
    sorted_types = sorted(list(all_node_types))
    print("所有唯一的节点类型ID如下:")
    print(sorted_types)
    print("="*40)


if __name__ == '__main__':
    count_node_types_in_pkl() 