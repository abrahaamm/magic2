#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import networkx as nx
import json
import argparse

def load_and_stat_graphs(dataset):
    """
    从pkl文件加载图数据并分别统计训练集和测试集的边分布
    """
    print(f"正在加载 {dataset} 数据集...")
    
    # 加载预处理好的图数据
    train_graphs_data = pkl.load(open(f'./data/{dataset}/train.pkl', 'rb'))
    test_graphs_data = pkl.load(open(f'./data/{dataset}/test.pkl', 'rb'))
    
    # 加载节点类型映射
    with open(f'./data/{dataset}/node_type_mapping.json', 'r', encoding='utf-8') as f:
        id_to_type = json.load(f)
        # 将键转换为整数
        id_to_type = {int(k): v for k, v in id_to_type.items()}
    
    # 将图数据转换为NetworkX图对象
    train_graphs = [nx.node_link_graph(g_data) for g_data in train_graphs_data]
    test_graphs = [nx.node_link_graph(g_data) for g_data in test_graphs_data]
    
    def stat_edge_distribution(graphs, graph_type):
        """统计边的源节点和目标节点类型分布"""
        print(f"\n=== {graph_type} 边的节点分类统计 ===")
        print("=" * 50)
        
        sbj_type_counts = {}
        obj_type_counts = {}
        total_edges = 0
        
        for g in graphs:
            for edge in g.edges():
                src_node_id, dst_node_id = edge
                src_type_id = g.nodes[src_node_id]['type']
                dst_type_id = g.nodes[dst_node_id]['type']
                
                # 将type_id转换回类型名称
                src_type_name = id_to_type[src_type_id]
                dst_type_name = id_to_type[dst_type_id]
                
                # 统计源节点类型（sbj+）
                sbj_type_counts[src_type_name] = sbj_type_counts.get(src_type_name, 0) + 1
                # 统计目标节点类型（obj+）
                obj_type_counts[dst_type_name] = obj_type_counts.get(dst_type_name, 0) + 1
                total_edges += 1
        
        print(f"总边数: {total_edges}")
        print()
        
        # 计算并输出源节点（sbj+）统计
        sbj_total = sum(sbj_type_counts.values())
        print("【sbj+】种类及占比：")
        for type_name in sorted(sbj_type_counts.keys()):
            count = sbj_type_counts[type_name]
            percentage = (count / sbj_total) * 100
            print(f"{type_name}: {count} ({percentage:.2f}%)")
        
        print()
        
        # 计算并输出目标节点（obj+）统计
        obj_total = sum(obj_type_counts.values())
        print("【obj+】种类及占比：")
        for type_name in sorted(obj_type_counts.keys()):
            count = obj_type_counts[type_name]
            percentage = (count / obj_total) * 100
            print(f"{type_name}: {count} ({percentage:.2f}%)")
        
        return sbj_type_counts, obj_type_counts, total_edges
    
    # 分别统计训练集和测试集
    print(f"\n正在统计 {dataset} 数据集的边分布...")
    train_sbj, train_obj, train_edges = stat_edge_distribution(train_graphs, "训练集")
    test_sbj, test_obj, test_edges = stat_edge_distribution(test_graphs, "测试集")
    
    # 对比总结
    print("\n=== 总结对比 ===")
    print("=" * 50)
    print(f"训练集总边数: {train_edges}")
    print(f"测试集总边数: {test_edges}")
    print(f"总边数: {train_edges + test_edges}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统计已预处理图数据的边分布')
    parser.add_argument("--dataset", type=str, default="trace", help="数据集名称")
    args = parser.parse_args()
    
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError(f"不支持的数据集: {args.dataset}")
    
    load_and_stat_graphs(args.dataset) 