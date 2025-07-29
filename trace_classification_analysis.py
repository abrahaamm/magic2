
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace数据集节点分类异常分析
找出分类错误、置信度低、或在嵌入空间中异常的节点
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from utils.loaddata import load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.config import build_args
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_test_data_for_classification(dataset_name='trace', device='cpu', max_samples=3000):
    """
    加载测试集数据用于节点分类分析
    """
    print(f"正在加载 {dataset_name} 测试集进行节点分类分析...")
    
    # 加载metadata
    metadata = load_metadata(dataset_name)
    n_test = metadata['n_test']
    
    # 构建模型参数
    args = build_args()
    args.dataset = dataset_name
    args.n_dim = metadata['node_feature_dim']
    args.e_dim = metadata['edge_feature_dim']
    args.num_hidden = 64
    args.num_layers = 3
    
    # 加载预训练的自编码器
    model = build_model(args)
    checkpoint_path = f"./checkpoints/checkpoint-{dataset_name}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 加载预训练的节点分类器
    from eval import NodeClassifier
    classifier = NodeClassifier(input_dim=args.num_hidden, num_classes=args.n_dim)
    classifier_path = f"./checkpoints/node_classifier-{dataset_name}.pt"
    
    try:
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print("已加载预训练的节点分类器")
    except FileNotFoundError:
        print("未找到预训练的节点分类器，将使用随机初始化的分类器")
    
    classifier = classifier.to(device)
    classifier.eval()
    
    embeddings = []
    true_labels = []
    pred_labels = []
    pred_probs = []
    node_indices = []
    graph_indices = []
    sample_count = 0
    
    with torch.no_grad():
        for i in range(n_test):
            if sample_count >= max_samples:
                break
                
            g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
            node_features = g.ndata['attr']
            true_node_labels = node_features.argmax(dim=1)
            
            # 获取节点嵌入
            node_embeddings = model.embed(g)
            
            # 获取分类预测
            logits = classifier(node_embeddings)
            probs = torch.softmax(logits, dim=1)
            pred_node_labels = torch.argmax(logits, dim=1)
            
            num_nodes = g.number_of_nodes()
            
            embeddings.append(node_embeddings.cpu().numpy())
            true_labels.extend(true_node_labels.cpu().numpy())
            pred_labels.extend(pred_node_labels.cpu().numpy())
            pred_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy())  # 最大概率作为置信度
            node_indices.extend(list(range(sample_count, sample_count + num_nodes)))
            graph_indices.extend([i] * num_nodes)
            
            sample_count += num_nodes
            del g, node_embeddings, logits, probs
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{n_test} 个测试图，累计节点数: {sample_count}")
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    print(f"总共加载了 {len(embeddings)} 个节点")
    print(f"节点类型分布:")
    
    # 节点类型名称
    node_type_names = {
        0: "SRCSINK_UNKNOWN", 1: "SUBJECT", 2: "FILE_OBJECT", 3: "MEMORY_OBJECT",
        4: "NETFLOW_OBJECT", 5: "PRINCIPAL", 6: "PROCESS", 7: "THREAD",
        8: "UNNAMED_PIPE_OBJECT", 9: "FILE_OBJECT_LINK", 10: "FILE_OBJECT_BLOCK"
    }
    
    for i in range(11):
        count = np.sum(np.array(true_labels) == i)
        type_name = node_type_names.get(i, f"Type_{i}")
        print(f"  {type_name}: {count} 个节点")
    
    return {
        'embeddings': embeddings,
        'true_labels': np.array(true_labels),
        'pred_labels': np.array(pred_labels),
        'pred_probs': np.array(pred_probs),
        'node_indices': np.array(node_indices),
        'graph_indices': np.array(graph_indices),
        'node_type_names': node_type_names
    }

def find_classification_anomalies(data):
    """
    找出分类异常的节点
    """
    print("正在分析分类异常...")
    
    embeddings = data['embeddings']
    true_labels = data['true_labels']
    pred_labels = data['pred_labels']
    pred_probs = data['pred_probs']
    
    # 1. 分类错误的节点
    misclassified = (true_labels != pred_labels)
    misclassified_indices = np.where(misclassified)[0]
    
    # 2. 低置信度预测的节点
    low_confidence_threshold = np.percentile(pred_probs, 10)  # 最低10%置信度
    low_confidence = pred_probs < low_confidence_threshold
    low_confidence_indices = np.where(low_confidence)[0]
    
    # 3. 在嵌入空间中的离群点（使用DBSCAN）
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(embeddings)
    outliers = (cluster_labels == -1)
    outlier_indices = np.where(outliers)[0]
    
    # 4. 每个类别内部的离群点
    intra_class_outliers = []
    for class_id in np.unique(true_labels):
        class_mask = (true_labels == class_id)
        if np.sum(class_mask) < 10:  # 类别样本太少，跳过
            continue
            
        class_embeddings = embeddings[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # 使用孤立森林检测类内离群点
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        class_outlier_labels = iso_forest.fit_predict(class_embeddings)
        class_outlier_mask = (class_outlier_labels == -1)
        
        intra_class_outliers.extend(class_indices[class_outlier_mask])
    
    intra_class_outliers = np.array(intra_class_outliers)
    
    # 5. 基于k近邻的异常检测
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # 计算平均k近邻距离
    mean_distances = np.mean(distances[:, 1:], axis=1)  # 排除自己
    distance_threshold = np.percentile(mean_distances, 95)  # 最远5%
    distant_nodes = mean_distances > distance_threshold
    distant_indices = np.where(distant_nodes)[0]
    
    anomaly_results = {
        'misclassified': misclassified_indices,
        'low_confidence': low_confidence_indices,
        'dbscan_outliers': outlier_indices,
        'intra_class_outliers': intra_class_outliers,
        'distant_neighbors': distant_indices,
        'low_confidence_threshold': low_confidence_threshold,
        'distance_threshold': distance_threshold
    }
    
    print(f"发现的异常类型:")
    print(f"  分类错误: {len(misclassified_indices)} 个节点")
    print(f"  低置信度: {len(low_confidence_indices)} 个节点")
    print(f"  DBSCAN离群点: {len(outlier_indices)} 个节点")
    print(f"  类内离群点: {len(intra_class_outliers)} 个节点")
    print(f"  k近邻距离异常: {len(distant_indices)} 个节点")
    
    return anomaly_results

def create_tsne_visualization(data, anomaly_results, save_path='trace_classification_analysis.png'):
    """
    创建TSNE可视化，标注各种异常点
    """
    print("正在生成TSNE可视化...")
    
    embeddings = data['embeddings']
    true_labels = data['true_labels']
    pred_labels = data['pred_labels']
    pred_probs = data['pred_probs']
    node_type_names = data['node_type_names']
    
    # 计算TSNE
    perplexity = min(30, len(embeddings) // 4)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trace节点分类异常分析 - TSNE可视化', fontsize=16)
    
    # 1. 真实标签分布
    scatter1 = axes[0, 0].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                  c=true_labels, cmap='tab10', alpha=0.6, s=1)
    axes[0, 0].set_title('真实节点类型分布')
    axes[0, 0].set_xlabel('TSNE-1')
    axes[0, 0].set_ylabel('TSNE-2')
    
    # 2. 预测标签分布
    scatter2 = axes[0, 1].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                  c=pred_labels, cmap='tab10', alpha=0.6, s=1)
    axes[0, 1].set_title('预测节点类型分布')
    axes[0, 1].set_xlabel('TSNE-1')
    axes[0, 1].set_ylabel('TSNE-2')
    
    # 3. 分类错误点
    error_colors = np.zeros(len(embeddings))
    error_colors[anomaly_results['misclassified']] = 1
    scatter3 = axes[0, 2].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                  c=error_colors, cmap='coolwarm', alpha=0.6, s=1)
    axes[0, 2].set_title(f'分类错误节点 ({len(anomaly_results["misclassified"])}个)')
    axes[0, 2].set_xlabel('TSNE-1')
    axes[0, 2].set_ylabel('TSNE-2')
    
    # 4. 预测置信度
    scatter4 = axes[1, 0].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                  c=pred_probs, cmap='viridis', alpha=0.6, s=1)
    axes[1, 0].set_title('预测置信度分布')
    axes[1, 0].set_xlabel('TSNE-1')
    axes[1, 0].set_ylabel('TSNE-2')
    plt.colorbar(scatter4, ax=axes[1, 0])
    
    # 5. DBSCAN离群点
    outlier_colors = np.zeros(len(embeddings))
    outlier_colors[anomaly_results['dbscan_outliers']] = 1
    scatter5 = axes[1, 1].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                  c=outlier_colors, cmap='coolwarm', alpha=0.6, s=1)
    axes[1, 1].set_title(f'DBSCAN离群点 ({len(anomaly_results["dbscan_outliers"])}个)')
    axes[1, 1].set_xlabel('TSNE-1')
    axes[1, 1].set_ylabel('TSNE-2')
    
    # 6. 综合异常分数
    combined_score = np.zeros(len(embeddings))
    for key in ['misclassified', 'low_confidence', 'dbscan_outliers', 'intra_class_outliers', 'distant_neighbors']:
        combined_score[anomaly_results[key]] += 1
    
    scatter6 = axes[1, 2].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                  c=combined_score, cmap='plasma', alpha=0.6, s=1)
    axes[1, 2].set_title('综合异常分数')
    axes[1, 2].set_xlabel('TSNE-1')
    axes[1, 2].set_ylabel('TSNE-2')
    plt.colorbar(scatter6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"TSNE可视化已保存到: {save_path}")
    
    return tsne_embeddings, combined_score

def save_anomaly_analysis(data, anomaly_results, tsne_embeddings, combined_score, 
                         save_path='trace_classification_anomalies.json'):
    """
    保存异常分析结果
    """
    print("正在保存异常分析结果...")
    
    # 收集所有异常节点
    all_anomaly_indices = set()
    for key in ['misclassified', 'low_confidence', 'dbscan_outliers', 
                'intra_class_outliers', 'distant_neighbors']:
        all_anomaly_indices.update(anomaly_results[key])
    
    anomaly_nodes = []
    for idx in all_anomaly_indices:
        node_info = {
            'node_index': int(data['node_indices'][idx]),
            'graph_index': int(data['graph_indices'][idx]),
            'true_label': int(data['true_labels'][idx]),
            'pred_label': int(data['pred_labels'][idx]),
            'pred_confidence': float(data['pred_probs'][idx]),
            'true_type_name': data['node_type_names'][data['true_labels'][idx]],
            'pred_type_name': data['node_type_names'][data['pred_labels'][idx]],
            'is_misclassified': idx in anomaly_results['misclassified'],
            'is_low_confidence': idx in anomaly_results['low_confidence'],
            'is_dbscan_outlier': idx in anomaly_results['dbscan_outliers'],
            'is_intra_class_outlier': idx in anomaly_results['intra_class_outliers'],
            'is_distant_neighbor': idx in anomaly_results['distant_neighbors'],
            'combined_anomaly_score': int(combined_score[idx]),
            'tsne_x': float(tsne_embeddings[idx, 0]),
            'tsne_y': float(tsne_embeddings[idx, 1])
        }
        anomaly_nodes.append(node_info)
    
    # 按综合异常分数排序
    anomaly_nodes.sort(key=lambda x: x['combined_anomaly_score'], reverse=True)
    
    # 统计信息
    stats = {
        'total_nodes': len(data['embeddings']),
        'misclassified_nodes': len(anomaly_results['misclassified']),
        'low_confidence_nodes': len(anomaly_results['low_confidence']),
        'dbscan_outliers': len(anomaly_results['dbscan_outliers']),
        'intra_class_outliers': len(anomaly_results['intra_class_outliers']),
        'distant_neighbors': len(anomaly_results['distant_neighbors']),
        'total_anomaly_nodes': len(anomaly_nodes),
        'classification_accuracy': float(np.mean(data['true_labels'] == data['pred_labels'])),
        'average_confidence': float(np.mean(data['pred_probs'])),
        'low_confidence_threshold': float(anomaly_results['low_confidence_threshold']),
        'distance_threshold': float(anomaly_results['distance_threshold'])
    }
    
    # 类型分布统计
    type_stats = {}
    for i in range(11):
        type_name = data['node_type_names'][i]
        true_count = np.sum(data['true_labels'] == i)
        pred_count = np.sum(data['pred_labels'] == i)
        type_stats[type_name] = {
            'true_count': int(true_count),
            'pred_count': int(pred_count)
        }
    
    results = {
        'analysis_summary': {
            'task_type': '节点分类异常分析',
            'dataset': 'trace',
            'description': '分析节点分类任务中的各种异常：分类错误、低置信度、聚类离群等'
        },
        'statistics': stats,
        'type_distribution': type_stats,
        'anomaly_detection_methods': {
            'misclassified': '分类错误的节点',
            'low_confidence': '预测置信度低的节点',
            'dbscan_outliers': 'DBSCAN聚类中的离群点',
            'intra_class_outliers': '类内使用孤立森林检测的离群点',
            'distant_neighbors': 'k近邻距离异常的节点'
        },
        'anomaly_nodes': anomaly_nodes[:500]  # 只保存前500个最异常的节点
    }
    
    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"异常分析结果已保存到: {save_path}")
    print(f"发现 {len(anomaly_nodes)} 个异常节点")
    print(f"分类准确率: {stats['classification_accuracy']:.2%}")
    print(f"平均预测置信度: {stats['average_confidence']:.3f}")
    
    return results

def main():
    """
    主函数
    """
    print("开始Trace节点分类异常分析...")
    
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载测试数据
        data = load_test_data_for_classification(device=device, max_samples=3000)
        
        # 2. 分析分类异常
        anomaly_results = find_classification_anomalies(data)
        
        # 3. 创建TSNE可视化
        tsne_embeddings, combined_score = create_tsne_visualization(data, anomaly_results)
        
        # 4. 保存结果
        results = save_anomaly_analysis(data, anomaly_results, tsne_embeddings, combined_score)
        
        print("\n节点分类异常分析完成！")
        print("生成的文件:")
        print("- trace_classification_analysis.png: TSNE可视化图")
        print("- trace_classification_anomalies.json: 详细的异常分析结果")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()