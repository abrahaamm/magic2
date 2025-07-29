import torch
import torch.nn as nn
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import json
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')

# DARPA 风格分类常量
PLUS = '>>>>'

def extract_type_from_darpa_format(darpa_entity):
    """
    从 DARPA 格式的实体字符串中提取类型
    例如：'PROCESS>>>>bash' -> 'PROCESS'
    """
    if darpa_entity and PLUS in darpa_entity:
        return darpa_entity.split(PLUS)[0]
    return darpa_entity


# ------------------ 节点分类评估 (trace 数据集) ------------------


class NodeClassifier(nn.Module):
    """与 train.py 中相同的简单节点分类器"""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)
    dataset_name = main_args.dataset
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    set_random_seed(0)

    if dataset_name == 'trace':
        # ---------------- 节点分类评估 ----------------
        encoder_path = f"./checkpoints/checkpoint-{dataset_name}-encoder.pt"
        classifier_path = f"./checkpoints/checkpoint-{dataset_name}-classifier.pt"

        if not (os.path.exists(encoder_path) and os.path.exists(classifier_path)):
            print("未找到节点分类模型，请先运行训练脚本： python train.py --dataset trace")
            return

        metadata = load_metadata(dataset_name)
        n_dim = metadata['node_feature_dim']
        e_dim = metadata['edge_feature_dim']

        # 动态从 node_type_mapping.json 加载节点类型名称映射
        try:
            with open(f'./data/{dataset_name}/node_type_mapping.json', 'r') as f:
                node_type_mapping = json.load(f)
                # JSON加载的键是字符串，需要转回整数
                node_type_names = {int(k): v for k, v in node_type_mapping.items()}
                num_classes = len(node_type_names) # <-- 关键修改
            print(f"成功从文件加载节点类型映射，共 {num_classes} 个类别。")
        except FileNotFoundError:
            print(f"错误: 未找到 'node_type_mapping.json'。请先运行 trace_parser.py。")
            return

        # 构建并加载编码器
        args_clone = main_args
        args_clone.n_dim = n_dim
        args_clone.e_dim = e_dim
        args_clone.num_hidden = 64  # 修改为64
        args_clone.num_layers = 3   # 修改为3
        encoder = build_model(args_clone).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.eval()

        # 构建并加载分类器
        classifier = NodeClassifier(encoder.output_hidden_dim, num_classes).to(device) # <-- 使用动态 num_classes
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()

        all_preds, all_labels, all_embeddings = [], [], []
        total_nodes = 0
        n_test = metadata['n_test']

        print("\n" + "="*80)
        print(f"评估测试集中的 {n_test} 个图...")
        print("="*80 + "\n")
        
        with torch.no_grad():
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                node_labels = g.ndata['attr'].argmax(dim=1)
                
                # 获取节点嵌入和预测
                node_embeddings = encoder.embed(g)
                node_logits = classifier(node_embeddings)
                node_preds = torch.argmax(node_logits, dim=1)
                
                all_preds.extend(node_preds.cpu().numpy())
                all_labels.extend(node_labels.cpu().numpy())
                all_embeddings.append(node_embeddings.cpu().numpy())
                total_nodes += g.number_of_nodes()
                
                del g, node_embeddings, node_logits
                torch.cuda.empty_cache() if device != "cpu" else None

        # 计算总体指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        print("总体性能指标:")
        print("-" * 40)
        print(f"总体准确率: {accuracy:.2%}")
        print(f"宏平均F1分数: {f1_macro:.2%}")
        print("\n")
        
        # 获取详细的分类报告
        class_names = [node_type_names.get(i, f"类型_{i}") for i in range(num_classes)] # <-- 使用动态 num_classes
        
        # 计算每个类别的样本数量和比例
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        label_percentages = (label_counts / total_nodes) * 100
        
        # 获取分类报告
        report_dict = classification_report(
            all_labels, 
            all_preds,
            target_names=class_names,
            zero_division=0,
            digits=4,
            output_dict=True
        )
        
        # 打印详细评估表格
        print("各类别详细评估指标:")
        print("-" * 120)
        print(f"{'类别名称':<20} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>12} {'样本占比':>10}")
        print("-" * 120)
        
        for label in unique_labels:
            class_name = node_type_names.get(label, f"类型_{label}")
            metrics = report_dict[class_name]
            count = label_counts[label == unique_labels][0]
            percentage = label_percentages[label == unique_labels][0]
            
            print(f"{class_name:<20} {metrics['precision']:>10.2%} {metrics['recall']:>10.2%} "
                  f"{metrics['f1-score']:>10.2%} {count:>12,d} {percentage:>9.2f}%")
        
        print("-" * 120)
        print(f"{'加权平均':<20} "
              f"{report_dict['weighted avg']['precision']:>10.2%} "
              f"{report_dict['weighted avg']['recall']:>10.2%} "
              f"{report_dict['weighted avg']['f1-score']:>10.2%} "
              f"{total_nodes:>12,d} {100:>9.2f}%")
        print("=" * 120)

        # ------------------ DARPA 风格分类统计 ------------------
        print("\n正在进行 DARPA 风格分类统计...")
        
        # 统计主体和对象类型
        # 注意：在节点分类任务中，我们无法直接区分主体和对象
        # 因此我们将所有节点按类型进行统计
        
        print("\n节点类型分布统计:")
        print("=" * 60)
        
        darpa_type_counts = {}
        total_nodes = len(all_labels)
        
        for label_id, pred_id in zip(all_labels, all_preds):
            type_name = node_type_names.get(label_id, f"UNKNOWN_{label_id}")
            if type_name not in darpa_type_counts:
                darpa_type_counts[type_name] = {'true': 0, 'pred': 0}
            darpa_type_counts[type_name]['true'] += 1
            
            pred_type_name = node_type_names.get(pred_id, f"UNKNOWN_{pred_id}")
            if pred_type_name not in darpa_type_counts:
                darpa_type_counts[pred_type_name] = {'true': 0, 'pred': 0}
            darpa_type_counts[pred_type_name]['pred'] += 1
        
        print(f"{'类型名称':<15} {'真实数量':<12} {'比例':<10} {'预测数量':<12} {'比例':<10}")
        print("-" * 60)
        
        for type_name in sorted(darpa_type_counts.keys()):
            true_count = darpa_type_counts[type_name]['true']
            pred_count = darpa_type_counts[type_name]['pred']
            true_ratio = (true_count / total_nodes) * 100
            pred_ratio = (pred_count / total_nodes) * 100
            print(f"{type_name:<15} {true_count:<12,d} {true_ratio:<9.2f}% {pred_count:<12,d} {pred_ratio:<9.2f}%")
        
        print("=" * 60)

        # ------------------ t-SNE 可视化 ------------------
        print("\n正在进行 t-SNE 降维和可视化...")
        start_time = time.time()

        # 将嵌入和标签列表转换为Numpy数组
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds) # 新增：转换预测结果

        # 找到分类错误的节点
        misclassified_indices = np.where(all_labels != all_preds)[0]
        misclassified_labels = all_labels[misclassified_indices]
        misclassified_preds = all_preds[misclassified_indices]

        # 保存分类错误的节点信息
        misclassified_output_path = './misclassified_nodes.txt'
        with open(misclassified_output_path, 'w') as f:
            f.write("Misclassified Node Information\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Node Index':<15} {'True Label':<25} {'Predicted Label':<25}\n")
            f.write("-" * 50 + "\n")
            for i, idx in enumerate(misclassified_indices):
                true_label_name = node_type_names.get(misclassified_labels[i], f"类型_{misclassified_labels[i]}")
                pred_label_name = node_type_names.get(misclassified_preds[i], f"类型_{misclassified_preds[i]}")
                f.write(f"{idx:<15} {true_label_name:<25} {pred_label_name:<25}\n")
        print(f"分类错误的节点信息已保存到: {misclassified_output_path}")


        # 对于新的分类体系，我们暂时不排除任何类别，先观察所有类别的表现
        labels_to_exclude = []  # 新分类体系中暂不排除任何类别
        exclude_names = [node_type_names.get(l) for l in labels_to_exclude]
        if exclude_names:
            print(f"\n从可视化中剔除以下类别: {exclude_names}")
        else:
            print(f"\n使用所有 {num_classes} 个类别进行可视化")
        
        mask = ~np.isin(all_labels, labels_to_exclude)
        embeddings_after_filter = all_embeddings[mask]
        labels_after_filter = all_labels[mask]
        # 同样需要过滤错误分类的索引
        preds_after_filter = all_preds[mask]
        
        nodes_after_filter = len(labels_after_filter)
        print(f"剔除后剩余节点数: {nodes_after_filter}")

        # 找出过滤后的错误分类节点
        misclassified_mask_after_filter = labels_after_filter != preds_after_filter
        
        # 为了性能，如果节点数太多，则随机抽样一部分进行可视化
        sample_size = 10000
        if nodes_after_filter > sample_size:
            print(f"节点总数 ({nodes_after_filter}) 较多，将随机抽取 {sample_size} 个节点进行可视化...")
            # 确保抽样时同时包含正确和错误分类的节点，以保持代表性
            correctly_classified_indices = np.where(~misclassified_mask_after_filter)[0]
            misclassified_indices_after_filter = np.where(misclassified_mask_after_filter)[0]
            
            num_misclassified_to_sample = int(sample_size * (len(misclassified_indices_after_filter) / nodes_after_filter))
            num_correct_to_sample = sample_size - num_misclassified_to_sample
            
            # 安全地选择样本
            selected_misclassified_indices = np.random.choice(
                misclassified_indices_after_filter, 
                min(num_misclassified_to_sample, len(misclassified_indices_after_filter)), 
                replace=False
            )
            selected_correct_indices = np.random.choice(
                correctly_classified_indices, 
                min(num_correct_to_sample, len(correctly_classified_indices)), 
                replace=False
            )
            
            indices = np.concatenate([selected_correct_indices, selected_misclassified_indices])
            np.random.shuffle(indices) # 随机打乱索引

            embeddings_for_tsne = embeddings_after_filter[indices]
            labels_for_tsne = labels_after_filter[indices]
            misclassified_for_tsne = misclassified_mask_after_filter[indices]
        else:
            print(f"使用全部 {nodes_after_filter} 个节点进行可视化...")
            embeddings_for_tsne = embeddings_after_filter
            labels_for_tsne = labels_after_filter
            misclassified_for_tsne = misclassified_mask_after_filter

        # 执行 t-SNE
        # 增加迭代次数和早期夸大系数以获得更好的簇分离效果
        tsne = TSNE(
            n_components=2, 
            perplexity=30, 
            n_iter=2500,          # 增加迭代次数
            early_exaggeration=20, # 增大早期夸大系数
            random_state=42, 
            n_jobs=-1             # 使用所有CPU核心加速
        )
        tsne_results = tsne.fit_transform(embeddings_for_tsne)
        
        end_time = time.time()
        print(f"t-SNE 降维完成，耗时: {end_time - start_time:.2f} 秒")

        # ------------------ 离群点检测 (Anomaly Detection) ------------------
        print("\n正在使用 LOF (Local Outlier Factor) 检测异常/离群节点...")
        # 使用 Local Outlier Factor (LOF) 识别离群点
        # contamination 参数表示数据集中离群点的比例, 这里假设为2%
        lof = LocalOutlierFactor(n_neighbors=30, contamination=0.02, novelty=False, n_jobs=-1)
        outlier_preds = lof.fit_predict(tsne_results)  # -1 表示离群点, 1 表示正常点

        outlier_mask_in_tsne = outlier_preds == -1
        outlier_indices_in_tsne = np.where(outlier_mask_in_tsne)[0]
        print(f"检测到 {len(outlier_indices_in_tsne)} 个异常(离群)节点。")

        # 追踪离群点在原始数据集中的索引并保存
        original_indices_all = np.arange(len(all_labels))
        indices_after_filter = original_indices_all[mask]

        if nodes_after_filter > sample_size:
            original_indices_of_sampled_points = indices_after_filter[indices]
            anomalous_original_indices = original_indices_of_sampled_points[outlier_indices_in_tsne]
        else:
            anomalous_original_indices = indices_after_filter[outlier_indices_in_tsne]

        anomalous_output_path = './anomalous_nodes.txt'
        with open(anomalous_output_path, 'w') as f:
            f.write("Anomalous (Outlier) Node Information\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Original Index':<15} {'True Label':<25} {'Predicted Label':<25}\n")
            f.write("-" * 80 + "\n")
            for idx in anomalous_original_indices:
                true_label = all_labels[idx]
                pred_label = all_preds[idx]
                true_label_name = node_type_names.get(true_label, f"类型_{true_label}")
                pred_label_name = node_type_names.get(pred_label, f"类型_{pred_label}")
                f.write(f"{idx:<15} {true_label_name:<25} {pred_label_name:<25}\n")
        print(f"异常(离群)节点信息已保存到: {anomalous_output_path}")
        
        # 绘图
        plt.figure(figsize=(16, 12))
        unique_labels_in_plot = np.unique(labels_for_tsne)
        
        # 使用 matplotlib 默认的颜色循环
        colors = plt.cm.get_cmap('tab20', len(unique_labels_in_plot))
        # 定义一组不同的标记样式
        markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']

        for i, label in enumerate(unique_labels_in_plot):
            # 找到当前类别且正确分类的节点
            idx = (labels_for_tsne == label) & ~misclassified_for_tsne
            plt.scatter(
                tsne_results[idx, 0], 
                tsne_results[idx, 1], 
                color=colors(i),
                marker=markers[i % len(markers)], # 为不同类别循环使用不同标记
                label=node_type_names.get(label, f"类型_{label}"),
                alpha=0.7,
                s=25,      # 适当增大点的大小以便观察标记
                edgecolors='k', # 为点添加黑色边框
                linewidths=0.5
            )
        
        # 单独绘制并标注所有分类错误的节点
        misclassified_idx_in_tsne = np.where(misclassified_for_tsne)[0]
        if len(misclassified_idx_in_tsne) > 0:
            plt.scatter(
                tsne_results[misclassified_idx_in_tsne, 0],
                tsne_results[misclassified_idx_in_tsne, 1],
                c='red',
                marker='x',
                s=50,
                label='Misclassified',
                alpha=0.9,
                linewidths=1.5
            )

        # 单独绘制并标注所有异常(离群)节点
        if len(outlier_indices_in_tsne) > 0:
            plt.scatter(
                tsne_results[outlier_indices_in_tsne, 0],
                tsne_results[outlier_indices_in_tsne, 1],
                facecolors='none',
                edgecolors='purple',
                s=150,
                linewidths=2,
                label='Anomalous (Outlier)',
                marker='o'
            )


        plt.title("t-SNE Visualization of Node Embeddings (with Misclassified and Anomalous Points)", fontsize=18)
        plt.xlabel("t-SNE Dimension 1", fontsize=14)
        plt.ylabel("t-SNE Dimension 2", fontsize=14)
        
        # 创建一个更易读的图例
        legend = plt.legend(loc='best', shadow=True, fontsize='large', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        legend.set_title("Node Types", prop={'size':'x-large'})
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = './figs/tsne_node_embeddings.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE 可视化图已保存到: {save_path}")
        plt.show()

    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        model.eval()
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            test_auc, test_std, _, _ = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test,
                                                                       result_y_test)
            print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
