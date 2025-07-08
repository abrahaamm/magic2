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
warnings.filterwarnings('ignore')


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
        classifier = NodeClassifier(encoder.output_hidden_dim, n_dim).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()

        all_preds, all_labels = [], []
        total_nodes = 0
        n_test = metadata['n_test']

        print("\n" + "="*80)
        print(f"评估测试集中的 {n_test} 个图...")
        print("="*80 + "\n")
        
        # 节点类型名称映射
        node_type_names = {
            0: "SRCSINK_UNKNOWN",
            1: "SUBJECT",
            2: "FILE_OBJECT",
            3: "MEMORY_OBJECT",
            4: "NETFLOW_OBJECT",
            5: "PRINCIPAL",
            6: "PROCESS",
            7: "THREAD",
            8: "UNNAMED_PIPE_OBJECT",
            9: "FILE_OBJECT_LINK",
            10: "FILE_OBJECT_BLOCK"
        }

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
        class_names = [node_type_names.get(i, f"类型_{i}") for i in range(n_dim)]
        
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
