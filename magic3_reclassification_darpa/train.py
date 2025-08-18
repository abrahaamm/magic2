import os
import random
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


# ------------------ 节点分类器与训练 (仅用于 trace 数据集) ------------------


class NodeClassifier(nn.Module):
    """简单的两层全连接节点分类器"""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 权重初始化
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)


def train_node_classification_trace(device, dataset_name, main_args):
    """专门为trace数据集训练节点分类模型"""
    metadata = load_metadata(dataset_name)
    main_args.n_dim = metadata['node_feature_dim']
    main_args.e_dim = metadata['edge_feature_dim']
    
    # 构建模型
    encoder = build_model(main_args)
    encoder = encoder.to(device)
    
    # 添加节点分类头 - 固定使用5类 (UNIT=0, FILE=1, NETFLOW=2, PROCESS=3, DIR=4)
    num_classes = 5  # 与another目录保持一致的5类分类
    print(f"✅ 强制使用与another一致的5类节点分类系统")
    
    # 确保类别ID映射与another完全一致
    another_class_mapping = {0: 'UNIT', 1: 'FILE', 2: 'NETFLOW', 3: 'PROCESS', 4: 'DIR'}
    print(f"📊 类别映射: {another_class_mapping}")
    node_classifier = NodeClassifier(
        input_dim=encoder.output_hidden_dim,
        num_classes=num_classes,
        hidden_dim=64,
        dropout=0.1
    ).to(device)
    
    print(f"📊 使用与another一致的5类节点分类: UNIT(0), FILE(1), NETFLOW(2), PROCESS(3), DIR(4)")
    print(f"📝 输入数据格式: another标准格式 (数字ID)")
    print(f"🔍 应用another系统的实体映射规则和过滤逻辑")
    
    # 计算类别权重
    print("计算训练数据的类别分布...")
    class_counts = torch.zeros(num_classes, dtype=torch.float)
    n_train = metadata['n_train']
    
    for i in range(n_train):
        g = load_entity_level_dataset(dataset_name, 'train', i)
        node_labels = g.ndata['attr'].argmax(dim=1)
        for class_id in range(num_classes):
            class_counts[class_id] += (node_labels == class_id).sum().item()
    
    # 计算类别权重 - 基于实际数据分布，与another保持一致的处理方式
    total_samples = class_counts.sum()
    computed_weights = total_samples / (num_classes * class_counts + 1e-6)  # 避免除零
    
    print(f"计算得到的类别权重: {computed_weights}")
    
    # 选择权重策略：可以使用计算得到的权重，或使用与another相同的固定权重
    use_fixed_weights = False  # 设为True使用固定权重，False使用计算权重
    
    if use_fixed_weights:
        # 使用与another目录相同的固定权重 [1.0, 100.0, 160.0, 10240.0, 12800.0]
        class_weights = torch.tensor([1.0, 100.0, 160.0, 10240.0, 12800.0], dtype=torch.float).to(device)
        print(f"使用固定类别权重 (与another一致)")
    else:
        # 使用计算得到的权重
        class_weights = computed_weights.to(device)
        print(f"使用基于数据分布计算的类别权重")
    
    print(f"类别分布:")
    for i in range(num_classes):
        class_names = ['UNIT', 'FILE', 'NETFLOW', 'PROCESS', 'DIR']
        print(f"  类别 {i} ({class_names[i]}): {int(class_counts[i]):,} 样本 (权重: {class_weights[i]:.1f})")
    
    # 创建优化器
    encoder_optimizer = create_optimizer(main_args.optimizer, encoder, main_args.lr, main_args.weight_decay)
    classifier_optimizer = torch.optim.Adam(node_classifier.parameters(), lr=main_args.lr * 0.1)
    
    # 创建加权损失函数
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"最终使用的类别权重 [UNIT, FILE, NETFLOW, PROCESS, DIR]: {class_weights.cpu().numpy()}")
    
    # 训练循环
    epoch_iter = tqdm(range(main_args.max_epoch))
    
    print(f"Training node classification model on {dataset_name} dataset...")
    print(f"Training graphs: {n_train}")
    print(f"Node feature dimension: {main_args.n_dim}")
    print(f"Edge feature dimension: {main_args.e_dim}")
    print(f"Number of node classes: {num_classes}")
    print(f"Encoder hidden dimension: {encoder.output_hidden_dim}")
    print(f"Using class weights to handle imbalanced data")
    
    for epoch in epoch_iter:
        epoch_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_reconstruction_loss = 0.0
        total_nodes = 0
        
        for i in range(n_train):
            g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
            node_labels = g.ndata['attr'].argmax(dim=1)
            
            encoder.train()
            node_classifier.train()
            
            node_embeddings = encoder.embed(g)
            node_logits = node_classifier(node_embeddings)
            
            classification_loss = classification_criterion(node_logits, node_labels)
            reconstruction_loss = encoder(g)
            
            total_loss = classification_loss + 0.05 * reconstruction_loss
            
            encoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            total_loss.backward()
            encoder_optimizer.step()
            classifier_optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_classification_loss += classification_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()
            total_nodes += g.number_of_nodes()
            
            del g, node_embeddings, node_logits
            torch.cuda.empty_cache() if device != "cpu" else None
        
        avg_epoch_loss = epoch_loss / n_train
        avg_classification_loss = epoch_classification_loss / n_train
        avg_reconstruction_loss = epoch_reconstruction_loss / n_train
        
        epoch_iter.set_description(
            f"Epoch {epoch} | Total: {avg_epoch_loss:.4f} | "
            f"Class: {avg_classification_loss:.4f} | "
            f"Recon: {avg_reconstruction_loss:.4f}"
        )
    
    # 保存模型到原来的路径
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(encoder.state_dict(), f"./checkpoints/checkpoint-{dataset_name}-encoder.pt")
    torch.save(node_classifier.state_dict(), f"./checkpoints/checkpoint-{dataset_name}-classifier.pt")
    
    # 删除距离保存文件
    save_dict_path = f'./eval_result/distance_save_{dataset_name}.pkl'
    if os.path.exists(save_dict_path):
        os.unlink(save_dict_path)


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset
    if dataset_name == 'streamspot':
        main_args.num_hidden = 256
        main_args.max_epoch = 5
        main_args.num_layers = 4
    elif dataset_name == 'wget':
        main_args.num_hidden = 256
        main_args.max_epoch = 2
        main_args.num_layers = 4
    elif dataset_name == 'trace':
        main_args.num_hidden = 64
        main_args.max_epoch = 50
        main_args.num_layers = 3
    else:
        main_args.num_hidden = 64
        main_args.max_epoch = 50
        main_args.num_layers = 3
    set_random_seed(0)

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        if dataset_name == 'streamspot':
            batch_size = 12
        else:
            batch_size = 1
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model = model.to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        model = batch_level_train(model, graphs, (extract_dataloaders(train_index, batch_size)),
                                  optimizer, main_args.max_epoch, device, main_args.n_dim, main_args.e_dim)
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
        
        save_dict_path = f'./eval_result/distance_save_{dataset_name}.pkl'
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)
    elif dataset_name == 'trace':
        # 使用新节点分类训练流程
        train_node_classification_trace(device, dataset_name, main_args)
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model = model.to(device)
        model.train()
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        epoch_iter = tqdm(range(main_args.max_epoch))
        n_train = metadata['n_train']
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                model.train()
                loss = model(g)
                loss /= n_train
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
        save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset_name)
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
