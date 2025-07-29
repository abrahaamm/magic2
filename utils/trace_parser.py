import argparse
import json
import os
import random
import re
from collections import defaultdict, Counter

from tqdm import tqdm
import networkx as nx
import pickle as pkl


node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0

# darpa.py风格的常量
PREFIX = 'com.bbn.tc.schema.avro.cdm18.'
PLUS = '>>>>'
DELIMITER = '><'
FLOW = '>>'

metadata = {
    'trace':{
        'train': ['ta1-trace-e3-official-1.json'],
        'test': ['ta1-trace-e3-official-1.json.4']
    },
    'theia':{
            'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1', 'ta1-theia-e3-official-6r.json.2', 'ta1-theia-e3-official-6r.json.3'],
            'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets':{
            'train': ['ta1-cadets-e3-official.json','ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official.json.2', 'ta1-cadets-e3-official-2.json.1'],
            'test': ['ta1-cadets-e3-official-2.json']
    }
}


pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')


def generate_uuid2entity_mapping(dataset):
    """生成UUID到实体的映射，采用darpa.py的增强JSON解析逻辑"""
    print("构建UUID到实体的映射...")
    uuid2entity = {}
    
    # 添加统计计数器
    entity_type_stats = {}
    
    # 获取所有数据文件
    all_files = [f for f in os.listdir('../data/{}/'.format(dataset)) 
                if 'json' in f and not any(x in f for x in ['.txt', 'names', 'types', 'metadata'])]
    
    for file in tqdm(all_files, desc=f'生成 {dataset} uuid2entity映射'):
        file_path = '../data/{}/'.format(dataset) + file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if 'datum' in entry:
                            result = append_uuid2entity_enhanced(entry['datum'], uuid2entity)
                            # 统计实体类型
                            if result:
                                entity_type_stats[result] = entity_type_stats.get(result, 0) + 1
                    except json.JSONDecodeError:
                        # 尝试处理以逗号结尾的JSON格式
                        try:
                            entry = json.loads(line.strip()[:-1])
                            if 'datum' in entry:
                                result = append_uuid2entity_enhanced(entry['datum'], uuid2entity)
                                if result:
                                    entity_type_stats[result] = entity_type_stats.get(result, 0) + 1
                        except:
                            continue
                    except Exception:
                        continue
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            continue
    
    print(f"处理前实体类型统计: {entity_type_stats}")
    print(f"UUID引用处理前，uuid2entity条目数: {len(uuid2entity)}")
    
    # 更新使用现有UUIDs（处理UUID引用）
    uuid_reference_count = 0
    unit_found_count = 0
    
    for uuid, entity in list(uuid2entity.items()):
        if entity and entity.startswith(f'UUID{PLUS}'):
            uuid_reference_count += 1
            referenced_uuid = entity.replace(f'UUID{PLUS}', '')
            if referenced_uuid in uuid2entity:
                original_entity = uuid2entity[referenced_uuid]
                uuid2entity[uuid] = original_entity
                
                # 检查是否产生了UNIT类型
                if original_entity and 'UNIT' in str(original_entity):
                    unit_found_count += 1
    
    print(f"处理了 {uuid_reference_count} 个UUID引用")
    print(f"通过引用产生的UNIT类型: {unit_found_count}")
    
    # 最终统计
    final_type_stats = {}
    for uuid, entity in uuid2entity.items():
        if entity:
            if PLUS in entity:
                entity_type = entity.split(PLUS)[0]
            else:
                entity_type = entity
            final_type_stats[entity_type] = final_type_stats.get(entity_type, 0) + 1
    
    print(f"最终实体类型统计: {final_type_stats}")
    print(f"UUID映射总数: {len(uuid2entity)}")
    return uuid2entity


def append_uuid2entity_enhanced(entry, uuid2entity):
    """增强版的实体添加函数，支持更多DARPA实体类型，返回处理的实体类型"""
    # 处理事件
    if f'{PREFIX}Event' in entry:
        x = entry[f'{PREFIX}Event']
        entity_info = x['type'].replace('EVENT_', f'EVENT{PLUS}')
        uuid2entity[x['uuid']] = entity_info
        return f"EVENT ({x['type']})"

    # 处理主体（Subject）
    elif f'{PREFIX}Subject' in entry:
        x = entry[f'{PREFIX}Subject']
        try:
            if x['type'] == 'SUBJECT_UNIT':
                entity_info = f"UNIT{PLUS}{x['properties']['map']['name']}"
                uuid2entity[x['uuid']] = entity_info
                return "SUBJECT_UNIT"
            elif x['type'] == 'SUBJECT_PROCESS':
                if 'name' in x['properties']['map']:
                    entity_info = f"PROCESS{PLUS}{x['properties']['map']['name']}"
                    uuid2entity[x['uuid']] = entity_info
                    return "SUBJECT_PROCESS (with name)"
                elif 'host' in x['properties']['map']:
                    entity_info = f"UUID{PLUS}{x['properties']['map']['host'].upper()}"
                    uuid2entity[x['uuid']] = entity_info
                    return "SUBJECT_PROCESS (host ref)"
                else:
                    raise KeyError('No identified key.')
            else:
                entity_info = f"SUBJECT{PLUS}{x['properties']['map']['name']}"
                uuid2entity[x['uuid']] = entity_info
                return f"OTHER_SUBJECT ({x['type']})"
        except (KeyError, TypeError):
            entity_info = f"UUID{PLUS}{x['uuid']}"
            uuid2entity[x['uuid']] = entity_info
            return f"SUBJECT_ERROR ({x['type']})"

    # 处理文件对象
    elif f'{PREFIX}FileObject' in entry:
        x = entry[f'{PREFIX}FileObject']
        try:
            if x['type'] == 'FILE_OBJECT_DIR':
                entity_info = f"DIR{PLUS}{x['baseObject']['properties']['map']['path']}"
                uuid2entity[x['uuid']] = entity_info
                return "FILE_OBJECT_DIR"
            else:
                entity_info = f"FILE{PLUS}{x['baseObject']['properties']['map']['path']}"
                uuid2entity[x['uuid']] = entity_info
                return f"FILE_OBJECT ({x['type']})"
        except (KeyError, TypeError):
            entity_info = f"UUID{PLUS}{x['uuid']}"
            uuid2entity[x['uuid']] = entity_info
            return f"FILE_ERROR ({x['type']})"

    # 处理网络流对象
    elif f'{PREFIX}NetFlowObject' in entry:
        x = entry[f'{PREFIX}NetFlowObject']
        entity_info = (f'NETFLOW{PLUS}{x["localAddress"]}:{x["localPort"]}'
                      f'{FLOW}{x["remoteAddress"]}:{x["remotePort"]}')
        uuid2entity[x['uuid']] = entity_info
        return "NetFlowObject"

    # 处理IPC对象
    elif f'{PREFIX}IpcObject' in entry:
        x = entry[f'{PREFIX}IpcObject']
        entity_info = f"IPC{PLUS}{x['properties']['map']['path']}"
        uuid2entity[x['uuid']] = entity_info
        return "IpcObject"

    # 处理内存对象
    elif f'{PREFIX}MemoryObject' in entry:
        x = entry[f'{PREFIX}MemoryObject']
        entity_info = f'MEMORY{PLUS}{x["memoryAddress"]}'
        uuid2entity[x['uuid']] = entity_info
        return "MemoryObject"

    # 处理主机
    elif f'{PREFIX}Host' in entry:
        x = entry[f'{PREFIX}Host']
        ips = [(z if '.' in z else ':'.join(o.zfill(4) for o in z.split('%')[0].split(':')))  # ipv6
               for y in x['interfaces'] for z in y['ipAddresses']]
        ips = sorted(ips, key=lambda ip: len(ip.replace(':', '.').split('.')))
        ips = DELIMITER.join(ips)
        entity_info = f"HOST{PLUS}{ips}"
        uuid2entity[x['uuid']] = entity_info    

        return "Host"

    # 处理其他无特定信息的实体（注意：这些不加PLUS分隔符）
    elif f'{PREFIX}ProvenanceTagNode' in entry:
        x = entry[f'{PREFIX}ProvenanceTagNode']
        uuid2entity[x['tagId']] = 'ProvenanceTagNode'
        return "ProvenanceTagNode"
    elif f'{PREFIX}Principal' in entry:
        x = entry[f'{PREFIX}Principal']
        uuid2entity[x['uuid']] = 'Principal'
        return "Principal"
    elif f'{PREFIX}SrcSinkObject' in entry:
        x = entry[f'{PREFIX}SrcSinkObject']
        uuid2entity[x['uuid']] = 'SrcSinkObject'
        return "SrcSinkObject"
    elif f'{PREFIX}UnnamedPipeObject' in entry:
        x = entry[f'{PREFIX}UnnamedPipeObject']
        uuid2entity[x['uuid']] = 'UnnamedPipeObject'  # 注意：不加PLUS分隔符
        return "UnnamedPipeObject"
    elif any(key in entry for key in
             [f'{PREFIX}TimeMarker', f'{PREFIX}StartMarker', f'{PREFIX}EndMarker',
              f'{PREFIX}RegistryKeyObject', f'{PREFIX}UnitDependency']):
        return "SKIPPED_MARKER"

    return None


def append_uuid2entity(entry, uuid2entity):
    """原始的实体添加函数，保持向后兼容"""
    return append_uuid2entity_enhanced(entry, uuid2entity)


def unify_entity_info(entity_info):
    """统一实体信息格式，类似darpa.py的plus函数"""
    if entity_info is None or PLUS not in entity_info:
        return None, entity_info
    else:
        return entity_info.split(PLUS, 1)


def get_simplified_type(entity_info):
    """获取简化的实体类型"""
    if entity_info is None:
        return None
    
    if PLUS in entity_info:
        return entity_info.split(PLUS)[0]
    else:
        return entity_info


def get_unified_type_debug(entity_id, original_type, uuid2entity, debug=False):
    """获取经过unify处理的简化类型名称（调试版本）"""
    # 首先尝试从uuid2entity映射获取
    if entity_id in uuid2entity:
        entity_info = uuid2entity[entity_id]
        
        if entity_info:
            if PLUS in entity_info:
                # 解析复合信息，提取简化类型
                parts = entity_info.split(PLUS)
                if len(parts) >= 1:
                    simplified_type = parts[0]
                    normalized = normalize_type_name(simplified_type)
                    return normalized

            else:
                # 处理没有PLUS分隔符的类型（如UnnamedPipeObject等）
                normalized = normalize_simple_type(entity_info)
                return normalized
    else:
        if debug:
            print(f"    🔧 entity_id不在uuid2entity中")
    
    # 如果uuid2entity中没有，则根据原始类型转换
    # 这里是关键：确保SUBJECT_UNIT能被正确转换为UNIT
    converted_type = convert_original_type(original_type)
    if debug:
        print(f"    🔧 原始类型转换: {original_type} -> {converted_type}")
    
    if converted_type:
        return converted_type
    
    # 最后的回退：直接使用原始类型
    if debug:
        print(f"    🔧 使用原始类型: {original_type}")
    return original_type


def get_unified_type(entity_id, original_type, uuid2entity):
    """获取经过unify处理的简化类型名称"""
    # 首先尝试从uuid2entity映射获取
    if entity_id in uuid2entity:
        entity_info = uuid2entity[entity_id]
        if entity_info:
            if PLUS in entity_info:
                # 解析复合信息，提取简化类型
                parts = entity_info.split(PLUS)
                if len(parts) >= 1:
                    simplified_type = parts[0]
                    return normalize_type_name(simplified_type)
            else:
                # 处理没有PLUS分隔符的类型（如UnnamedPipeObject等）
                return normalize_simple_type(entity_info)
    
    # 如果uuid2entity中没有，则根据原始类型转换
    # 这里是关键：确保SUBJECT_UNIT能被正确转换为UNIT
    converted_type = convert_original_type(original_type)
    if converted_type:
        return converted_type
    
    # 最后的回退：直接使用原始类型
    return original_type


def normalize_type_name(type_name):
    """标准化类型名称，转换为与DARPA统计一致的格式"""
    type_mapping = {
        'PROCESS': 'PROCESS',
        'UNIT': 'UNIT', 
        'FILE': 'FILE',
        'DIR': 'DIR',
        'NETFLOW': 'NETFLOW',
        'MEMORY': 'MEMORY',
        'IPC': 'IPC',
        'HOST': 'HOST',
        'EVENT': 'EVENT',
        'SUBJECT': 'PROCESS',  # 通用SUBJECT映射为PROCESS
    }
    return type_mapping.get(type_name, type_name)


def normalize_simple_type(type_name):
    """处理没有PLUS分隔符的简单类型名称"""
    simple_type_mapping = {
        'UnnamedPipeObject': 'IPC',
        'SrcSinkObject': 'SRCSINK',
        'ProvenanceTagNode': 'PROVENANCE',
        'Principal': 'PRINCIPAL'
    }
    return simple_type_mapping.get(type_name, type_name)


def convert_original_type(original_type):
    """将原始DARPA类型转换为简化类型"""
    type_conversion = {
        'SUBJECT_PROCESS': 'PROCESS',
        'SUBJECT_UNIT': 'UNIT',
        'FILE_OBJECT_FILE': 'FILE',
        'FILE_OBJECT_DIR': 'DIR', 
        'FILE_OBJECT_CHAR': 'FILE',
        'FILE_OBJECT_LINK': 'FILE',
        'FILE_OBJECT_UNIX_SOCKET': 'FILE',
        'NetFlowObject': 'NETFLOW',
        'MemoryObject': 'MEMORY',
        'UnnamedPipeObject': 'IPC',
        'IpcObject': 'IPC',
        'SRCSINK_UNKNOWN': 'SRCSINK',  # 保持原始分类，不映射为UNKNOWN
        'SrcSinkObject': 'SRCSINK',    # 同样映射为SRCSINK
    }
    return type_conversion.get(original_type, original_type)


def collect_and_output_statistics(dataset, uuid2entity):
    """收集并输出统计信息，采用详细的统计格式"""
    print("\n" + "="*60)
    print(f"统计分析 - 数据集: {dataset.upper()}")
    print("="*60)
    
    # 统计实体类型分布
    entity_type_stats = Counter()
    simplified_type_stats = Counter()
    
    for entity_info in uuid2entity.values():
        if entity_info:
            # 完整类型统计
            entity_type_stats[entity_info] += 1
            
            # 简化类型统计
            simplified_type = get_simplified_type(entity_info)
            if simplified_type:
                simplified_type_stats[simplified_type] += 1
    
    # 输出实体类型分布
    print("\n【实体类型分布】")
    print("-" * 40)
    for entity_type, count in simplified_type_stats.most_common():
        percentage = count / sum(simplified_type_stats.values()) * 100
        print(f"{entity_type:15}: {count:,} ({percentage:.2f}%)")
    
    print(f"\n总实体数量: {sum(simplified_type_stats.values()):,}")
    
    # 检查是否有UNIT类型
    if 'UNIT' in simplified_type_stats:
        print(f"\n✅ UNIT类型已正确识别: {simplified_type_stats['UNIT']} 个")
    else:
        print(f"\n❌ 未发现UNIT类型")
    
    # 统计边信息
    train_stats, test_stats = collect_edge_statistics_detailed(dataset, uuid2entity)
    
    # 输出详细的数据集统计信息
    print_dataset_statistics(train_stats, test_stats)
    
    print("\n" + "="*60)


def collect_edge_statistics_detailed(dataset, uuid2entity):
    """收集详细的边统计信息，分训练和测试数据"""
    train_stats = {'sbj': Counter(), 'obj': Counter()}
    test_stats = {'sbj': Counter(), 'obj': Counter()}
    
    # 收集训练数据统计
    for file in metadata[dataset]['train']:
        collect_edge_statistics_for_file(dataset, file, train_stats, uuid2entity)
    
    # 收集测试数据统计
    for file in metadata[dataset]['test']:
        collect_edge_statistics_for_file(dataset, file, test_stats, uuid2entity)
    
    return train_stats, test_stats


def collect_edge_statistics_for_file(dataset, file, stats, uuid2entity):
    """为单个文件收集边统计信息，使用经过unify处理的简化类型名称"""
    edge_file_path = '../data/{}/'.format(dataset) + file + '.txt'
    if not os.path.exists(edge_file_path):
        return
    
    unit_count = 0  # 添加UNIT类型计数器
    conversion_examples = {}  # 记录类型转换示例
    unit_debug_info = []  # 调试：记录UNIT类型的详细信息
    
    try:
        with open(edge_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) >= 6:
                        src_id, src_type, dst_id, dst_type, edge_type, timestamp = parts[:6]
                        
                        # 获取简化的类型名称
                        # 先检查是否可能包含UNIT类型，再决定是否开启调试
                        debug_mode = (src_type == 'UNIT' or dst_type == 'UNIT' or 
                                    'SUBJECT_UNIT' in str(src_type) or 'SUBJECT_UNIT' in str(dst_type))
                        
                        src_unified_type = get_unified_type_debug(src_id, src_type, uuid2entity, debug=debug_mode)
                        dst_unified_type = get_unified_type_debug(dst_id, dst_type, uuid2entity, debug=debug_mode)
                        
                        # 记录类型转换示例
                        if src_type not in conversion_examples:
                            conversion_examples[src_type] = src_unified_type
                        if dst_type not in conversion_examples:
                            conversion_examples[dst_type] = dst_unified_type
                        
                        # 特别关注UNIT类型的检测
                        if src_unified_type == 'UNIT':
                            unit_count += 1
                            unit_debug_info.append(f"主体: {src_id} | 原始: {src_type} -> 统一: {src_unified_type}")
                        
                        if dst_unified_type == 'UNIT':
                            unit_count += 1
                            unit_debug_info.append(f"客体: {dst_id} | 原始: {dst_type} -> 统一: {dst_unified_type}")
                        
                        # 统计主体类型
                        if src_unified_type:
                            stats['sbj'][src_unified_type] += 1
                        
                        # 统计客体类型
                        if dst_unified_type:
                            stats['obj'][dst_unified_type] += 1
                            
                except (ValueError, IndexError):
                    continue
                    
    except Exception as e:
        print(f"收集统计信息时出错 {file}: {e}")
    
    # 输出文件级别的统计信息
    if unit_count > 0:
        print(f"🔍 调试：文件 {file} 统计阶段发现 {unit_count} 个UNIT类型实体")
        # 显示前3个UNIT类型的详细信息
        for i, info in enumerate(unit_debug_info[:3]):
            print(f"  {info}")
        if len(unit_debug_info) > 3:
            print(f"  ... 还有 {len(unit_debug_info) - 3} 个UNIT实体")
    else:
        print(f"🔍 调试：文件 {file} 统计阶段未发现UNIT类型实体")
    
    # 输出类型转换示例（仅显示前10个）
    print(f"文件 {file} 的类型转换示例:")
    for i, (orig, unified) in enumerate(list(conversion_examples.items())[:10]):
        if orig != unified:  # 只显示有转换的
            print(f"  {orig} -> {unified}")
    if len(conversion_examples) > 10:
        print(f"  ... 共{len(conversion_examples)}种类型")


def collect_edge_statistics(dataset):
    """简化版边统计收集，保持向后兼容"""
    stats = {'sbj': Counter(), 'obj': Counter()}
    
    for phase in ['train', 'test']:
        for file in metadata[dataset][phase]:
            edge_file_path = '../data/{}/'.format(dataset) + file + '.txt'
            if not os.path.exists(edge_file_path):
                continue
            
            try:
                with open(edge_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split('\t')
                        if len(parts) >= 6:
                            src_id, src_type, dst_id, dst_type, edge_type, timestamp = parts[:6]
                            
                            # 统计类型
                            stats['sbj'][src_type] += 1
                            stats['obj'][dst_type] += 1
                            
            except Exception as e:
                print(f"收集边统计信息时出错 {file}: {e}")
    
    return stats


def print_dataset_statistics(train_stats, test_stats):
    """输出数据集统计信息，格式与图片一致"""
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    
    # 输出训练数据统计
    if train_stats['sbj'] or train_stats['obj']:
        print(f"\n训练数据统计:")
        print_entity_statistics("sbj", train_stats['sbj'])
        print_entity_statistics("obj", train_stats['obj'])
    
    # 输出测试数据统计  
    if test_stats['sbj'] or test_stats['obj']:
        print(f"\n测试数据统计:")
        print_entity_statistics("sbj", test_stats['sbj'])
        print_entity_statistics("obj", test_stats['obj'])


def print_entity_statistics(entity_type, type_counts):
    """打印单类实体的统计信息"""
    if not type_counts:
        return
    
    # 根据实体类型过滤要显示的类型
    filtered_counts = {}
    
    if entity_type == 'sbj':
        # 主体统计：包含所有类型（包括UNIT）
        filtered_counts = type_counts.copy()
    elif entity_type == 'obj':
        # 客体统计：排除UNKNOWN和IPC，但包含UNIT
        exclude_types = {'UNKNOWN', 'IPC'}
        filtered_counts = {k: v for k, v in type_counts.items() if k not in exclude_types}
    
    if not filtered_counts:
        return
    
    total_count = sum(filtered_counts.values())
    print(f"\n[{entity_type}] 种类及比例:")
    
    # 按数量降序排列
    sorted_types = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    
    for type_name, count in sorted_types:
        percentage = (count / total_count) * 100
        print(f"  {type_name}: {count} ({percentage:.2f}%)")


def print_edge_statistics(edge_stats):
    """打印边统计信息，格式与图片一致"""
    print("\n【边类型统计】")
    print("-" * 40)
    
    sbj_stats = edge_stats.get('sbj', {})
    obj_stats = edge_stats.get('obj', {})
    
    # 主体统计
    if sbj_stats:
        total_sbj = sum(sbj_stats.values())
        print(f"\n[sbj] 种类及比例:")
        sorted_sbj = sorted(sbj_stats.items(), key=lambda x: x[1], reverse=True)
        for type_name, count in sorted_sbj:
            percentage = (count / total_sbj) * 100
            print(f"  {type_name}: {count} ({percentage:.2f}%)")
    
    # 客体统计
    if obj_stats:
        # 过滤某些类型
        filtered_obj_stats = {k: v for k, v in obj_stats.items() 
                            if k not in ['MemoryObject', 'UnnamedPipeObject']}
        if filtered_obj_stats:
            total_obj = sum(filtered_obj_stats.values())
            print(f"\n[obj] 种类及比例:")
            sorted_obj = sorted(filtered_obj_stats.items(), key=lambda x: x[1], reverse=True)
            for type_name, count in sorted_obj:
                percentage = (count / total_obj) * 100
                print(f"  {type_name}: {count} ({percentage:.2f}%)")


def read_single_graph(dataset, malicious, path, test=False):
    global node_type_cnt, edge_type_cnt
    g = nx.DiGraph()
    print('converting {} ...'.format(path))
    path = '../data/{}/'.format(dataset) + path + '.txt'
    f = open(path, 'r')
    lines = []
    for l in f.readlines():
        split_line = l.split('\t')
        src, src_type, dst, dst_type, edge_type, ts = split_line
        ts = int(ts)
        if not test:
            if src in malicious or dst in malicious:
                if src in malicious and src_type != 'MemoryObject':
                    continue
                if dst in malicious and dst_type != 'MemoryObject':
                    continue

        if src_type not in node_type_dict:
            node_type_dict[src_type] = node_type_cnt
            node_type_cnt += 1
        if dst_type not in node_type_dict:
            node_type_dict[dst_type] = node_type_cnt
            node_type_cnt += 1
        if edge_type not in edge_type_dict:
            edge_type_dict[edge_type] = edge_type_cnt
            edge_type_cnt += 1
        if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
            lines.append([dst, src, dst_type, src_type, edge_type, ts])
        else:
            lines.append([src, dst, src_type, dst_type, edge_type, ts])
    lines.sort(key=lambda l: l[5])

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []
    for l in lines:
        src, dst, src_type, dst_type, edge_type = l[:5]
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)
    return node_map, g


def preprocess_dataset(dataset):
    # 首先生成UUID到实体的映射，使用增强版本
    uuid2entity = generate_uuid2entity_mapping(dataset)
    
    id_nodetype_map = {}
    id_nodename_map = {}
    
    # 从uuid2entity中提取类型和名称信息，采用darpa.py的unify逻辑
    for uuid, entity_info in uuid2entity.items():
        if entity_info and entity_info != f"UUID{PLUS}{uuid}":
            if PLUS in entity_info:
                entity_type, entity_name = entity_info.split(PLUS, 1)
                id_nodetype_map[uuid] = entity_type
                if entity_name:
                    id_nodename_map[uuid] = entity_name
            else:
                # 没有PLUS分隔符的简单类型
                id_nodetype_map[uuid] = entity_info
    
    print(f"实体类型映射数量: {len(id_nodetype_map)}")
    print(f"实体名称映射数量: {len(id_nodename_map)}")
    
    # 调试：检查id_nodetype_map中的UNIT类型
    unit_entities = {uuid: type_name for uuid, type_name in id_nodetype_map.items() if type_name == 'UNIT'}
    print(f"🔍 调试：id_nodetype_map中的UNIT类型实体数量: {len(unit_entities)}")
    if len(unit_entities) > 0:
        print(f"🔍 调试：前5个UNIT实体: {list(unit_entities.items())[:5]}")
    
    # 生成边文件 - 保持原有的处理逻辑
    total_unit_edges = 0  # 调试：计算包含UNIT类型的边数量
    
    # 调试：获取一些UNIT类型的UUID用于检查
    unit_uuids = [uuid for uuid, type_name in list(id_nodetype_map.items())[:100] if type_name == 'UNIT']
    print(f"🔍 调试：用于检查的UNIT UUID示例: {unit_uuids[:5]}")
    
    for key in metadata[dataset]:
        for file in metadata[dataset][key]:
            if os.path.exists('../data/{}/'.format(dataset) + file + '.txt'):
                continue
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw = open('../data/{}/'.format(dataset) + file + '.txt', 'w', encoding='utf-8')
            print('processing {} ...'.format(file))
            file_unit_edges = 0  # 文件级别的UNIT边计数
            
            # 调试计数器
            total_events = 0
            processed_events = 0
            unit_as_src_count = 0
            unit_as_dst_count = 0
            
            for line in tqdm(f):
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    total_events += 1
                    edgeType = pattern_type.findall(line)[0]
                    timestamp = pattern_time.findall(line)[0]
                    srcId = pattern_src.findall(line)

                    if len(srcId) == 0: continue
                    srcId = srcId[0]
                    
                    # 调试：检查是否为UNIT类型的srcId
                    if srcId in unit_uuids[:5]:  # 只检查前5个UNIT UUID
                        print(f"🔍 发现UNIT作为srcId: {srcId}")
                    
                    if not srcId in id_nodetype_map:
                        continue
                    srcType = id_nodetype_map[srcId]
                    
                    # 调试：如果srcType是UNIT，记录
                    if srcType == 'UNIT':
                        unit_as_src_count += 1
                        if unit_as_src_count <= 3:  # 只打印前3个
                            print(f"🔍 发现UNIT作为主体: srcId={srcId}, edgeType={edgeType}")
                    
                    processed_events += 1
                    dstId1 = pattern_dst1.findall(line)
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dstId1 = dstId1[0]
                        if not dstId1 in id_nodetype_map:
                            continue
                        dstType1 = id_nodetype_map[dstId1]
                        
                        # 调试：如果dstType1是UNIT，记录
                        if dstType1 == 'UNIT':
                            unit_as_dst_count += 1
                            if unit_as_dst_count <= 3:  # 只打印前3个
                                print(f"🔍 发现UNIT作为客体1: dstId1={dstId1}, edgeType={edgeType}")
                        
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge1)
                        
                        # 调试：检查是否包含UNIT类型
                        if srcType == 'UNIT' or dstType1 == 'UNIT':
                            file_unit_edges += 1
                            total_unit_edges += 1

                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]
                        if not dstId2 in id_nodetype_map.keys():
                            continue
                        dstType2 = id_nodetype_map[dstId2]
                        
                        # 调试：如果dstType2是UNIT，记录
                        if dstType2 == 'UNIT':
                            unit_as_dst_count += 1
                            if unit_as_dst_count <= 3:  # 只打印前3个
                                print(f"🔍 发现UNIT作为客体2: dstId2={dstId2}, edgeType={edgeType}")
                        
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge2)
                        
                        # 调试：检查是否包含UNIT类型
                        if srcType == 'UNIT' or dstType2 == 'UNIT':
                            file_unit_edges += 1
                            total_unit_edges += 1
            
            print(f"🔍 调试：文件 {file} 处理统计:")
            print(f"  总事件数: {total_events}")
            print(f"  已处理事件数: {processed_events}")
            print(f"  UNIT作为主体次数: {unit_as_src_count}")
            print(f"  UNIT作为客体次数: {unit_as_dst_count}")
            print(f"  包含UNIT类型的边数量: {file_unit_edges}")
            fw.close()
            f.close()
    
    print(f"🔍 调试：总共生成的包含UNIT类型的边数量: {total_unit_edges}")
    if len(id_nodename_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)
    
    # 收集并输出统计信息，使用增强的统计功能
    collect_and_output_statistics(dataset, uuid2entity)


def read_graphs(dataset):
    malicious_entities = '../data/{}/{}.txt'.format(dataset, dataset)
    f = open(malicious_entities, 'r')
    malicious_entities = set()
    for l in f.readlines():
        malicious_entities.add(l.lstrip().rstrip())

    preprocess_dataset(dataset)
    train_gs = []
    for file in metadata[dataset]['train']:
        _, train_g = read_single_graph(dataset, malicious_entities, file, False)
        train_gs.append(train_g)
    test_gs = []
    test_node_map = {}
    count_node = 0
    for file in metadata[dataset]['test']:
        node_map, test_g = read_single_graph(dataset, malicious_entities, file, True)
        assert len(node_map) == test_g.number_of_nodes()
        test_gs.append(test_g)
        for key in node_map:
            if key not in test_node_map:
                test_node_map[key] = node_map[key] + count_node
        count_node += test_g.number_of_nodes()

    if os.path.exists('../data/{}/names.json'.format(dataset)) and os.path.exists('../data/{}/types.json'.format(dataset)):
        with open('../data/{}/names.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodename_map = json.load(f)
        with open('../data/{}/types.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodetype_map = json.load(f)
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []
        for e in malicious_entities:
            if e in test_node_map and e in id_nodetype_map and id_nodetype_map[e] != 'MemoryObject' and id_nodetype_map[e] != 'UnnamedPipeObject':
                final_malicious_entities.append(test_node_map[e])
                if e in id_nodename_map:
                    malicious_names.append(id_nodename_map[e])
                    f.write('{}\t{}\n'.format(e, id_nodename_map[e]))
                else:
                    malicious_names.append(e)
                    f.write('{}\t{}\n'.format(e, e))
    else:
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []
        for e in malicious_entities:
            if e in test_node_map:
                final_malicious_entities.append(test_node_map[e])
                malicious_names.append(e)
                f.write('{}\t{}\n'.format(e, e))

    pkl.dump((final_malicious_entities, malicious_names), open('../data/{}/malicious.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(train_g) for train_g in train_gs], open('../data/{}/train.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(test_g) for test_g in test_gs], open('../data/{}/test.pkl'.format(dataset), 'wb'))

    print(f"\n成功处理 {dataset} 数据集:")
    print(f"  训练图数量: {len(train_gs)}")
    print(f"  测试图数量: {len(test_gs)}")
    print(f"  节点类型数量: {node_type_cnt}")
    print(f"  边类型数量: {edge_type_cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser - 改进版本')
    parser.add_argument("--dataset", type=str, default="trace", 
                       choices=['trace', 'theia', 'cadets'], 
                       help='要处理的数据集名称')
    args = parser.parse_args()
    
    print(f"开始处理 {args.dataset} 数据集...")
    read_graphs(args.dataset)
    print("处理完成!")
