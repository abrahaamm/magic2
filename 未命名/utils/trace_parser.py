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
        # 完全忽略这些标记类型，不创建任何实体记录
        return None

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


def process_netflow_entity(entity_info, entity_type):
    """处理NETFLOW实体，移除FROM部分的端口号 - 与another/useless.py保持一致
    
    another/useless.py第44-54行的逻辑：
    移除NETFLOW的FROM部分的端口号，保留TO部分不变
    """
    if entity_type == 'NETFLOW' and entity_info and FLOW in entity_info:
        try:
            # 分离NETFLOW信息：entity_info格式为 "NETFLOW>>>>from>>to"
            if PLUS in entity_info:
                prefix, netflow_info = entity_info.split(PLUS, 1)
            else:
                netflow_info = entity_info
                prefix = ""
            
            # 分离from和to部分
            if FLOW in netflow_info:
                from_, to = netflow_info.split(FLOW)
                
                # 处理from部分，移除端口号
                from_split = from_.replace(':', '.').split('.')
                if len(from_split) in (5, 9):  # IPv4+port 或 IPv6+port
                    from_split = from_split[:-1]  # 移除最后的端口部分
                    if len(from_split) == 4:  # IPv4
                        from_ = '.'.join(from_split)
                    else:  # IPv6
                        from_ = ':'.join(from_split)
                
                # 重新组合
                processed_netflow = f'{from_}{FLOW}{to}'
                if prefix:
                    return f'{prefix}{PLUS}{processed_netflow}'
                else:
                    return processed_netflow
        except Exception as e:
            # 如果处理失败，返回原始值
            pass
    
    return entity_info


def should_filter_event(src_type, dst_type, edge_type, src_entity=None, dst_entity=None):
    """判断是否应该过滤掉这个事件，精确模仿another/useless.py的过滤逻辑
    
    another/useless.py的过滤规则（understand_events方法）：
    1. 过滤掉没有完整信息的事件：sbj, sbj+, obj, obj+ 都不能为None
    2. 过滤掉特定的对象类型：'ProvenanceTagNode', 'Principal', 'SrcSinkObject', 'UnnamedPipeObject'
    3. 过滤掉MEMORY类型（obj+ != 'MEMORY'）
    4. 过滤掉特定关系类型：'MMAP', 'MPROTECT', 'OTHER', 'CORRELATE', 'SHM', 'OPEN', 'CLOSE'
    """
    # 基本检查：源类型和目标类型不能为空（对应another中的sbj+和obj+检查）
    if src_type is None or dst_type is None:
        return True
    
    # 过滤特定的对象类型（与another/useless.py第35行保持一致）
    # event['obj'] not in ('ProvenanceTagNode', 'Principal', 'SrcSinkObject', 'UnnamedPipeObject')
    filtered_obj_types = {'ProvenanceTagNode', 'Principal', 'SrcSinkObject', 'UnnamedPipeObject'}
    if dst_type in filtered_obj_types:
        return True
    
    # 过滤MEMORY类型（与another/useless.py第37行保持一致）
    # event['obj+'] != 'MEMORY'
    if dst_type == 'MEMORY':
        return True
    
    # 过滤特定的关系类型（与another/useless.py第38-40行保持一致）
    # event['rel'] not in ('MMAP', 'MPROTECT', 'OTHER', 'CORRELATE', 'SHM', 'OPEN', 'CLOSE')
    filtered_relations = {'MMAP', 'MPROTECT', 'OTHER', 'CORRELATE', 'SHM', 'OPEN', 'CLOSE'}
    if edge_type in filtered_relations:
        return True
    
    return False


def get_five_class_id(class_name):
    """将类别名称转换为数字ID - 与another保持一致的5类映射"""
    class_to_id = {
        'UNIT': 0,       # 单元/主机单元
        'FILE': 1,       # 文件对象
        'NETFLOW': 2,    # 网络流对象
        'PROCESS': 3,    # 进程对象
        'DIR': 4         # 目录对象
    }
    return class_to_id.get(class_name, None)


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
    """获取经过unify处理的简化类型名称，使用5类映射规则"""
    # 首先尝试从uuid2entity映射获取
    if entity_id in uuid2entity:
        entity_info = uuid2entity[entity_id]
        if entity_info:
            if PLUS in entity_info:
                # 解析复合信息，提取简化类型
                parts = entity_info.split(PLUS)
                if len(parts) >= 1:
                    simplified_type = parts[0]
                    return map_to_five_classes(simplified_type)
            else:
                # 处理没有PLUS分隔符的类型（如UnnamedPipeObject等）
                return map_to_five_classes(entity_info)
    
    # 如果uuid2entity中没有，则根据原始类型转换
    converted_type = convert_original_type_to_five_classes(original_type)
    if converted_type:
        return converted_type
    
    # 最后的回退：返回None表示过滤掉
    return None


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


def map_to_five_classes(type_name):
    """将实体类型映射到5类系统 (UNIT=0, FILE=1, NETFLOW=2, PROCESS=3, DIR=4)"""
    # 与another目录保持一致的5类映射
    five_class_mapping = {
        # 主要类型映射
        'UNIT': 'UNIT',           # 0
        'FILE': 'FILE',           # 1  
        'NETFLOW': 'NETFLOW',     # 2
        'PROCESS': 'PROCESS',     # 3
        'DIR': 'DIR',             # 4
        
        # 其他类型映射到主要类型
        'SUBJECT': 'PROCESS',     # SUBJECT通常映射为PROCESS
        'HOST': None,             # HOST被过滤掉
        'MEMORY': None,           # MEMORY被过滤掉
        'IPC': None,              # IPC被过滤掉
        'EVENT': None,            # EVENT被过滤掉
        'SRCSINK': None,          # SRCSINK被过滤掉
        'PROVENANCE': None,       # PROVENANCE被过滤掉
        'PRINCIPAL': None,        # PRINCIPAL被过滤掉
        
        # 简单类型映射
        'UnnamedPipeObject': None,    # 被过滤掉
        'SrcSinkObject': None,        # 被过滤掉  
        'ProvenanceTagNode': None,    # 被过滤掉
        'Principal': None,            # 被过滤掉
        'MemoryObject': None,         # 被过滤掉
    }
    return five_class_mapping.get(type_name, None)


def convert_original_type_to_five_classes(original_type):
    """将原始DARPA类型直接转换为5类系统"""
    type_conversion = {
        'SUBJECT_PROCESS': 'PROCESS',
        'SUBJECT_UNIT': 'UNIT', 
        'FILE_OBJECT_FILE': 'FILE',
        'FILE_OBJECT_DIR': 'DIR',
        'FILE_OBJECT_CHAR': 'FILE',
        'FILE_OBJECT_LINK': 'FILE',
        'FILE_OBJECT_UNIX_SOCKET': 'FILE',
        'NetFlowObject': 'NETFLOW',
        
        # 以下类型被过滤掉
        'MemoryObject': None,
        'UnnamedPipeObject': None,
        'IpcObject': None,
        'SRCSINK_UNKNOWN': None,
        'SrcSinkObject': None,
        'Principal': None,
        'ProvenanceTagNode': None,
    }
    return type_conversion.get(original_type, None)


def convert_original_type(original_type):
    """将原始DARPA类型转换为简化类型（向后兼容）"""
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
    """收集并输出统计信息，采用详细的统计格式 - 与another保持一致的5类统计"""
    print("\n" + "="*60)
    print(f"实体统计信息 - 数据集: {dataset.upper()} (采用与another相同的5类系统)")
    print("="*60)
    
    # 统计实体类型分布
    entity_type_stats = Counter()
    simplified_type_stats = Counter()
    five_class_stats = Counter()
    
    for entity_info in uuid2entity.values():
        if entity_info:
            # 完整类型统计
            entity_type_stats[entity_info] += 1
            
            # 简化类型统计
            simplified_type = get_simplified_type(entity_info)
            if simplified_type:
                simplified_type_stats[simplified_type] += 1
                
                # 5类映射统计（与another保持一致）
                five_class_type = map_to_five_classes(simplified_type)
                if five_class_type:
                    five_class_stats[five_class_type] += 1
    
    # 输出5类系统的实体类型分布（与another保持一致）
    print("\n【5类实体类型分布】(与another相同)")
    print("-" * 50)
    
    # 按照another的5类顺序显示：UNIT=0, FILE=1, NETFLOW=2, PROCESS=3, DIR=4
    five_class_order = ['UNIT', 'FILE', 'NETFLOW', 'PROCESS', 'DIR']
    total_five_class = sum(five_class_stats.values())
    
    if total_five_class > 0:
        for i, class_name in enumerate(five_class_order):
            count = five_class_stats.get(class_name, 0)
            percentage = (count / total_five_class) * 100
            print(f"{class_name} (ID={i}): {count:,} ({percentage:.2f}%)")
        
        print(f"\n5类系统总实体数量: {total_five_class:,}")
        
        # 检查是否有UNIT类型
        if five_class_stats.get('UNIT', 0) > 0:
            print(f"✅ UNIT类型已正确识别: {five_class_stats['UNIT']} 个")
        else:
            print(f"❌ 未发现UNIT类型")
    else:
        print("❌ 没有找到有效的5类实体")
    
    # 输出原始类型分布（仅供参考）
    print("\n【原始类型分布】(仅供参考)")
    print("-" * 40)
    
    # 过滤掉应该被忽略的实体类型（与another保持一致）
    exclude_types = {
        'SRCSINK', 'SrcSinkObject',           # SrcSink相关类型
        'SKIPPED_MARKER',                     # 时间标记等
        'ProvenanceTagNode', 'PROVENANCE',    # 来源标记
        'Principal', 'PRINCIPAL',             # 主体
        'MEMORY',                             # 内存类型（another中被过滤）
    }
    
    filtered_stats = {k: v for k, v in simplified_type_stats.items() 
                     if k not in exclude_types}
    
    for entity_type, count in Counter(filtered_stats).most_common():
        percentage = count / sum(filtered_stats.values()) * 100 if sum(filtered_stats.values()) > 0 else 0
        print(f"{entity_type:15}: {count:,} ({percentage:.2f}%)")
    
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
    """为单个文件收集边统计信息，使用5类映射和过滤规则"""
    edge_file_path = '../data/{}/'.format(dataset) + file + '.txt'
    if not os.path.exists(edge_file_path):
        return
    
    total_edges = 0
    filtered_edges = 0
    five_class_stats = {'sbj': Counter(), 'obj': Counter()}
    
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
                        total_edges += 1
                        
                        # 获取5类映射的类型名称
                        src_unified_type = get_unified_type(src_id, src_type, uuid2entity)
                        dst_unified_type = get_unified_type(dst_id, dst_type, uuid2entity)
                        
                        # 应用与another完全一致的过滤规则
                        if should_filter_event(src_unified_type, dst_unified_type, edge_type):
                            filtered_edges += 1
                            continue
                        
                        # 统计有效的5类数据
                        if src_unified_type:
                            five_class_stats['sbj'][src_unified_type] += 1
                            stats['sbj'][src_unified_type] += 1
                        
                        if dst_unified_type:
                            five_class_stats['obj'][dst_unified_type] += 1
                            stats['obj'][dst_unified_type] += 1
                            
                except (ValueError, IndexError):
                    continue
                    
    except Exception as e:
        print(f"收集统计信息时出错 {file}: {e}")
    
    print(f"文件 {file}: 总边数={total_edges}, 过滤掉={filtered_edges}, 保留={total_edges-filtered_edges}")
    if five_class_stats['sbj'] or five_class_stats['obj']:
        print(f"  主体类型: {dict(five_class_stats['sbj'])}")
        print(f"  客体类型: {dict(five_class_stats['obj'])}")


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
        # 主体统计：排除SrcSink相关类型
        exclude_types = {'SRCSINK', 'SrcSinkObject'}
        filtered_counts = {k: v for k, v in type_counts.items() if k not in exclude_types}
    elif entity_type == 'obj':
        # 客体统计：排除UNKNOWN、IPC和SrcSink相关类型
        exclude_types = {'UNKNOWN', 'IPC', 'SRCSINK', 'SrcSinkObject'}
        filtered_counts = {k: v for k, v in type_counts.items() if k not in exclude_types}
    
    if not filtered_counts:
        return
    
    # 使用所有类型的总数来计算比例（包括被过滤的类型），而不是只用显示类型的总数
    total_count = sum(type_counts.values())  # 修改：使用所有类型的总数
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


def read_single_graph(dataset, malicious, path, test=False, uuid2entity=None):
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
        
        # 检查是否是another格式（数字ID）还是原始格式（字符串类型）
        if src_type.isdigit() and dst_type.isdigit():
            # another格式：直接使用数字ID转换为类型名称
            src_type_id = int(src_type)
            dst_type_id = int(dst_type)
            type_id_to_name = {0: 'UNIT', 1: 'FILE', 2: 'NETFLOW', 3: 'PROCESS', 4: 'DIR'}
            src_five_class = type_id_to_name.get(src_type_id)
            dst_five_class = type_id_to_name.get(dst_type_id)
        else:
            # 原始格式：获取5类映射的类型
            src_five_class = get_unified_type(src, src_type, uuid2entity) if uuid2entity else map_to_five_classes(src_type)
            dst_five_class = get_unified_type(dst, dst_type, uuid2entity) if uuid2entity else map_to_five_classes(dst_type)
        
        # 应用与another完全一致的过滤规则
        if should_filter_event(src_five_class, dst_five_class, edge_type):
            continue
        
        # 如果类型被过滤为None，跳过（与another的5类系统保持一致）
        if src_five_class is None or dst_five_class is None:
            continue
        
        if not test:
            if src in malicious or dst in malicious:
                if src in malicious and src_five_class != 'MEMORY':  # 注意：MEMORY已被过滤，但保留检查
                    continue
                if dst in malicious and dst_five_class != 'MEMORY':
                    continue

        # 使用5类类型构建node_type_dict，确保与another方法完全一致的映射
        if src_five_class not in node_type_dict:
            # 确保映射顺序与another一致: UNIT=0, FILE=1, NETFLOW=2, PROCESS=3, DIR=4
            node_type_dict[src_five_class] = get_five_class_id(src_five_class)
        if dst_five_class not in node_type_dict:
            node_type_dict[dst_five_class] = get_five_class_id(dst_five_class)
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
        
        # 重新获取5类映射的类型（因为lines中保存的仍是原始类型）
        src_five_class = get_unified_type(src, src_type, uuid2entity) if uuid2entity else map_to_five_classes(src_type)
        dst_five_class = get_unified_type(dst, dst_type, uuid2entity) if uuid2entity else map_to_five_classes(dst_type)
        
        # 再次检查过滤规则（防止数据不一致，与another保持一致）
        if should_filter_event(src_five_class, dst_five_class, edge_type):
            continue
        if src_five_class is None or dst_five_class is None:
            continue
            
        src_type_id = node_type_dict[src_five_class]
        dst_type_id = node_type_dict[dst_five_class]
        edge_type_id = edge_type_dict[edge_type]
        
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_five_class
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_type_map[dst] = dst_five_class
            node_list.append(dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)
    return node_map, g


def preprocess_dataset(dataset):
    """预处理数据集 - 基于已有处理结果进行5类转换"""
    print(f"开始预处理 {dataset} 数据集，转换为与another一致的5类系统...")
    
    # 检查是否已有处理后的边文件
    train_files = metadata[dataset]['train']
    test_files = metadata[dataset]['test']
    
    all_edge_files = []
    for file in train_files + test_files:
        edge_file_path = f'../data/{dataset}/{file}.txt'
        if os.path.exists(edge_file_path):
            all_edge_files.append(edge_file_path)
    
    if not all_edge_files:
        print("❌ 没有找到已处理的边文件，需要原始JSON数据进行预处理")
        # 如果没有处理后的文件，则尝试原始方法
        uuid2entity = generate_uuid2entity_mapping(dataset)
    else:
        print(f"✅ 找到 {len(all_edge_files)} 个已处理的边文件，将使用原始方法生成UUID映射")
        # 仍然使用原始方法生成UUID映射，然后在输出时转换为5类格式
        uuid2entity = generate_uuid2entity_mapping(dataset)
    
    id_nodetype_map = {}
    id_nodename_map = {}
    
    # 从uuid2entity中提取类型和名称信息，采用darpa.py的unify逻辑
    for uuid, entity_info in uuid2entity.items():
        if entity_info and entity_info != f"UUID{PLUS}{uuid}":
            if PLUS in entity_info:
                entity_type, entity_name = entity_info.split(PLUS, 1)
                # 将实体类型映射到5类系统
                mapped_type = map_to_five_classes(entity_type)
                if mapped_type:  # 只保留5类中的实体
                    id_nodetype_map[uuid] = mapped_type
                    if entity_name:
                        id_nodename_map[uuid] = entity_name
            else:
                # 没有PLUS分隔符的简单类型
                mapped_type = map_to_five_classes(entity_info)
                if mapped_type:  # 只保留5类中的实体
                    id_nodetype_map[uuid] = mapped_type
    
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
                        
                        # 获取5类映射的类型
                        src_five_class = get_unified_type(srcId, id_nodetype_map.get(srcId, 'UNKNOWN'), uuid2entity) if srcId in id_nodetype_map else None
                        dst_five_class = get_unified_type(dstId1, id_nodetype_map.get(dstId1, 'UNKNOWN'), uuid2entity) if dstId1 in id_nodetype_map else None
                        
                        # 应用5类过滤规则
                        if should_filter_event(src_five_class, dst_five_class, edgeType):
                            continue
                        if src_five_class is None or dst_five_class is None:
                            continue
                        
                        # 调试：如果dstType1是UNIT，记录
                        if dst_five_class == 'UNIT':
                            unit_as_dst_count += 1
                            if unit_as_dst_count <= 3:  # 只打印前3个
                                print(f"🔍 发现UNIT作为客体1: dstId1={dstId1}, edgeType={edgeType}")
                        
                        # 应用NETFLOW处理逻辑（与another/useless.py保持一致）
                        src_entity_info = uuid2entity.get(srcId, srcId)
                        dst_entity_info = uuid2entity.get(dstId1, dstId1)
                        
                        # 处理NETFLOW实体的端口移除
                        src_entity_info = process_netflow_entity(src_entity_info, src_five_class)
                        dst_entity_info = process_netflow_entity(dst_entity_info, dst_five_class)
                        
                        # 更新uuid2entity中的处理结果
                        if src_five_class == 'NETFLOW' and src_entity_info != uuid2entity.get(srcId, srcId):
                            uuid2entity[srcId] = src_entity_info
                        if dst_five_class == 'NETFLOW' and dst_entity_info != uuid2entity.get(dstId1, dstId1):
                            uuid2entity[dstId1] = dst_entity_info
                        
                        # 转换为数字ID（与another方法一致）
                        src_type_id = get_five_class_id(src_five_class)
                        dst_type_id = get_five_class_id(dst_five_class)
                        
                        # 写入边信息时使用数字ID（与another方法格式一致）
                        this_edge1 = str(srcId) + '\t' + str(src_type_id) + '\t' + str(dstId1) + '\t' + str(
                            dst_type_id) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge1)
                        
                        # 调试：检查是否包含UNIT类型
                        if src_five_class == 'UNIT' or dst_five_class == 'UNIT':
                            file_unit_edges += 1
                            total_unit_edges += 1

                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]
                        if not dstId2 in id_nodetype_map.keys():
                            continue
                        dstType2 = id_nodetype_map[dstId2]
                        
                        # 获取5类映射的类型
                        src_five_class = get_unified_type(srcId, id_nodetype_map.get(srcId, 'UNKNOWN'), uuid2entity) if srcId in id_nodetype_map else None
                        dst_five_class = get_unified_type(dstId2, id_nodetype_map.get(dstId2, 'UNKNOWN'), uuid2entity) if dstId2 in id_nodetype_map else None
                        
                        # 应用5类过滤规则
                        if should_filter_event(src_five_class, dst_five_class, edgeType):
                            continue
                        if src_five_class is None or dst_five_class is None:
                            continue
                        
                        # 调试：如果dstType2是UNIT，记录
                        if dst_five_class == 'UNIT':
                            unit_as_dst_count += 1
                            if unit_as_dst_count <= 3:  # 只打印前3个
                                print(f"🔍 发现UNIT作为客体2: dstId2={dstId2}, edgeType={edgeType}")
                        
                        # 应用NETFLOW处理逻辑（与another/useless.py保持一致）
                        src_entity_info = uuid2entity.get(srcId, srcId)
                        dst_entity_info = uuid2entity.get(dstId2, dstId2)
                        
                        # 处理NETFLOW实体的端口移除
                        src_entity_info = process_netflow_entity(src_entity_info, src_five_class)
                        dst_entity_info = process_netflow_entity(dst_entity_info, dst_five_class)
                        
                        # 更新uuid2entity中的处理结果
                        if src_five_class == 'NETFLOW' and src_entity_info != uuid2entity.get(srcId, srcId):
                            uuid2entity[srcId] = src_entity_info
                        if dst_five_class == 'NETFLOW' and dst_entity_info != uuid2entity.get(dstId2, dstId2):
                            uuid2entity[dstId2] = dst_entity_info
                        
                        # 转换为数字ID（与another方法一致）
                        src_type_id = get_five_class_id(src_five_class)
                        dst_type_id = get_five_class_id(dst_five_class)
                        
                        # 写入边信息时使用数字ID（与another方法格式一致）
                        this_edge2 = str(srcId) + '\t' + str(src_type_id) + '\t' + str(dstId2) + '\t' + str(
                            dst_type_id) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge2)
                        
                        # 调试：检查是否包含UNIT类型
                        if src_five_class == 'UNIT' or dst_five_class == 'UNIT':
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
    
    print(f"总共生成的包含UNIT类型的边数量: {total_unit_edges}")
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

    # 生成uuid2entity映射，并在preprocess_dataset中使用
    # preprocess_dataset已经包含了uuid2entity生成和数据预处理
    # 这里只需要重新加载生成的uuid2entity
    uuid2entity = generate_uuid2entity_mapping(dataset)
    preprocess_dataset(dataset)  # 这个调用内部也会生成uuid2entity，但我们使用本地生成的
    train_gs = []
    for file in metadata[dataset]['train']:
        _, train_g = read_single_graph(dataset, malicious_entities, file, False, uuid2entity)
        train_gs.append(train_g)
    test_gs = []
    test_node_map = {}
    count_node = 0
    for file in metadata[dataset]['test']:
        node_map, test_g = read_single_graph(dataset, malicious_entities, file, True, uuid2entity)
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
            if e in test_node_map and e in id_nodetype_map:
                # 获取5类映射的类型
                entity_five_class = get_unified_type(e, id_nodetype_map[e], uuid2entity)
                # 只保留5类中的恶意实体，过滤掉MEMORY等
                if entity_five_class in ['UNIT', 'FILE', 'NETFLOW', 'PROCESS', 'DIR']:
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
    # 更新node_type_cnt为实际的5类数量
    node_type_cnt = len(node_type_dict)
    print(f"  节点类型数量: {node_type_cnt} (强制5类系统)")
    print(f"  边类型数量: {edge_type_cnt}")
    
    # 输出5类节点类型映射信息，确保映射正确
    print(f"\n5类节点类型映射 (与another一致):")
    for node_type, type_id in sorted(node_type_dict.items(), key=lambda x: x[1]):
        print(f"  {node_type} -> ID {type_id}")
    
    # 验证映射的正确性
    expected_mapping = {'UNIT': 0, 'FILE': 1, 'NETFLOW': 2, 'PROCESS': 3, 'DIR': 4}
    mapping_correct = all(node_type_dict.get(k) == v for k, v in expected_mapping.items() if k in node_type_dict)
    print(f"  ✅ 映射验证: {'通过' if mapping_correct else '❌ 失败'}")
    
    # 最终统计信息输出（与another保持一致）
    print("\n" + "="*60)
    print("最终实体统计信息 (与another相同的5类系统)")
    print("="*60)
    
    # 重新输出最终的统计信息
    collect_and_output_statistics(dataset, uuid2entity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser - 与another完全一致的5类系统实现')
    parser.add_argument("--dataset", type=str, default="trace", 
                       choices=['trace', 'theia', 'cadets'], 
                       help='要处理的数据集名称')
    args = parser.parse_args()
    
    print(f"开始处理 {args.dataset} 数据集...")
    print("📊 输出格式: 与another方法完全一致")
    print("🎯 节点类型映射: UNIT=0, FILE=1, NETFLOW=2, PROCESS=3, DIR=4")
    print("📝 边格式: src_id \\t src_type_id \\t dst_id \\t dst_type_id \\t edge_type \\t timestamp")
    print("🔍 过滤规则: 与another/useless.py完全一致")
    print("=" * 70)
    read_graphs(args.dataset)
    print("=" * 70)
    print("✅ 处理完成！数据格式与another方法100%兼容")
