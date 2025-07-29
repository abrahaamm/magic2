#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的节点分类规则
"""

import sys
import os

# 直接定义重分类函数以便测试
def reclassify_node_type(original_type):
    """
    将原始节点类型重新分类到新的7个类别中：
    SUBJECT_PROCESS, SUBJECT_UNIT, FILE_OBJECT_FILE, NetFlowObject, 
    FILE_OBJECT_DIR, FILE_OBJECT_PROCESS, FILE_OBJECT_UNIT
    """
    if original_type == 'SUBJECT_PROCESS':
        return 'SUBJECT_PROCESS'
    elif original_type == 'SUBJECT_UNIT':
        return 'SUBJECT_UNIT'
    elif original_type in ['FILE_OBJECT_FILE', 'FileObject']:
        return 'FILE_OBJECT_FILE'
    elif original_type == 'NetFlowObject':
        return 'NetFlowObject'
    elif original_type in ['FILE_OBJECT_DIR', 'FILE_OBJECT_DIRECTORY']:
        return 'FILE_OBJECT_DIR'
    elif original_type in ['FILE_OBJECT_CHAR', 'FILE_OBJECT_BLOCK', 'FILE_OBJECT_LINK', 'FILE_OBJECT_UNIX_SOCKET']:
        # 将其他文件对象类型归类为 FILE_OBJECT_PROCESS
        return 'FILE_OBJECT_PROCESS'
    elif original_type in ['MemoryObject', 'UnnamedPipeObject', 'SRCSINK_UNKNOWN', 'PRINCIPAL_LOCAL']:
        # 将内存对象、管道对象等归类为 FILE_OBJECT_UNIT
        return 'FILE_OBJECT_UNIT'
    else:
        # 未知类型默认归类为 FILE_OBJECT_UNIT
        return 'FILE_OBJECT_UNIT'

def test_reclassification():
    """测试重分类函数"""
    
    # 定义测试用例：原始类型 -> 期望的新类型
    test_cases = {
        'SUBJECT_PROCESS': 'SUBJECT_PROCESS',
        'SUBJECT_UNIT': 'SUBJECT_UNIT',
        'FILE_OBJECT_FILE': 'FILE_OBJECT_FILE',
        'FileObject': 'FILE_OBJECT_FILE',
        'NetFlowObject': 'NetFlowObject',
        'FILE_OBJECT_DIR': 'FILE_OBJECT_DIR',
        'FILE_OBJECT_DIRECTORY': 'FILE_OBJECT_DIR',
        'FILE_OBJECT_CHAR': 'FILE_OBJECT_PROCESS',
        'FILE_OBJECT_BLOCK': 'FILE_OBJECT_PROCESS',
        'FILE_OBJECT_LINK': 'FILE_OBJECT_PROCESS',
        'FILE_OBJECT_UNIX_SOCKET': 'FILE_OBJECT_PROCESS',
        'MemoryObject': 'FILE_OBJECT_UNIT',
        'UnnamedPipeObject': 'FILE_OBJECT_UNIT',
        'SRCSINK_UNKNOWN': 'FILE_OBJECT_UNIT',
        'PRINCIPAL_LOCAL': 'FILE_OBJECT_UNIT',
    }
    
    print("测试节点重分类规则:")
    print("=" * 60)
    print(f"{'原始类型':<25} -> {'新类型':<20} {'结果'}")
    print("-" * 60)
    
    all_passed = True
    for original_type, expected_new_type in test_cases.items():
        actual_new_type = reclassify_node_type(original_type)
        status = "✓" if actual_new_type == expected_new_type else "✗"
        if actual_new_type != expected_new_type:
            all_passed = False
        print(f"{original_type:<25} -> {actual_new_type:<20} {status}")
    
    print("-" * 60)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    
    print("\n新的分类体系包含以下7个类别:")
    new_categories = set(test_cases.values())
    for i, category in enumerate(sorted(new_categories)):
        print(f"{i}: {category}")

if __name__ == '__main__':
    test_reclassification() 