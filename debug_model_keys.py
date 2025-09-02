#!/usr/bin/env python3
"""
调试模型键名不匹配问题的工具脚本
"""

import torch
import argparse
import yaml
from model.dit import DITModel

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def dict2namespace(config):
    """将字典转换为命名空间"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def debug_model_keys(config_path, checkpoint_path):
    """调试模型键名问题"""
    
    print("=== 调试模型键名不匹配问题 ===")
    print(f"配置文件: {config_path}")
    print(f"检查点文件: {checkpoint_path}")
    print()
    
    # 加载配置
    config_dict = load_config(config_path)
    config = dict2namespace(config_dict)
    
    # 创建模型
    print("创建DITModel...")
    model = DITModel(config)
    model_keys = list(model.state_dict().keys())
    
    print(f"模型期望的键数量: {len(model_keys)}")
    print("前10个模型键:")
    for i, key in enumerate(model_keys[:10]):
        print(f"  {i+1}. {key}")
    print()
    
    # 加载检查点
    print("加载检查点...")
    try:
        states = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(states, (list, tuple)):
            state_dict = states[0]
            print(f"检查点是列表/元组，使用第一个元素")
        else:
            state_dict = states
            print(f"检查点是字典")
            
        checkpoint_keys = list(state_dict.keys())
        print(f"检查点中的键数量: {len(checkpoint_keys)}")
        print("前10个检查点键:")
        for i, key in enumerate(checkpoint_keys[:10]):
            print(f"  {i+1}. {key}")
        print()
        
        # 分析键名差异
        print("=== 键名分析 ===")
        
        # 检查前缀
        model_has_dit_prefix = any(key.startswith('dit.') for key in model_keys)
        checkpoint_has_dit_prefix = any(key.startswith('dit.') for key in checkpoint_keys)
        
        print(f"模型键是否有'dit.'前缀: {model_has_dit_prefix}")
        print(f"检查点键是否有'dit.'前缀: {checkpoint_has_dit_prefix}")
        
        # 检查匹配情况
        exact_matches = set(model_keys) & set(checkpoint_keys)
        print(f"完全匹配的键数量: {len(exact_matches)}")
        
        # 尝试前缀转换
        if model_has_dit_prefix and not checkpoint_has_dit_prefix:
            print("\n尝试给检查点键添加'dit.'前缀...")
            converted_keys = [f"dit.{key}" for key in checkpoint_keys]
            matches_with_prefix = set(model_keys) & set(converted_keys)
            print(f"添加前缀后匹配的键数量: {len(matches_with_prefix)}")
            
        elif not model_has_dit_prefix and checkpoint_has_dit_prefix:
            print("\n尝试从检查点键移除'dit.'前缀...")
            converted_keys = [key[4:] if key.startswith('dit.') else key for key in checkpoint_keys]
            matches_without_prefix = set(model_keys) & set(converted_keys)
            print(f"移除前缀后匹配的键数量: {len(matches_without_prefix)}")
        
        # 显示不匹配的键
        missing_keys = set(model_keys) - set(checkpoint_keys)
        unexpected_keys = set(checkpoint_keys) - set(model_keys)
        
        if missing_keys:
            print(f"\n缺失的键 (前5个): {list(missing_keys)[:5]}")
        if unexpected_keys:
            print(f"意外的键 (前5个): {list(unexpected_keys)[:5]}")
            
        # 尝试实际加载
        print("\n=== 尝试加载模型 ===")
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("✓ 成功加载 (strict=False)")
            if missing:
                print(f"  缺失键数量: {len(missing)}")
            if unexpected:
                print(f"  意外键数量: {len(unexpected)}")
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            
    except Exception as e:
        print(f"加载检查点失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='调试模型键名不匹配问题')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='检查点文件路径')
    
    args = parser.parse_args()
    
    debug_model_keys(args.config, args.checkpoint)

if __name__ == '__main__':
    main()
