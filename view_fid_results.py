#!/usr/bin/env python3
"""
FID结果查看器
用于查看和分析保存在CSV中的FID结果
"""

try:
    import pandas as pd
except ImportError:
    print("错误: 需要安装pandas库")
    print("请运行: pip install pandas")
    exit(1)

import argparse
import os

def view_results(csv_file, sort_by='timestamp', ascending=False, filter_architecture=None):
    """查看FID结果"""
    
    if not os.path.exists(csv_file):
        print(f"CSV文件不存在: {csv_file}")
        return
    
    try:
        # 读取CSV
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("CSV文件为空")
            return
        
        # 过滤架构
        if filter_architecture:
            df = df[df['architecture'].str.contains(filter_architecture, case=False, na=False)]
            if df.empty:
                print(f"没有找到架构包含 '{filter_architecture}' 的记录")
                return
        
        # 排序
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        print(f"\n=== FID结果总览 (共 {len(df)} 条记录) ===")
        print()
        
        # 显示主要信息
        display_columns = [
            'timestamp', 'fid_score', 'architecture', 'training_steps',
            'batch_size', 'resolution', 'dataset', 'computation_time_formatted'
        ]
        
        # 只显示存在的列
        available_columns = [col for col in display_columns if col in df.columns]
        
        # 设置显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print(df[available_columns].to_string(index=False))
        
        # 统计信息
        print(f"\n=== 统计信息 ===")
        print(f"FID分数范围: {df['fid_score'].min():.2f} - {df['fid_score'].max():.2f}")
        print(f"平均FID分数: {df['fid_score'].mean():.2f}")
        print(f"最佳FID分数: {df['fid_score'].min():.2f}")
        
        if 'architecture' in df.columns:
            print(f"架构分布: {dict(df['architecture'].value_counts())}")
        
        # 最佳结果
        best_idx = df['fid_score'].idxmin()
        best_result = df.loc[best_idx]
        print(f"\n=== 最佳结果 ===")
        print(f"FID分数: {best_result['fid_score']:.2f}")
        print(f"架构: {best_result['architecture']}")
        print(f"训练步数: {best_result['training_steps']}")
        print(f"时间: {best_result['timestamp']}")
        print(f"生成数据路径: {best_result['generated_data_path']}")
        
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='查看FID结果')
    parser.add_argument('--csv', default='data/datasheet.csv', help='CSV文件路径')
    parser.add_argument('--sort', default='fid_score', 
                       choices=['timestamp', 'fid_score', 'architecture', 'training_steps'],
                       help='排序字段')
    parser.add_argument('--ascending', action='store_true', help='升序排列')
    parser.add_argument('--filter-arch', help='过滤特定架构')
    
    args = parser.parse_args()
    
    view_results(args.csv, args.sort, args.ascending, args.filter_arch)

if __name__ == '__main__':
    main()
