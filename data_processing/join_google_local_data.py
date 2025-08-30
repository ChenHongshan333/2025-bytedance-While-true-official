#!/usr/bin/env python3
"""
联合meta-Alabama.json和review-Alabama.json文件，根据gmap_id进行匹配
将结果保存为CSV文件
"""

import json
import pandas as pd
from collections import defaultdict
import sys

def load_reviews(review_file):
    """加载评论数据，按gmap_id分组"""
    print("正在加载评论数据...")
    reviews_by_gmap = defaultdict(list)
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"已处理 {line_num} 行评论数据")
            
            try:
                review = json.loads(line.strip())
                gmap_id = review.get('gmap_id')
                if gmap_id:
                    reviews_by_gmap[gmap_id].append(review)
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line_num}")
                continue
    
    print(f"评论数据加载完成，共有 {len(reviews_by_gmap)} 个不同的商家")
    return reviews_by_gmap

def load_meta_and_join(meta_file, reviews_by_gmap, output_file):
    """加载商家元数据并与评论数据联合"""
    print("正在处理商家数据并联合评论...")
    
    joined_data = []
    processed_count = 0
    matched_count = 0
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"已处理 {line_num} 个商家")
            
            try:
                meta = json.loads(line.strip())
                gmap_id = meta.get('gmap_id')
                processed_count += 1
                
                if not gmap_id:
                    continue
                
                # 获取该商家的所有评论
                reviews = reviews_by_gmap.get(gmap_id, [])
                
                if reviews:
                    matched_count += 1
                    # 为每条评论创建一行数据，包含商家信息和评论信息
                    for review in reviews:
                        row_data = {}
                        
                        # 添加商家信息，前缀为 'meta_'
                        for key, value in meta.items():
                            if key != 'gmap_id':  # gmap_id单独处理
                                # 处理复杂数据类型
                                if isinstance(value, (dict, list)):
                                    row_data[f'meta_{key}'] = json.dumps(value, ensure_ascii=False)
                                else:
                                    row_data[f'meta_{key}'] = value
                        
                        # 添加评论信息，前缀为 'review_'
                        for key, value in review.items():
                            if key != 'gmap_id':  # gmap_id单独处理
                                # 处理复杂数据类型
                                if isinstance(value, (dict, list)):
                                    row_data[f'review_{key}'] = json.dumps(value, ensure_ascii=False)
                                else:
                                    row_data[f'review_{key}'] = value
                        
                        # 添加共同的gmap_id
                        row_data['gmap_id'] = gmap_id
                        
                        joined_data.append(row_data)
                        
            except json.JSONDecodeError:
                print(f"跳过无效的商家JSON行: {line_num}")
                continue
    
    print(f"商家数据处理完成:")
    print(f"- 总共处理了 {processed_count} 个商家")
    print(f"- 成功匹配了 {matched_count} 个商家")
    print(f"- 生成了 {len(joined_data)} 行联合数据")
    
    # 转换为DataFrame并保存为CSV
    if joined_data:
        df = pd.DataFrame(joined_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"数据已保存到: {output_file}")
        print(f"CSV文件包含 {len(df)} 行, {len(df.columns)} 列")
    else:
        print("没有找到匹配的数据!")

import click

@click.command()
@click.option('--meta-file', required=True, type=click.Path(exists=True), help='商家元数据文件路径 (meta json)')
@click.option('--review-file', required=True, type=click.Path(exists=True), help='评论数据文件路径 (review json)')
@click.option('--output-file', required=True, type=click.Path(), help='输出CSV文件路径')
def main(meta_file, review_file, output_file):
    try:
        # 首先加载所有评论数据到内存中，按gmap_id分组
        reviews_by_gmap = load_reviews(review_file)
        
        # 然后处理商家数据并联合
        load_meta_and_join(meta_file, reviews_by_gmap, output_file)
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
