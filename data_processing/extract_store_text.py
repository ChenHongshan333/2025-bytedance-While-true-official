#!/usr/bin/env python3
"""
Extract store information and text information from the joined CSV
Filter out rows where description or text is empty
"""

import pandas as pd
import json
from tqdm import tqdm

def is_empty_or_null(value):
    """
    Check if a value is empty, null, or meaningless
    """
    if pd.isna(value) or value is None:
        return True
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() in ['null', 'none', 'nan']:
            return True
    return False

def extract_store_and_text_info(input_file, output_file):
    """
    Extract store-related information and text information into two columns
    Filter out rows where description or text is empty
    """
    print(f"Loading data from {input_file}...")
    # Load with low_memory=False to avoid dtype warnings
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Loaded {len(df):,} rows")
    
    # Store-related information columns
    store_info_columns = [
        'meta_name',           # 商店名称
        'meta_address',        # 商店地址
        'meta_description',    # 商店描述
        'meta_category',       # 商店类别
        'meta_avg_rating',     # 平均评分
        'meta_num_of_reviews', # 评论数量
        'meta_price',          # 价格信息
        'meta_hours',          # 营业时间
        'meta_state',          # 商店状态
    ]
    
    # Text-related information columns
    text_info_columns = [
        'review_text',         # 评论文本
        'review_name',         # 评论者姓名
        'review_rating',       # 评论评分
        'review_time',         # 评论时间
        'review_resp',         # 商家回复
    ]
    
    print("Processing store and text information with filtering...")
    processed_data = []
    filtered_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Check if description or text is empty - if so, skip this row
        has_description = not is_empty_or_null(row.get('meta_description'))
        has_text = not is_empty_or_null(row.get('review_text'))
        
        if not has_description or not has_text:
            filtered_count += 1
            continue
        
        # Extract store information
        store_info = {}
        for col in store_info_columns:
            if col in df.columns and pd.notna(row[col]):
                store_info[col.replace('meta_', '')] = row[col]
        
        # Extract text information
        text_info = {}
        for col in text_info_columns:
            if col in df.columns and pd.notna(row[col]):
                if col == 'review_resp' and row[col] and row[col] != 'null':
                    # Parse review_resp if it's JSON
                    try:
                        resp_data = json.loads(row[col]) if isinstance(row[col], str) else row[col]
                        if isinstance(resp_data, dict):
                            text_info['response_text'] = resp_data.get('text', '')
                            text_info['response_time'] = resp_data.get('time', '')
                        else:
                            text_info['response_text'] = str(resp_data)
                    except:
                        text_info['response_text'] = str(row[col])
                else:
                    text_info[col.replace('review_', '')] = row[col]
        
        # Create combined store information string
        store_info_str = ""
        if store_info:
            store_parts = []
            if 'name' in store_info:
                store_parts.append(f"Store Name: {store_info['name']}")
            if 'address' in store_info:
                store_parts.append(f"Address: {store_info['address']}")
            if 'description' in store_info:
                store_parts.append(f"Description: {store_info['description']}")
            if 'category' in store_info:
                try:
                    # Parse category if it's JSON
                    category = json.loads(store_info['category']) if isinstance(store_info['category'], str) and store_info['category'].startswith('[') else store_info['category']
                    if isinstance(category, list):
                        store_parts.append(f"Category: {', '.join(category)}")
                    else:
                        store_parts.append(f"Category: {category}")
                except:
                    store_parts.append(f"Category: {store_info['category']}")
            if 'avg_rating' in store_info:
                store_parts.append(f"Average Rating: {store_info['avg_rating']}")
            if 'num_of_reviews' in store_info:
                store_parts.append(f"Number of Reviews: {store_info['num_of_reviews']}")
            if 'state' in store_info:
                store_parts.append(f"Status: {store_info['state']}")
            
            store_info_str = " | ".join(store_parts)
        
        # Create combined text information string
        text_info_str = ""
        if text_info:
            text_parts = []
            if 'text' in text_info and text_info['text']:
                text_parts.append(f"Review: {text_info['text']}")
            if 'name' in text_info:
                text_parts.append(f"Reviewer: {text_info['name']}")
            if 'rating' in text_info:
                text_parts.append(f"Rating: {text_info['rating']}")
            if 'response_text' in text_info and text_info['response_text']:
                text_parts.append(f"Store Response: {text_info['response_text']}")
            
            text_info_str = " | ".join(text_parts)
        
        # Add to processed data
        processed_data.append({
            'gmap_id': row.get('meta_gmap_id', ''),
            'store_info': store_info_str,
            'text_info': text_info_str
        })
    
    # Create DataFrame and save
    result_df = pd.DataFrame(processed_data)
    
    print(f"Saving processed data to {output_file}...")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Successfully saved {len(result_df):,} rows to {output_file}")
    print(f"Filtered out {filtered_count:,} rows due to empty description or text")
    
    # Print some statistics
    print("\n=== Statistics ===")
    print(f"Original records: {len(df):,}")
    print(f"Filtered out: {filtered_count:,}")
    print(f"Final records: {len(result_df):,}")
    print(f"Records with store info: {len(result_df[result_df['store_info'] != '']):,}")
    print(f"Records with text info: {len(result_df[result_df['text_info'] != '']):,}")
    print(f"Records with both: {len(result_df[(result_df['store_info'] != '') & (result_df['text_info'] != '')]):,}")
    
    # Show sample data
    print("\n=== Sample Data ===")
    print("First few rows:")
    for i, row in result_df.head(3).iterrows():
        print(f"\nRow {i+1}:")
        print(f"Store Info: {row['store_info'][:200]}...")
        print(f"Text Info: {row['text_info'][:200]}...")

import click

@click.command()
@click.option('--input-file', default="data/joined_meta_review_10w.csv", show_default=True, help="Input CSV file with joined meta and review data")
@click.option('--output-file', default="data/store_text_extracted_english.csv", show_default=True, help="Output CSV file for extracted store and text info")
def main(input_file, output_file):
    """Extract store and text information from joined meta-review CSV."""
    print("Starting store and text information extraction with filtering...")
    extract_store_and_text_info(input_file, output_file)
    print("Extraction completed!")

if __name__ == "__main__":
    main()
