#!/usr/bin/env python3
"""
Data Analysis Script for Google Local Alabama Dataset

This script analyzes the train_gt.csv and test_gt.csv files to provide comprehensive
statistics and insights about the review classification dataset.
"""

import pandas as pd
import numpy as np
import click
import re
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def extract_rating_from_text(text_info):
    """Extract rating from text_info column using regex."""
    if pd.isna(text_info):
        return None
    
    # Look for "Rating: X.X" pattern
    rating_match = re.search(r'Rating:\s*(\d+\.?\d*)', str(text_info))
    if rating_match:
        try:
            return float(rating_match.group(1))
        except:
            return None
    return None


def extract_store_category(store_info):
    """Extract store category from store_info column."""
    if pd.isna(store_info):
        return "Unknown"
    
    # Look for "Category:" pattern
    category_match = re.search(r'Category:\s*([^|]+)', str(store_info))
    if category_match:
        category = category_match.group(1).strip()
        # Get the first category if multiple categories exist
        first_category = category.split(',')[0].strip()
        return first_category
    return "Unknown"


def extract_store_rating(store_info):
    """Extract store average rating from store_info column."""
    if pd.isna(store_info):
        return None
    
    # Look for "Average Rating: X.X" pattern
    rating_match = re.search(r'Average Rating:\s*(\d+\.?\d*)', str(store_info))
    if rating_match:
        try:
            return float(rating_match.group(1))
        except:
            return None
    return None


def analyze_dataset(df, dataset_name):
    """Perform comprehensive analysis on a single dataset."""
    click.echo(f"\n{'='*60}")
    click.echo(f"📊 {dataset_name.upper()} DATASET ANALYSIS")
    click.echo(f"{'='*60}")
    
    # Basic statistics
    click.echo(f"\n🔢 Basic Statistics:")
    click.echo(f"   Total records: {len(df):,}")
    click.echo(f"   Columns: {list(df.columns)}")
    click.echo(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    click.echo(f"\n❌ Missing Values:")
    missing = df.isnull().sum()
    for col in missing.index:
        if missing[col] > 0:
            click.echo(f"   {col}: {missing[col]:,} ({missing[col]/len(df)*100:.2f}%)")
        else:
            click.echo(f"   {col}: 0 (0.00%)")
    
    # Classification distribution
    if 'ai_classification' in df.columns:
        click.echo(f"\n🏷️  Classification Distribution:")
        class_counts = df['ai_classification'].value_counts()
        for class_name, count in class_counts.items():
            percentage = count / len(df) * 100
            click.echo(f"   {class_name}: {count:,} ({percentage:.2f}%)")
    
    # Extract and analyze ratings
    df['extracted_rating'] = df['text_info'].apply(extract_rating_from_text)
    ratings_available = df['extracted_rating'].notna().sum()
    
    if ratings_available > 0:
        click.echo(f"\n⭐ Review Ratings Analysis:")
        click.echo(f"   Reviews with ratings: {ratings_available:,} ({ratings_available/len(df)*100:.2f}%)")
        click.echo(f"   Average rating: {df['extracted_rating'].mean():.2f}")
        click.echo(f"   Rating std dev: {df['extracted_rating'].std():.2f}")
        click.echo(f"   Rating distribution:")
        
        rating_dist = df['extracted_rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            percentage = count / ratings_available * 100
            click.echo(f"     {rating}: {count:,} ({percentage:.1f}%)")
    
    # Store category analysis
    df['store_category'] = df['store_info'].apply(extract_store_category)
    click.echo(f"\n🏪 Store Category Analysis:")
    category_counts = df['store_category'].value_counts().head(10)
    click.echo(f"   Top 10 store categories:")
    for i, (category, count) in enumerate(category_counts.items(), 1):
        percentage = count / len(df) * 100
        click.echo(f"     {i:2d}. {category}: {count:,} ({percentage:.1f}%)")
    
    # Store rating analysis
    df['store_avg_rating'] = df['store_info'].apply(extract_store_rating)
    store_ratings_available = df['store_avg_rating'].notna().sum()
    
    if store_ratings_available > 0:
        click.echo(f"\n🏬 Store Average Rating Analysis:")
        click.echo(f"   Stores with ratings: {store_ratings_available:,} ({store_ratings_available/len(df)*100:.2f}%)")
        click.echo(f"   Average store rating: {df['store_avg_rating'].mean():.2f}")
        click.echo(f"   Store rating std dev: {df['store_avg_rating'].std():.2f}")
    
    # Text length analysis
    df['text_length'] = df['text_info'].astype(str).str.len()
    click.echo(f"\n📝 Text Length Analysis:")
    click.echo(f"   Average text length: {df['text_length'].mean():.0f} characters")
    click.echo(f"   Median text length: {df['text_length'].median():.0f} characters")
    click.echo(f"   Min text length: {df['text_length'].min()}")
    click.echo(f"   Max text length: {df['text_length'].max()}")
    
    # Data quality issues
    click.echo(f"\n⚠️  Data Quality Issues:")
    
    # Check for potential errors
    error_patterns = [
        "ERROR:",
        "Error calling",
        "Rate limit",
        "API Error"
    ]
    
    total_errors = 0
    for pattern in error_patterns:
        error_count = df.astype(str).apply(
            lambda x: x.str.contains(pattern, na=False, case=False)
        ).any(axis=1).sum()
        if error_count > 0:
            click.echo(f"   Rows containing '{pattern}': {error_count:,}")
            total_errors += error_count
    
    if total_errors == 0:
        click.echo(f"   ✅ No obvious error patterns found")
    
    return df


def compare_datasets(train_df, test_df):
    """Compare train and test datasets."""
    click.echo(f"\n{'='*60}")
    click.echo(f"🔄 DATASET COMPARISON")
    click.echo(f"{'='*60}")
    
    # Size comparison
    click.echo(f"\n📏 Size Comparison:")
    click.echo(f"   Train set: {len(train_df):,} records")
    click.echo(f"   Test set:  {len(test_df):,} records")
    click.echo(f"   Ratio:     {len(train_df)/len(test_df):.2f}:1 (train:test)")
    
    # Classification distribution comparison
    if 'ai_classification' in train_df.columns and 'ai_classification' in test_df.columns:
        click.echo(f"\n🏷️  Classification Distribution Comparison:")
        train_dist = train_df['ai_classification'].value_counts(normalize=True) * 100
        test_dist = test_df['ai_classification'].value_counts(normalize=True) * 100
        
        all_classes = set(train_dist.index) | set(test_dist.index)
        click.echo(f"   {'Class':<15} {'Train %':<10} {'Test %':<10} {'Difference'}")
        click.echo(f"   {'-'*50}")
        
        for class_name in sorted(all_classes):
            train_pct = train_dist.get(class_name, 0)
            test_pct = test_dist.get(class_name, 0)
            diff = train_pct - test_pct
            click.echo(f"   {class_name:<15} {train_pct:>7.2f}%   {test_pct:>7.2f}%   {diff:>+7.2f}%")
    
    # Rating comparison
    train_ratings = train_df['extracted_rating'].dropna()
    test_ratings = test_df['extracted_rating'].dropna()
    
    if len(train_ratings) > 0 and len(test_ratings) > 0:
        click.echo(f"\n⭐ Rating Comparison:")
        click.echo(f"   Train avg rating: {train_ratings.mean():.2f}")
        click.echo(f"   Test avg rating:  {test_ratings.mean():.2f}")
        click.echo(f"   Difference:       {train_ratings.mean() - test_ratings.mean():+.2f}")
    
    # Category overlap
    train_categories = set(train_df['store_category'].unique())
    test_categories = set(test_df['store_category'].unique())
    
    overlap = train_categories & test_categories
    train_only = train_categories - test_categories
    test_only = test_categories - train_categories
    
    click.echo(f"\n🏪 Category Overlap:")
    click.echo(f"   Common categories: {len(overlap)}")
    click.echo(f"   Train-only categories: {len(train_only)}")
    click.echo(f"   Test-only categories: {len(test_only)}")
    
    if train_only:
        click.echo(f"   Train-only: {', '.join(list(train_only)[:5])}{'...' if len(train_only) > 5 else ''}")
    if test_only:
        click.echo(f"   Test-only: {', '.join(list(test_only)[:5])}{'...' if len(test_only) > 5 else ''}")


@click.command()
@click.argument(
    'csv_file',
    type=click.Path(exists=True, readable=True)
)
@click.option(
    '--compare-with',
    help='Optional: Path to a second CSV file for comparison',
    type=click.Path(exists=True, readable=True)
)
@click.option(
    '--detailed', '-d',
    is_flag=True,
    help='Show detailed analysis including examples'
)
@click.option(
    '--export-summary',
    help='Export analysis summary to a text file',
    type=click.Path()
)
def analyze_data(csv_file, compare_with, detailed, export_summary):
    """
    Comprehensive data analysis for CSV review classification dataset.
    
    This tool analyzes a single CSV file, providing statistics about
    classifications, ratings, store categories, and data quality.
    
    CSV_FILE: Path to the CSV file to analyze (required)
    
    Examples:
    
        # Analyze a single file
        python data_analysis.py ../data/google_local-Alabama/train_gt.csv
        
        # Analyze and compare two files
        python data_analysis.py ../data/google_local-Alabama/train_gt.csv --compare-with ../data/google_local-Alabama/test_gt.csv
        
        # Detailed analysis
        python data_analysis.py ../data/google_local-Alabama/train_gt.csv --detailed
    """
    
    click.echo("🚀 Starting Data Analysis...")
    click.echo(f"📁 Primary file: {csv_file}")
    
    if compare_with:
        click.echo(f"📁 Comparison file: {compare_with}")
    
    try:
        # Load primary dataset
        click.echo("\n📖 Loading primary dataset...")
        primary_df = pd.read_csv(csv_file)
        
        # Get filename for display
        primary_name = Path(csv_file).stem.upper().replace('_', ' ')
        
        # Analyze primary dataset
        primary_df = analyze_dataset(primary_df, primary_name)
        
        total_records = len(primary_df)
        
        # If comparison file provided, load and analyze it
        if compare_with:
            click.echo("\n📖 Loading comparison dataset...")
            comparison_df = pd.read_csv(compare_with)
            
            # Get filename for display
            comparison_name = Path(compare_with).stem.upper().replace('_', ' ')
            
            # Analyze comparison dataset
            comparison_df = analyze_dataset(comparison_df, comparison_name)
            
            # Compare datasets
            compare_datasets(primary_df, comparison_df)
            
            total_records += len(comparison_df)
        
        # Summary
        click.echo(f"\n{'='*60}")
        click.echo(f"✅ ANALYSIS COMPLETE")
        click.echo(f"{'='*60}")
        
        click.echo(f"📊 Total records analyzed: {total_records:,}")
        if compare_with:
            click.echo(f"📁 Files analyzed: 2")
        else:
            click.echo(f"📁 Files analyzed: 1")
        click.echo(f"🔍 Analysis completed successfully!")
        
        if export_summary:
            click.echo(f"\n💾 Exporting summary to: {export_summary}")
            # Note: In a real implementation, you would capture the output and save it
            click.echo("   (Export functionality would be implemented here)")
        
    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    analyze_data()
