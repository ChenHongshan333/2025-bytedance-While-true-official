#!/usr/bin/env python3
"""
Review Classification Benchmark Tool

Calculate accuracy, precision, recall, F1-score, and confusion matrix for review classification results.
Supports both multi-class (4 categories) and binary (valid/invalid) evaluation.
"""

import os
import click
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_results(gt_file_path, pred_file_path, gt_column='ai_classification', pred_column='ai_classification'):
    """Load ground truth and prediction results from separate files."""
    # Load ground truth
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file_path}")
    
    gt_df = pd.read_csv(gt_file_path)
    if gt_column not in gt_df.columns:
        raise ValueError(f"Ground truth column '{gt_column}' not found in {gt_file_path}")
    
    # Load predictions
    if not os.path.exists(pred_file_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_file_path}")
    
    pred_df = pd.read_csv(pred_file_path)
    if pred_column not in pred_df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found in {pred_file_path}")
    
    # Check that files have same number of rows
    if len(gt_df) != len(pred_df):
        raise ValueError(f"Row count mismatch: GT file has {len(gt_df)} rows, prediction file has {len(pred_df)} rows")
    
    # Filter out error predictions
    valid_mask = ~pred_df[pred_column].str.startswith('ERROR:', na=False)
    gt_clean = gt_df[valid_mask][gt_column].values
    pred_clean = pred_df[valid_mask][pred_column].values
    
    if len(gt_clean) == 0:
        raise ValueError("No valid predictions found (all are errors)")
    
    error_count = len(gt_df) - len(gt_clean)
    if error_count > 0:
        click.echo(f"⚠️  Filtered out {error_count} error predictions")
    
    return gt_clean, pred_clean


def calculate_multiclass_metrics(y_true, y_pred, labels=None):
    """Calculate multi-class classification metrics."""
    if labels is None:
        labels = ['valid', 'advertisement', 'irrelevant', 'rants without visit']
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)[0]
    recall_weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)[1]
    f1_weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)[2]
    
    # Macro averages
    precision_macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)[0]
    recall_macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)[1]
    f1_macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)[2]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'labels': labels
    }


def calculate_binary_metrics(y_true, y_pred):
    """Calculate binary classification metrics (valid vs invalid)."""
    # Convert to binary: valid vs invalid
    y_true_binary = np.array(['valid' if label == 'valid' else 'invalid' for label in y_true])
    y_pred_binary = np.array(['valid' if label == 'valid' else 'invalid' for label in y_pred])
    
    labels_binary = ['valid', 'invalid']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, support = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=labels_binary, average=None, zero_division=0)
    
    # Weighted and macro averages
    precision_weighted = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=labels_binary, average='weighted', zero_division=0)[0]
    recall_weighted = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=labels_binary, average='weighted', zero_division=0)[1]
    f1_weighted = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=labels_binary, average='weighted', zero_division=0)[2]
    
    precision_macro = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=labels_binary, average='macro', zero_division=0)[0]
    recall_macro = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=labels_binary, average='macro', zero_division=0)[1]
    f1_macro = precision_recall_fscore_support(y_true_binary, y_pred_binary, labels=labels_binary, average='macro', zero_division=0)[2]
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=labels_binary)
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'labels': labels_binary
    }


def plot_confusion_matrix(cm, labels, title, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_report(metrics, output_path, title):
    """Save detailed metrics report to JSON and text files."""
    # JSON report
    json_metrics = {
        'title': title,
        'accuracy': float(metrics['accuracy']),
        'precision_weighted': float(metrics['precision_weighted']),
        'recall_weighted': float(metrics['recall_weighted']),
        'f1_weighted': float(metrics['f1_weighted']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'per_class_metrics': {}
    }
    
    for i, label in enumerate(metrics['labels']):
        json_metrics['per_class_metrics'][label] = {
            'precision': float(metrics['precision_per_class'][i]),
            'recall': float(metrics['recall_per_class'][i]),
            'f1': float(metrics['f1_per_class'][i]),
            'support': int(metrics['support_per_class'][i])
        }
    
    # Save JSON
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    # Text report
    with open(output_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (weighted): {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (weighted): {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-score (weighted): {metrics['f1_weighted']:.4f}\n")
        f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-score (macro): {metrics['f1_macro']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        for i, label in enumerate(metrics['labels']):
            f.write(f"{label}:\n")
            f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
            f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
            f.write(f"  F1-score: {metrics['f1_per_class'][i]:.4f}\n")
            f.write(f"  Support: {metrics['support_per_class'][i]}\n\n")


@click.command()
@click.option(
    '--gt-file',
    required=True,
    type=click.Path(exists=True),
    help='Path to CSV file with ground truth labels'
)
@click.option(
    '--pred-file',
    required=True,
    type=click.Path(exists=True),
    help='Path to CSV file with prediction results'
)
@click.option(
    '--output-dir',
    required=True,
    help='Directory to save benchmark results'
)
@click.option(
    '--gt-column',
    default='ai_classification',
    help='Column name for ground truth (default: ai_classification)'
)
@click.option(
    '--pred-column',
    default='ai_classification',
    help='Column name for predictions (default: ai_classification)'
)
@click.option(
    '--template-name',
    default='unknown',
    help='Template name for report titles'
)
def main(gt_file, pred_file, output_dir, gt_column, pred_column, template_name):
    """
    Calculate benchmark metrics for review classification results.
    
    Generates comprehensive evaluation including:
    - Multi-class metrics (4 categories)
    - Binary metrics (valid vs invalid)
    - Confusion matrices
    - Detailed reports in JSON and text formats
    """
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        click.echo(f"📊 Loading ground truth from: {gt_file}")
        click.echo(f"🔮 Loading predictions from: {pred_file}")
        click.echo(f"📁 Output directory: {output_dir}")
        click.echo(f"🏷️  Template: {template_name}")
        click.echo("")
        
        # Load data
        y_true, y_pred = load_results(gt_file, pred_file, gt_column, pred_column)
        total_samples = len(y_true)
        
        click.echo(f"✅ Loaded {total_samples} valid predictions")
        
        # Multi-class evaluation
        click.echo("🔄 Calculating multi-class metrics...")
        multiclass_metrics = calculate_multiclass_metrics(y_true, y_pred)
        
        # Binary evaluation
        click.echo("🔄 Calculating binary metrics...")
        binary_metrics = calculate_binary_metrics(y_true, y_pred)
        
        # Save multi-class results
        multiclass_title = f"Multi-class Classification Results - {template_name}"
        multiclass_report_path = os.path.join(output_dir, f'{template_name}_multiclass_report.txt')
        save_metrics_report(multiclass_metrics, multiclass_report_path, multiclass_title)
        
        # Save binary results
        binary_title = f"Binary Classification Results (Valid vs Invalid) - {template_name}"
        binary_report_path = os.path.join(output_dir, f'{template_name}_binary_report.txt')
        save_metrics_report(binary_metrics, binary_report_path, binary_title)
        
        # Plot confusion matrices
        click.echo("📈 Generating confusion matrices...")
        
        # Multi-class confusion matrix
        multiclass_cm_path = os.path.join(output_dir, f'{template_name}_multiclass_confusion_matrix.png')
        plot_confusion_matrix(
            multiclass_metrics['confusion_matrix'], 
            multiclass_metrics['labels'],
            f'Multi-class Confusion Matrix - {template_name}',
            multiclass_cm_path
        )
        
        # Binary confusion matrix
        binary_cm_path = os.path.join(output_dir, f'{template_name}_binary_confusion_matrix.png')
        plot_confusion_matrix(
            binary_metrics['confusion_matrix'],
            binary_metrics['labels'], 
            f'Binary Confusion Matrix - {template_name}',
            binary_cm_path
        )
        
        # Display summary
        click.echo("\n" + "="*60)
        click.echo(f"📊 BENCHMARK RESULTS - {template_name.upper()}")
        click.echo("="*60)
        
        click.echo("\n🔢 MULTI-CLASS METRICS:")
        click.echo(f"   Accuracy: {multiclass_metrics['accuracy']:.4f}")
        click.echo(f"   Precision (weighted): {multiclass_metrics['precision_weighted']:.4f}")
        click.echo(f"   Recall (weighted): {multiclass_metrics['recall_weighted']:.4f}")
        click.echo(f"   F1-score (weighted): {multiclass_metrics['f1_weighted']:.4f}")
        
        click.echo("\n✅ BINARY METRICS (Valid vs Invalid):")
        click.echo(f"   Accuracy: {binary_metrics['accuracy']:.4f}")
        click.echo(f"   Precision (weighted): {binary_metrics['precision_weighted']:.4f}")
        click.echo(f"   Recall (weighted): {binary_metrics['recall_weighted']:.4f}")
        click.echo(f"   F1-score (weighted): {binary_metrics['f1_weighted']:.4f}")
        
        click.echo(f"\n📁 Results saved to: {output_dir}")
        click.echo("   - Multi-class report: JSON + TXT")
        click.echo("   - Binary report: JSON + TXT") 
        click.echo("   - Confusion matrices: PNG")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise


if __name__ == '__main__':
    main()
