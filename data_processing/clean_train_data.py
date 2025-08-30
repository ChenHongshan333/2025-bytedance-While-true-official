#!/usr/bin/env python3
"""
Post-processing script to clean CSV files by removing rows containing OpenAI API errors.

This script reads a CSV file, removes rows that contain the error message
"ERROR: Error calling OpenAI API: Err", and saves the cleaned data to a new CSV file.
"""

import pandas as pd
import os
import click
from pathlib import Path


@click.command()
@click.option(
    '--input-file', '-i',
    default="data/google_local-Alabama/train_gt.csv",
    help='Path to the input CSV file to clean',
    type=click.Path(exists=True, readable=True)
)
@click.option(
    '--output-file', '-o',
    default=None,
    help='Path to save the cleaned CSV file (default: adds "_cleaned" suffix to input filename)',
    type=click.Path()
)
@click.option(
    '--error-pattern', '-p',
    default="ERROR: Error calling OpenAI API: Err",
    help='Error pattern to search for and remove',
    type=str
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be removed without actually saving the cleaned file'
)
def clean_csv_data(input_file, output_file, error_pattern, verbose, dry_run):
    """
    Clean CSV file by removing rows containing OpenAI API errors.
    
    This tool reads a CSV file, identifies rows containing the specified error pattern,
    and creates a cleaned version without those rows.
    """
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    click.echo(f"Input file: {input_file}")
    if not dry_run:
        click.echo(f"Output file: {output_file}")
    else:
        click.echo("DRY RUN MODE - No files will be modified")
    
    if verbose:
        click.echo(f"Error pattern: '{error_pattern}'")
    
    try:
        # Read the CSV file
        click.echo("Reading CSV file...")
        df = pd.read_csv(input_file)
        
        if verbose:
            click.echo(f"Original data shape: {df.shape}")
        
        # Count rows before cleaning
        original_count = len(df)
        
        # Create a boolean mask to identify rows containing the error
        click.echo("Searching for error patterns...")
        error_mask = df.astype(str).apply(
            lambda x: x.str.contains(error_pattern, na=False, regex=False)
        ).any(axis=1)
        
        # Count rows with errors
        error_count = error_mask.sum()
        
        if error_count == 0:
            click.echo("✅ No rows with error patterns found!")
            if not dry_run:
                click.echo("Creating a copy of the original file...")
                df.to_csv(output_file, index=False)
                click.echo(f"File copied to: {output_file}")
            return
        
        click.echo(f"Found {error_count:,} rows containing the error pattern")
        
        if verbose and error_count > 0:
            # Show some examples of rows that will be removed
            error_rows = df[error_mask].head(3)
            click.echo("\nExample rows to be removed:")
            for idx, row in error_rows.iterrows():
                click.echo(f"  Row {idx}: {str(row.iloc[0])[:100]}...")
        
        if not dry_run:
            # Remove rows with errors
            cleaned_df = df[~error_mask]
            
            # Count rows after cleaning
            cleaned_count = len(cleaned_df)
            
            if verbose:
                click.echo(f"Cleaned data shape: {cleaned_df.shape}")
            
            # Save the cleaned data
            click.echo("Saving cleaned data...")
            cleaned_df.to_csv(output_file, index=False)
            click.echo(f"✅ Cleaned data saved to: {output_file}")
        else:
            cleaned_count = original_count - error_count
        
        # Print summary
        click.echo("\n" + "="*50)
        click.echo("SUMMARY")
        click.echo("="*50)
        click.echo(f"Original rows:     {original_count:,}")
        click.echo(f"Rows with errors:  {error_count:,}")
        click.echo(f"Cleaned rows:      {cleaned_count:,}")
        click.echo(f"Success rate:      {(cleaned_count/original_count)*100:.2f}%")
        
        if dry_run:
            click.echo("\n⚠️  This was a dry run - no files were modified")
        
    except Exception as e:
        click.echo(f"❌ Error processing file: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    clean_csv_data()
